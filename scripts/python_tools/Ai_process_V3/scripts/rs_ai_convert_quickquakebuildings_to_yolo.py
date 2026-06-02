#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["intact_building", "damaged_building"]
CLASS_DIRS = {
    "intact": 0,
    "undamaged": 0,
    "non_damage": 0,
    "non-damage": 0,
    "damaged": 1,
    "damage": 1,
    "collapsed": 1,
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class SplitStats:
    split: str
    images: int = 0
    boxes: int = 0
    skipped: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert QuickQuakeBuildings patch data to a YOLO detection dataset."
    )
    parser.add_argument("--qqb-root", required=True, help="QuickQuakeBuildings dataset root.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", default="opt", choices=["opt", "sar", "all"], help="Input channel rendered to RGB.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--copy-images", action="store_true", help="Copy ordinary images if they are already RGB files.")
    args = parser.parse_args()

    root = Path(args.qqb_root)
    output_root = Path(args.output_root)
    if not root.exists():
        raise FileNotFoundError(root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = _find_samples(root, mode=args.mode)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise FileNotFoundError(
            f"No QuickQuakeBuildings samples found under {root}. Expected class folders such as damaged/ and intact/."
        )

    stats: dict[str, SplitStats] = {}
    for sample in samples:
        split = _split_for_sample(sample, args.val_ratio, args.test_ratio)
        stats.setdefault(split, SplitStats(split=split))
        _convert_sample(sample, output_root, split, mode=args.mode, copy_images=args.copy_images, stats=stats[split])

    val_split = "val" if "val" in stats else "train"
    yaml_path = output_root / "quickquakebuildings_damage.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root}",
                "train: images/train",
                f"val: images/{val_split}",
                f"test: images/{'test' if 'test' in stats else val_split}",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = {
        "dataset": "QuickQuakeBuildings",
        "task": "earthquake building damage patch detection",
        "yaml": str(yaml_path),
        "output_root": str(output_root),
        "classes": CLASS_NAMES,
        "mode": args.mode,
        "splits": [stats[k].__dict__ for k in sorted(stats)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _find_samples(root: Path, mode: str) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS | {".mat", ".npy", ".npz"}:
            continue
        class_id = _infer_class(path, root)
        if class_id is None:
            continue
        lower = path.stem.lower()
        if mode == "opt" and ("sar" in lower and "opt" not in lower):
            continue
        if mode == "sar" and "opt" in lower:
            continue
        if mode != "all" and _looks_like_auxiliary(path):
            continue
        samples.append({"path": path, "class_id": class_id})
    return _deduplicate_samples(samples)


def _deduplicate_samples(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[tuple[int, str], dict[str, object]] = {}
    for sample in samples:
        path = sample["path"]
        assert isinstance(path, Path)
        class_id = int(sample["class_id"])
        key = (class_id, _base_stem(path))
        old = best.get(key)
        if old is None or _sample_priority(path) < _sample_priority(old["path"]):  # type: ignore[arg-type]
            best[key] = sample
    return [best[k] for k in sorted(best)]


def _sample_priority(path: Path) -> int:
    lower = path.stem.lower()
    if "opt" in lower and "ftp" not in lower:
        return 0
    if "sar" in lower and "ftp" not in lower:
        return 1
    if "ftp" in lower:
        return 2
    return 3


def _looks_like_auxiliary(path: Path) -> bool:
    lower = path.stem.lower()
    return any(token in lower for token in ["mask", "label", "gt", "footprint", "json"])


def _base_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ["_SARftp", "_sarftp", "_SAR", "_sar", "_optftp", "_OPTftp", "_opt", "_OPT"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _infer_class(path: Path, root: Path) -> int | None:
    rel_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    for part in reversed(rel_parts):
        if part in CLASS_DIRS:
            return CLASS_DIRS[part]
    lower = str(path).lower()
    if "damaged" in lower or "collapsed" in lower:
        return 1
    if "intact" in lower or "undamaged" in lower:
        return 0
    return None


def _split_for_sample(sample: dict[str, object], val_ratio: float, test_ratio: float) -> str:
    path = sample["path"]
    assert isinstance(path, Path)
    key = f"{sample['class_id']}:{_base_stem(path)}"
    frac = (int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 10000) / 10000.0
    if frac < test_ratio:
        return "test"
    if frac < test_ratio + val_ratio:
        return "val"
    return "train"


def _convert_sample(
    sample: dict[str, object],
    output_root: Path,
    split: str,
    mode: str,
    copy_images: bool,
    stats: SplitStats,
) -> None:
    path = sample["path"]
    assert isinstance(path, Path)
    class_id = int(sample["class_id"])
    try:
        image = _read_as_rgb(path, mode=mode)
    except Exception:
        stats.skipped += 1
        return
    h, w = image.shape[:2]
    if h < 4 or w < 4:
        stats.skipped += 1
        return

    out_stem = f"{path.parent.name}_{_base_stem(path)}"
    out_image_dir = output_root / "images" / split
    out_label_dir = output_root / "labels" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_image = out_image_dir / f"{out_stem}.png"
    out_label = out_label_dir / f"{out_stem}.txt"

    if copy_images and path.suffix.lower() in IMAGE_EXTS and mode != "all":
        shutil.copy2(path, out_image)
    else:
        cv2.imwrite(str(out_image), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    margin = 0.04
    xc = 0.5
    yc = 0.5
    bw = 1.0 - 2.0 * margin
    bh = 1.0 - 2.0 * margin
    out_label.write_text(f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")
    stats.images += 1
    stats.boxes += 1


def _read_as_rgb(path: Path, mode: str) -> np.ndarray:
    if path.suffix.lower() == ".mat":
        arr = _read_mat(path)
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        arr = data[sorted(data.files)[0]]
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise RuntimeError(f"failed to read image: {path}")
        if bgr.ndim == 2:
            arr = bgr
        else:
            arr = cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2RGB)
    return _array_to_rgb(arr, mode=mode)


def _read_mat(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat

        data = loadmat(path)
        candidates = []
        for key, value in data.items():
            if key.startswith("__") or not isinstance(value, np.ndarray):
                continue
            if value.ndim in {2, 3} and value.size > 64:
                candidates.append(value)
        if candidates:
            candidates.sort(key=lambda arr: (arr.ndim != 3, -arr.size))
            return np.asarray(candidates[0])
    except NotImplementedError:
        pass

    import h5py

    candidates: list[np.ndarray] = []
    with h5py.File(path, "r") as h5:
        def visit(_name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                arr = np.asarray(obj)
                if arr.ndim in {2, 3} and arr.size > 64 and arr.dtype.kind in {"u", "i", "f"}:
                    candidates.append(arr)

        h5.visititems(visit)
    if not candidates:
        raise RuntimeError(f"no numeric image array found in {path}")
    candidates.sort(key=lambda arr: (arr.ndim != 3, -arr.size))
    arr = candidates[0]
    if arr.ndim == 3:
        arr = np.transpose(arr)
    else:
        arr = arr.T
    return np.asarray(arr)


def _array_to_rgb(arr: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[-1] > 8:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] >= 3:
            rgb = arr[..., :3]
        elif arr.shape[-1] == 2:
            rgb = np.stack([arr[..., 0], arr[..., 1], 0.5 * (arr[..., 0] + arr[..., 1])], axis=-1)
        else:
            rgb = np.repeat(arr[..., :1], 3, axis=-1)
    else:
        raise ValueError(f"unsupported array shape: {arr.shape}")
    return _normalize_u8(rgb)


def _normalize_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    out = np.zeros_like(arr, dtype=np.float32)
    for ch in range(arr.shape[-1]):
        band = arr[..., ch]
        lo, hi = np.percentile(band, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[..., ch] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
