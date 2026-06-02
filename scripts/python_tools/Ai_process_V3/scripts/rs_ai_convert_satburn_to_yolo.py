from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CLASS_NAMES = ["burned_area"]
RASTER_EXTS = {".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg", ".bmp"}
LABEL_HINTS = ("label", "mask", "burn", "burned", "burnt", "scar", "severity", "groundtruth", "gt")
IMAGE_EXCLUDE_HINTS = ("label", "mask", "groundtruth", "gt", "severity")


@dataclass
class SplitStats:
    split: str
    labels: int = 0
    images: int = 0
    boxes: int = 0
    masks: int = 0
    skipped: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Satellite Burned Area Dataset masks to YOLO boxes and ROI masks."
    )
    parser.add_argument("--burn-root", required=True, help="Extracted Satellite Burned Area Dataset root.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sensor", default="s2", choices=["s2", "s1", "auto"])
    parser.add_argument("--copy-images", action="store_true", help="Copy RGB images when possible.")
    parser.add_argument("--min-area", type=int, default=64, help="Minimum burned component area in pixels.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick smoke conversion.")
    args = parser.parse_args()

    root = Path(args.burn_root)
    output_root = Path(args.output_root)
    if not root.exists():
        raise FileNotFoundError(root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_map = _load_split_map(root)
    label_files = _find_label_files(root)
    if args.max_samples > 0:
        label_files = label_files[: args.max_samples]
    if not label_files:
        raise FileNotFoundError(
            f"No burned-area mask/label rasters found under {root}. "
            "Expected files containing label, mask, burn, burned, scar, severity, gt, or groundtruth."
        )

    image_index = _build_image_index(root)
    stats: dict[str, SplitStats] = {}
    for label_path in label_files:
        split = _infer_split(label_path, root, split_map)
        stats.setdefault(split, SplitStats(split=split))
        _convert_sample(
            label_path=label_path,
            root=root,
            output_root=output_root,
            image_index=image_index,
            split=split,
            sensor=args.sensor,
            min_area=args.min_area,
            copy_images=args.copy_images,
            stats=stats[split],
        )

    if "train" not in stats and stats:
        first = sorted(stats)[0]
        stats["train"] = stats.pop(first)
        stats["train"].split = "train"
    val_split = "val" if "val" in stats else ("test" if "test" in stats else "train")

    yaml_path = output_root / "satburn_burned_area.yaml"
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
        "dataset": "Satellite Burned Area Dataset",
        "task": "wildfire burned-area detection and ROI generation",
        "yaml": str(yaml_path),
        "output_root": str(output_root),
        "classes": CLASS_NAMES,
        "sensor": args.sensor,
        "splits": [stats[k].__dict__ for k in sorted(stats)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _find_label_files(root: Path) -> list[Path]:
    labels: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in RASTER_EXTS:
            continue
        text = _path_text(path)
        if not any(hint in text for hint in LABEL_HINTS):
            continue
        if any(token in text for token in ("s1", "s2", "sentinel-1", "sentinel-2")) and not any(
            token in text for token in ("label", "mask", "groundtruth", "gt", "severity")
        ):
            continue
        labels.append(path)
    return sorted(labels)


def _build_image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in RASTER_EXTS:
            continue
        text = _path_text(path)
        if any(hint in text for hint in IMAGE_EXCLUDE_HINTS):
            continue
        for key in _image_keys(path, root):
            index.setdefault(key, []).append(path)
    return index


def _load_split_map(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for csv_path in root.rglob("*.csv"):
        try:
            rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()))
        except OSError:
            continue
        for row in rows:
            split = _split_from_row(row)
            if split is None:
                continue
            for value in row.values():
                for stem in _candidate_stems(str(value)):
                    mapping[stem] = split
    return mapping


def _split_from_row(row: dict[str, str]) -> str | None:
    for key, value in row.items():
        if key and key.lower() in {"fold", "split", "set", "subset"}:
            text = str(value).strip().lower()
            if text in {"train", "training", "0"}:
                return "train"
            if text in {"val", "valid", "validation", "1"}:
                return "val"
            if text in {"test", "testing", "2"}:
                return "test"
    for value in row.values():
        split = _split_from_text(str(value))
        if split is not None:
            return split
    return None


def _infer_split(path: Path, root: Path, split_map: dict[str, str]) -> str:
    for stem in _image_keys(path, root) + _candidate_stems(path.name):
        if stem in split_map:
            return split_map[stem]
    rel = _path_text(path.relative_to(root))
    split = _split_from_text(rel)
    if split is not None:
        return split
    bucket = int(hashlib.sha1(_sample_key(path).encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _convert_sample(
    label_path: Path,
    root: Path,
    output_root: Path,
    image_index: dict[str, list[Path]],
    split: str,
    sensor: str,
    min_area: int,
    copy_images: bool,
    stats: SplitStats,
) -> None:
    mask_arr = _read_raster(label_path)
    mask = _to_burn_mask(mask_arr)
    if mask.ndim != 2 or mask.size == 0:
        stats.skipped += 1
        return

    image_path = _find_matching_image(label_path, root, image_index, sensor)
    if image_path is None:
        stats.skipped += 1
        return

    out_image_dir = output_root / "images" / split
    out_label_dir = output_root / "labels" / split
    out_mask_dir = output_root / "masks" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    out_stem = _sample_key(label_path)
    out_image = out_image_dir / f"{out_stem}.png"
    out_label = out_label_dir / f"{out_stem}.txt"
    out_mask = out_mask_dir / f"{out_stem}.png"

    if not out_image.exists():
        _write_preview_png(image_path, out_image, sensor=sensor, copy_images=copy_images)
    _write_mask_png(mask, out_mask)

    h, w = mask.shape
    labels = []
    for x1, y1, x2, y2 in _mask_to_boxes(mask, min_area=min_area):
        bw = x2 - x1
        bh = y2 - y1
        xc = x1 + bw * 0.5
        yc = y1 + bh * 0.5
        labels.append(f"0 {_clip01(xc / w):.8f} {_clip01(yc / h):.8f} {_clip01(bw / w):.8f} {_clip01(bh / h):.8f}")
    out_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

    stats.labels += 1
    stats.images += 1
    stats.masks += 1
    stats.boxes += len(labels)


def _find_matching_image(label_path: Path, root: Path, image_index: dict[str, list[Path]], sensor: str) -> Path | None:
    candidates: list[Path] = []
    for key in _image_keys(label_path, root) + _candidate_stems(label_path.name):
        candidates.extend(image_index.get(key, []))
    for parent in [label_path.parent, *label_path.parents[:3]]:
        if root not in [parent, *parent.parents] and parent != root:
            continue
        try:
            near = [p for p in parent.iterdir() if p.is_file() and p.suffix.lower() in RASTER_EXTS]
        except OSError:
            continue
        candidates.extend(p for p in near if not any(h in _path_text(p) for h in IMAGE_EXCLUDE_HINTS))
    candidates = [p for p in candidates if p != label_path]
    if not candidates:
        return None
    return sorted(set(candidates), key=lambda p: _sensor_rank(p, sensor))[0]


def _sensor_rank(path: Path, sensor: str) -> tuple[int, int, str]:
    text = _path_text(path)
    if sensor == "s2":
        return (0 if any(t in text for t in ("s2", "sentinel-2", "sentinel2", "opt")) else 1, _rgb_rank(text), str(path))
    if sensor == "s1":
        return (0 if any(t in text for t in ("s1", "sentinel-1", "sentinel1", "sar")) else 1, _rgb_rank(text), str(path))
    return (
        0 if any(t in text for t in ("s2", "sentinel-2", "sentinel2", "opt")) else (1 if any(t in text for t in ("s1", "sentinel-1", "sentinel1", "sar")) else 2),
        _rgb_rank(text),
        str(path),
    )


def _rgb_rank(text: str) -> int:
    if any(t in text for t in ("rgb", "truecolor", "true_color", "b04", "b03", "b02")):
        return 0
    return 1


def _read_raster(path: Path) -> np.ndarray:
    try:
        import rasterio

        with rasterio.open(path) as src:
            return src.read()
    except Exception:
        pass
    try:
        import tifffile

        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] <= 32:
            return arr
        if arr.ndim == 3:
            return np.moveaxis(arr, -1, 0)
        return arr
    except Exception:
        pass
    from PIL import Image

    return np.asarray(Image.open(path))


def _to_burn_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] <= 8:
            arr = arr[0]
        else:
            arr = arr[..., 0]
    arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    valid = arr[arr != arr.min()] if arr.min() < 0 else arr
    if valid.size == 0:
        valid = arr
    unique = np.unique(valid)
    if unique.size <= 16:
        positive = arr > 0
    else:
        positive = arr > 0.5 if arr.max() <= 1.5 else arr >= np.percentile(valid, 75)
    return positive.astype(np.uint8)


def _write_preview_png(image_path: Path, out_path: Path, sensor: str, copy_images: bool) -> None:
    if copy_images and image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        shutil.copy2(image_path, out_path)
        return
    arr = _read_raster(image_path)
    rgb = _raster_to_rgb(arr, sensor=sensor)
    from PIL import Image

    Image.fromarray(rgb).save(out_path)


def _raster_to_rgb(arr: np.ndarray, sensor: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] <= 32:
        bands = arr
        if sensor == "s1" or bands.shape[0] == 2:
            b0 = bands[0]
            b1 = bands[min(1, bands.shape[0] - 1)]
            rgb = np.stack([b0, b1, 0.5 * (b0 + b1)], axis=-1)
        elif bands.shape[0] >= 12:
            rgb = np.stack([bands[3], bands[2], bands[1]], axis=-1)
        elif bands.shape[0] >= 4:
            rgb = np.stack([bands[3], bands[2], bands[1]], axis=-1)
        else:
            rgb = np.stack([bands[0], bands[min(1, bands.shape[0] - 1)], bands[min(2, bands.shape[0] - 1)]], axis=-1)
    elif arr.ndim == 3:
        rgb = arr[..., :3]
    else:
        raise ValueError(f"unsupported raster shape: {arr.shape}")
    return _normalize_u8(rgb)


def _normalize_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    for ch in range(arr.shape[-1]):
        band = arr[..., ch]
        valid = np.isfinite(band)
        if not valid.any():
            continue
        lo, hi = np.percentile(band[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[..., ch] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


def _write_mask_png(mask: np.ndarray, out_path: Path) -> None:
    from PIL import Image

    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)


def _mask_to_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    try:
        import cv2

        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        boxes: list[tuple[int, int, int, int]] = []
        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            if int(area) >= min_area and w > 1 and h > 1:
                boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes
    except Exception:
        ys, xs = np.where(mask > 0)
        if len(xs) < min_area:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))]


def _image_keys(path: Path, root: Path) -> list[str]:
    keys = set(_candidate_stems(path.name))
    try:
        rel = path.relative_to(root)
        parts = list(rel.parts)
    except ValueError:
        parts = list(path.parts)
    for part in parts[:-1]:
        if not any(hint in part.lower() for hint in ("s1", "s2", "sentinel", "label", "mask", "image")):
            keys.add(_clean_key(part))
    if len(parts) >= 2:
        keys.add(_clean_key(parts[-2]))
    return [k for k in keys if k]


def _candidate_stems(value: str) -> list[str]:
    path = Path(str(value).strip().strip('"').strip("'"))
    stem = path.stem if path.suffix else path.name
    stems = {stem, _sample_key(stem)}
    return [s for s in {_clean_key(s) for s in stems} if s]


def _sample_key(path_or_name: Path | str) -> str:
    name = path_or_name.stem if isinstance(path_or_name, Path) else Path(str(path_or_name)).stem
    return _clean_key(
        re.sub(
            r"(_?(s1|s2|sentinel1|sentinel2|sentinel-1|sentinel-2|label|labels|mask|burned|burn|scar|severity|groundtruth|gt|rgb|truecolor|true_color|post|pre))*$",
            "",
            name,
            flags=re.IGNORECASE,
        )
    )


def _clean_key(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\.[a-z0-9]+$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _path_text(path: Path) -> str:
    return "/".join(part.lower() for part in path.parts)


def _split_from_text(text: str) -> str | None:
    text = text.lower()
    if "train" in text or "/fold0/" in text:
        return "train"
    if "val" in text or "valid" in text or "/fold1/" in text:
        return "val"
    if "test" in text or "/fold2/" in text:
        return "test"
    return None


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


if __name__ == "__main__":
    raise SystemExit(main())
