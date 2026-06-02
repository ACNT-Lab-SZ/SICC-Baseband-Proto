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


CLASS_NAMES = ["flood_water"]
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


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
        description="Convert Sen1Floods11 Sentinel flood labels to YOLO boxes and ROI masks."
    )
    parser.add_argument("--sen1-root", required=True, help="Downloaded Sen1Floods11 root.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sensor", default="s2", choices=["s2", "s1", "auto"])
    parser.add_argument("--copy-images", action="store_true", help="Copy generated PNGs instead of symlinking originals.")
    parser.add_argument("--include-weak", action="store_true", help="Also use WeaklyLabeled samples. Default uses hand labels.")
    parser.add_argument("--min-area", type=int, default=64, help="Minimum connected flood component area in pixels.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick smoke conversion.")
    args = parser.parse_args()

    root = Path(args.sen1_root)
    output_root = Path(args.output_root)
    if not root.exists():
        raise FileNotFoundError(root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_map = _load_split_map(root)
    label_files = _find_label_files(root, include_weak=args.include_weak)
    if args.max_samples > 0:
        label_files = label_files[: args.max_samples]
    if not label_files:
        raise FileNotFoundError(f"No Sen1Floods11 label tif files found under {root}")

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

    yaml_path = output_root / "sen1floods11_flood.yaml"
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
        "dataset": "Sen1Floods11",
        "task": "satellite flood-water detection and ROI generation",
        "yaml": str(yaml_path),
        "output_root": str(output_root),
        "classes": CLASS_NAMES,
        "sensor": args.sensor,
        "splits": [stats[k].__dict__ for k in sorted(stats)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _find_label_files(root: Path, include_weak: bool) -> list[Path]:
    labels: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff", ".png"}:
            continue
        lower = "/".join(p.lower() for p in path.parts)
        is_label = "label" in lower or path.stem.lower().endswith("_class")
        if not is_label:
            continue
        if not include_weak and "weak" in lower:
            continue
        labels.append(path)
    return sorted(labels)


def _build_image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        lower = "/".join(p.lower() for p in path.parts)
        if "label" in lower or path.stem.lower().endswith("_class"):
            continue
        index.setdefault(_sample_key(path), []).append(path)
    return index


def _load_split_map(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for csv_path in root.rglob("*.csv"):
        split = _split_from_name(csv_path.name)
        if split is None:
            continue
        try:
            rows = csv.reader(csv_path.read_text(encoding="utf-8", errors="ignore").splitlines())
        except OSError:
            continue
        for row in rows:
            for token in row:
                for stem in _candidate_stems(token):
                    mapping[stem] = split
    return mapping


def _candidate_stems(value: str) -> Iterable[str]:
    value = value.strip().strip('"').strip("'")
    if not value or value.lower() in {"image", "label", "s1", "s2"}:
        return []
    path = Path(value)
    stem = path.stem if path.suffix else path.name
    stems = {stem}
    stems.add(re.sub(r"_(s1|s2|label|labels)$", "", stem, flags=re.IGNORECASE))
    stems.add(_strip_sensor_suffix(stem))
    return [s for s in stems if s]


def _split_from_name(name: str) -> str | None:
    lower = name.lower()
    if "train" in lower:
        return "train"
    if "val" in lower or "valid" in lower or "hold" in lower:
        return "val"
    if "test" in lower:
        return "test"
    return None


def _infer_split(path: Path, root: Path, split_map: dict[str, str]) -> str:
    for stem in {_sample_key(path), *_candidate_stems(path.name)}:
        if stem in split_map:
            return split_map[stem]
    rel = "/".join(p.lower() for p in path.relative_to(root).parts)
    if "train" in rel:
        return "train"
    if "val" in rel or "valid" in rel or "hold" in rel:
        return "val"
    if "test" in rel:
        return "test"
    # The compact Zenodo mirror is a flat directory and may not carry split CSVs.
    # Keep a deterministic split so YOLO validation is meaningful and rerunnable.
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
    mask = _read_raster(label_path)
    mask = _to_flood_mask(mask)
    if mask.ndim != 2:
        stats.skipped += 1
        return
    h, w = mask.shape

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

    boxes = _mask_to_boxes(mask, min_area=min_area)
    labels = []
    for x1, y1, x2, y2 in boxes:
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
    replacements = {
        "s2": [("LabelHand", "S2Hand"), ("LabelWeak", "S2Weak"), ("label", "S2"), ("labels", "S2")],
        "s1": [("LabelHand", "S1Hand"), ("LabelWeak", "S1Weak"), ("label", "S1"), ("labels", "S1")],
    }
    sensors = ["s2", "s1"] if sensor == "auto" else [sensor]
    label_str = str(label_path)
    for wanted in sensors:
        for old, new in replacements[wanted]:
            path = Path(label_str.replace(old, new))
            if path.exists():
                candidates.append(path)
        if label_path.stem.lower().endswith("_class"):
            suffix = "_S2.tif" if wanted == "s2" else "_S1Hand_S1.tif"
            path = label_path.with_name(f"{_sample_key(label_path)}{suffix}")
            if path.exists():
                candidates.append(path)
    for stem in {_sample_key(label_path), *_candidate_stems(label_path.name)}:
        candidates.extend(image_index.get(stem, []))
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda p: _sensor_rank(p, sensor))
    return ranked[0]


def _sensor_rank(path: Path, sensor: str) -> tuple[int, str]:
    lower = "/".join(p.lower() for p in path.parts)
    if sensor == "s2":
        return (0 if "s2" in lower or "sentinel-2" in lower else 1, str(path))
    if sensor == "s1":
        return (0 if "s1" in lower or "sentinel-1" in lower else 1, str(path))
    return (0 if "s2" in lower else (1 if "s1" in lower else 2), str(path))


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
        if arr.ndim == 3 and arr.shape[0] <= 16:
            return arr
        if arr.ndim == 3:
            return np.moveaxis(arr, -1, 0)
        return arr
    except Exception:
        pass
    from PIL import Image

    return np.asarray(Image.open(path))


def _to_flood_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.dtype.kind in {"f"}:
        return (arr > 0.5).astype(np.uint8)
    unique = np.unique(arr)
    if 1 in unique:
        return (arr == 1).astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def _sample_key(path_or_name: Path | str) -> str:
    name = path_or_name.stem if isinstance(path_or_name, Path) else Path(path_or_name).stem
    return _strip_sensor_suffix(name)


def _strip_sensor_suffix(stem: str) -> str:
    patterns = [
        r"_S1Hand_S1$",
        r"_S1Weak_S1$",
        r"_S1$",
        r"_S2Hand_S2$",
        r"_S2Weak_S2$",
        r"_S2$",
        r"_CLASS$",
        r"_NDWI$",
        r"_LabelHand$",
        r"_LabelWeak$",
        r"_label$",
        r"_labels$",
    ]
    out = stem
    for pattern in patterns:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE)
    return out


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
    elif arr.ndim == 3 and arr.shape[0] <= 16:
        bands = arr
        if sensor == "s1" or bands.shape[0] == 2:
            b0 = bands[0]
            b1 = bands[min(1, bands.shape[0] - 1)]
            rgb = np.stack([b0, b1, 0.5 * (b0 + b1)], axis=-1)
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

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
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


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


if __name__ == "__main__":
    raise SystemExit(main())
