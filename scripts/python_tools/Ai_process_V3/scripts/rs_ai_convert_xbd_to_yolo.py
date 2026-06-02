from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["no_damage", "minor_damage", "major_damage", "destroyed"]
SUBTYPE_TO_CLASS = {
    "no-damage": 0,
    "no_damage": 0,
    "minor-damage": 1,
    "minor_damage": 1,
    "major-damage": 2,
    "major_damage": 2,
    "destroyed": 3,
}


@dataclass
class SplitStats:
    split: str
    labels: int = 0
    images: int = 0
    boxes: int = 0
    skipped: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert xBD/xView2 building-damage labels to YOLO boxes.")
    parser.add_argument("--xbd-root", required=True, help="Root that contains xBD split folders such as train/tier3/test.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking when possible.")
    parser.add_argument("--include-undamaged", action="store_true", help="Keep no_damage buildings. Default: keep all damage classes including no_damage.")
    parser.add_argument("--post-only", action="store_true", default=True, help="Use post_disaster images/labels.")
    args = parser.parse_args()

    xbd_root = Path(args.xbd_root)
    output_root = Path(args.output_root)
    if not xbd_root.exists():
        raise FileNotFoundError(xbd_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats: dict[str, SplitStats] = {}
    label_files = sorted(xbd_root.rglob("*_post_disaster.json"))
    if not label_files:
        label_files = sorted(xbd_root.rglob("*.json"))
    for label_file in label_files:
        split = _infer_split(label_file, xbd_root)
        stats.setdefault(split, SplitStats(split=split))
        _convert_label(label_file, xbd_root, output_root, split, args.copy_images, stats[split])

    if "train" not in stats and stats:
        first = sorted(stats)[0]
        stats["train"] = stats.pop(first)
        stats["train"].split = "train"

    val_split = "val" if "val" in stats else ("test" if "test" in stats else "train")
    yaml_path = output_root / "xbd_damage.yaml"
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
        "dataset": "xBD/xView2 building damage",
        "yaml": str(yaml_path),
        "output_root": str(output_root),
        "classes": CLASS_NAMES,
        "splits": [stats[k].__dict__ for k in sorted(stats)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _infer_split(path: Path, root: Path) -> str:
    rel_parts = [p.lower() for p in path.relative_to(root).parts]
    joined = "/".join(rel_parts)
    if "train" in rel_parts or "/train/" in joined:
        return "train"
    if "val" in rel_parts or "hold" in rel_parts or "tier3" in rel_parts:
        return "val"
    if "test" in rel_parts:
        return "test"
    return "train"


def _convert_label(label_file: Path, root: Path, output_root: Path, split: str, copy_images: bool, stats: SplitStats) -> None:
    data = json.loads(label_file.read_text(encoding="utf-8"))
    image_name = data.get("metadata", {}).get("img_name") or label_file.with_suffix(".png").name
    image_path = _find_image(root, label_file, image_name)
    width = int(data.get("metadata", {}).get("width") or 1024)
    height = int(data.get("metadata", {}).get("height") or 1024)
    if image_path is None:
        stats.skipped += 1
        return

    out_image_dir = output_root / "images" / split
    out_label_dir = output_root / "labels" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_image = out_image_dir / image_path.name
    if not out_image.exists():
        if copy_images:
            shutil.copy2(image_path, out_image)
        else:
            try:
                out_image.symlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, out_image)

    labels = []
    for feature in data.get("features", {}).get("xy", []):
        subtype = str(feature.get("properties", {}).get("subtype", "")).lower()
        class_id = SUBTYPE_TO_CLASS.get(subtype)
        if class_id is None:
            stats.skipped += 1
            continue
        points = _parse_polygon_xy(str(feature.get("wkt", "")))
        if not points:
            stats.skipped += 1
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            stats.skipped += 1
            continue
        xc = (x1 + bw * 0.5) / width
        yc = (y1 + bh * 0.5) / height
        labels.append(f"{class_id} {_clip01(xc):.8f} {_clip01(yc):.8f} {_clip01(bw / width):.8f} {_clip01(bh / height):.8f}")

    out_label = out_label_dir / f"{image_path.stem}.txt"
    out_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    stats.labels += 1
    stats.images += 1
    stats.boxes += len(labels)


def _find_image(root: Path, label_file: Path, image_name: str) -> Path | None:
    candidates = [
        label_file.parents[1] / "images" / image_name if len(label_file.parents) > 1 else label_file.with_name(image_name),
        label_file.parent.parent / "images" / image_name,
        label_file.with_name(image_name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(image_name))
    return matches[0] if matches else None


def _parse_polygon_xy(wkt: str) -> list[tuple[float, float]]:
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", wkt)]
    if len(nums) < 4:
        return []
    return list(zip(nums[0::2], nums[1::2]))


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


if __name__ == "__main__":
    raise SystemExit(main())
