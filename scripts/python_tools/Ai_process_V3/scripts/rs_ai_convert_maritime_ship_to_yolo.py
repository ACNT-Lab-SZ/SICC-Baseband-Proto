from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["ship"]
SHIP_WORDS = {"ship", "vessel", "boat", "target"}


@dataclass
class SplitStats:
    split: str
    images: int = 0
    boxes: int = 0
    skipped: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert public maritime/SAR ship datasets to YOLO format.")
    parser.add_argument("--ship-root", required=True, help="Dataset root, e.g. HRSID, SSDD, xView3 crop export.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--format", choices=["auto", "coco", "voc", "yolo"], default="auto")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.ship_root)
    out = Path(args.output_root)
    if not root.exists():
        raise FileNotFoundError(root)
    out.mkdir(parents=True, exist_ok=True)

    fmt = _detect_format(root, args.format)
    if fmt == "coco":
        stats = _convert_coco(root, out, copy_images=args.copy_images, max_samples=args.max_samples)
    elif fmt == "voc":
        stats = _convert_voc(root, out, copy_images=args.copy_images, max_samples=args.max_samples)
    elif fmt == "yolo":
        stats = _convert_yolo(root, out, copy_images=args.copy_images, max_samples=args.max_samples)
    else:
        raise RuntimeError(f"Unsupported maritime dataset format: {fmt}")

    val_split = "val" if "val" in stats else ("test" if "test" in stats else "train")
    yaml_path = out / "maritime_ship.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                f"val: images/{val_split}",
                f"test: images/{'test' if 'test' in stats else val_split}",
                "names:",
                "  0: ship",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": "maritime ship detection",
                "format": fmt,
                "yaml": str(yaml_path),
                "output_root": str(out),
                "classes": CLASS_NAMES,
                "splits": [stats[k].__dict__ for k in sorted(stats)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _detect_format(root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if any(root.rglob("*.json")):
        return "coco"
    if any(root.rglob("*.xml")):
        return "voc"
    if any((p.name.lower() == "labels" or "labels" in [part.lower() for part in p.parts]) for p in root.rglob("*") if p.is_dir()):
        return "yolo"
    raise FileNotFoundError("Could not detect COCO json, VOC xml, or YOLO labels under dataset root.")


def _convert_coco(root: Path, out: Path, copy_images: bool, max_samples: int) -> dict[str, SplitStats]:
    image_index = _image_index(root)
    stats: dict[str, SplitStats] = {}
    converted = 0
    for json_path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
            continue
        categories = {int(c.get("id", -1)): str(c.get("name", "")).lower() for c in data.get("categories", [])}
        images = {int(img["id"]): img for img in data.get("images", []) if "id" in img}
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            cat = categories.get(int(ann.get("category_id", -1)), "ship")
            if categories and not any(word in cat for word in SHIP_WORDS):
                continue
            anns_by_image.setdefault(int(ann.get("image_id", -1)), []).append(ann)
        for image_id, img in images.items():
            if max_samples > 0 and converted >= max_samples:
                break
            anns = anns_by_image.get(image_id, [])
            if not anns:
                continue
            filename = str(img.get("file_name") or img.get("name") or "")
            image_path = _resolve_image(root, json_path, filename, image_index)
            split = _infer_split(image_path or json_path, root)
            stats.setdefault(split, SplitStats(split))
            if image_path is None:
                stats[split].skipped += 1
                continue
            width = int(img.get("width") or 0)
            height = int(img.get("height") or 0)
            if width <= 0 or height <= 0:
                width, height = _image_size(image_path)
            boxes = []
            for ann in anns:
                bbox = ann.get("bbox") or []
                if len(bbox) < 4:
                    continue
                x, y, w, h = [float(v) for v in bbox[:4]]
                if w <= 1 or h <= 1:
                    continue
                boxes.append(_yolo_line(x + w / 2, y + h / 2, w, h, width, height))
            if not boxes:
                stats[split].skipped += 1
                continue
            _write_sample(image_path, out, split, boxes, copy_images)
            stats[split].images += 1
            stats[split].boxes += len(boxes)
            converted += 1
    if not stats:
        raise FileNotFoundError(f"No usable COCO ship annotations found under {root}")
    return stats


def _convert_voc(root: Path, out: Path, copy_images: bool, max_samples: int) -> dict[str, SplitStats]:
    image_index = _image_index(root)
    xml_files = _voc_xml_files(root)
    stats: dict[str, SplitStats] = {}
    converted = 0
    written: set[tuple[str, str]] = set()
    for xml_path in xml_files:
        if max_samples > 0 and converted >= max_samples:
            break
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root_xml = tree.getroot()
        filename = _text(root_xml.find("filename")) or f"{xml_path.stem}.jpg"
        image_path = _resolve_image(root, xml_path, filename, image_index)
        split = _infer_split(image_path or xml_path, root)
        stats.setdefault(split, SplitStats(split))
        if image_path is None:
            stats[split].skipped += 1
            continue
        width = _int_text(root_xml.find("size/width"))
        height = _int_text(root_xml.find("size/height"))
        if width <= 0 or height <= 0:
            width, height = _image_size(image_path)
        boxes = []
        for obj in root_xml.findall("object"):
            name = (_text(obj.find("name")) or "ship").lower()
            if name and not any(word in name for word in SHIP_WORDS):
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            x1 = _float_text(box.find("xmin"))
            y1 = _float_text(box.find("ymin"))
            x2 = _float_text(box.find("xmax"))
            y2 = _float_text(box.find("ymax"))
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue
            boxes.append(_yolo_line(x1 + w / 2, y1 + h / 2, w, h, width, height))
        if not boxes:
            stats[split].skipped += 1
            continue
        key = (split, image_path.stem)
        if key in written:
            continue
        _write_sample(image_path, out, split, boxes, copy_images)
        written.add(key)
        stats[split].images += 1
        stats[split].boxes += len(boxes)
        converted += 1
    if not stats:
        raise FileNotFoundError(f"No usable VOC ship annotations found under {root}")
    return stats


def _convert_yolo(root: Path, out: Path, copy_images: bool, max_samples: int) -> dict[str, SplitStats]:
    image_index = _image_index(root)
    label_files = sorted(p for p in root.rglob("*.txt") if "label" in _path_text(p))
    stats: dict[str, SplitStats] = {}
    converted = 0
    for label in label_files:
        if max_samples > 0 and converted >= max_samples:
            break
        image_path = image_index.get(label.stem)
        split = _infer_split(label, root)
        stats.setdefault(split, SplitStats(split))
        if image_path is None:
            stats[split].skipped += 1
            continue
        lines = []
        for raw in label.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.split()
            if len(parts) >= 5:
                lines.append("0 " + " ".join(parts[1:5]))
        if not lines:
            stats[split].skipped += 1
            continue
        _write_sample(image_path, out, split, lines, copy_images)
        stats[split].images += 1
        stats[split].boxes += len(lines)
        converted += 1
    if not stats:
        raise FileNotFoundError(f"No usable YOLO ship labels found under {root}")
    return stats


def _write_sample(image_path: Path, out: Path, split: str, yolo_lines: list[str], copy_images: bool) -> None:
    image_dir = out / "images" / split
    label_dir = out / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    out_image = image_dir / f"{image_path.stem}{image_path.suffix.lower()}"
    out_label = label_dir / f"{image_path.stem}.txt"
    if copy_images:
        shutil.copy2(image_path, out_image)
    elif not out_image.exists():
        try:
            out_image.symlink_to(image_path.resolve())
        except OSError:
            shutil.copy2(image_path, out_image)
    out_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")


def _image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda p: (_image_rank(p), str(p))):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if _is_visualized_gt_image(path):
            continue
        index.setdefault(path.stem, path)
    return index


def _voc_xml_files(root: Path) -> list[Path]:
    xml_files = sorted(root.rglob("*.xml"))
    split_specific = [
        p
        for p in xml_files
        if any(part.lower().startswith("annotations_") for part in p.parts)
    ]
    return split_specific or xml_files


def _image_rank(path: Path) -> int:
    text = _path_text(path)
    if _is_visualized_gt_image(path):
        return 9
    if "/jpegimages/" in text or text.endswith("/jpegimages"):
        return 0
    if "train" in text or "test" in text or "val" in text:
        return 1
    return 2


def _is_visualized_gt_image(path: Path) -> bool:
    text = _path_text(path)
    return any(token in text for token in ("_gt", "bbox_gt", "rbox_gt", "pseg_gt", "label_on"))


def _resolve_image(root: Path, ann_path: Path, filename: str, image_index: dict[str, Path]) -> Path | None:
    if filename:
        direct = ann_path.parent / filename
        if direct.exists():
            return direct
        direct = root / filename
        if direct.exists():
            return direct
        stem = Path(filename).stem
        if stem in image_index:
            return image_index[stem]
    if ann_path.stem in image_index:
        return image_index[ann_path.stem]
    return None


def _infer_split(path: Path, root: Path) -> str:
    rel = _path_text(path.relative_to(root) if path.is_relative_to(root) else path)
    if "train" in rel:
        return "train"
    if "val" in rel or "valid" in rel:
        return "val"
    if "test" in rel:
        return "test"
    bucket = int(hashlib.sha1(path.stem.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return int(im.width), int(im.height)


def _yolo_line(xc: float, yc: float, w: float, h: float, image_w: int, image_h: int) -> str:
    return f"0 {_clip01(xc / image_w):.8f} {_clip01(yc / image_h):.8f} {_clip01(w / image_w):.8f} {_clip01(h / image_h):.8f}"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _path_text(path: Path) -> str:
    return "/".join(part.lower() for part in path.parts)


def _text(node: ET.Element | None) -> str:
    return "" if node is None or node.text is None else node.text.strip()


def _int_text(node: ET.Element | None) -> int:
    try:
        return int(float(_text(node)))
    except ValueError:
        return 0


def _float_text(node: ET.Element | None) -> float:
    try:
        return float(_text(node))
    except ValueError:
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
