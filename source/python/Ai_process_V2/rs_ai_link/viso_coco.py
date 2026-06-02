from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConvertStats:
    split: str
    images: int = 0
    annotations: int = 0
    labels_written: int = 0


def convert_viso_coco_to_yolo(
    coco_root: str | Path,
    output_root: str | Path,
    class_name: str = "car",
    copy_images: bool = False,
) -> dict:
    coco_root = Path(coco_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stats = []
    for split in ["train2017", "val2017", "test2017"]:
        ann = coco_root / "Annotations" / f"instances_{split}.json"
        image_dir = coco_root / split
        if not ann.exists() or not image_dir.exists():
            continue
        stats.append(_convert_split(split, ann, image_dir, output_root, copy_images))
    split_names = [s.split for s in stats]
    val_split = "val" if "val" in split_names else ("test" if "test" in split_names else "train")
    test_line = "test: images/test" if "test" in split_names else f"test: images/{val_split}"
    yaml_path = output_root / "viso_car.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root}",
                "train: images/train",
                f"val: images/{val_split}",
                test_line,
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "yaml": str(yaml_path),
        "output_root": str(output_root),
        "class_name": class_name,
        "copy_images": copy_images,
        "splits": [s.__dict__ for s in stats],
    }


def _convert_split(split: str, annotation_path: Path, image_dir: Path, output_root: Path, copy_images: bool) -> ConvertStats:
    split_name = {"train2017": "train", "val2017": "val", "test2017": "test"}[split]
    out_images = output_root / "images" / split_name
    out_labels = output_root / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data.get("images", [])}
    labels: dict[int, list[str]] = {image_id: [] for image_id in images}
    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        image = images.get(image_id)
        if image is None:
            continue
        w = float(image["width"])
        h = float(image["height"])
        x, y, bw, bh = [float(v) for v in ann["bbox"]]
        if bw <= 0 or bh <= 0 or w <= 0 or h <= 0:
            continue
        xc = (x + bw * 0.5) / w
        yc = (y + bh * 0.5) / h
        nw = bw / w
        nh = bh / h
        vals = [_clip01(v) for v in [xc, yc, nw, nh]]
        labels.setdefault(image_id, []).append("0 " + " ".join(f"{v:.8f}" for v in vals))

    for image_id, image in images.items():
        src = image_dir / image["file_name"]
        dst = out_images / image["file_name"]
        if copy_images:
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        else:
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
        label_path = out_labels / (Path(image["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(labels.get(image_id, [])) + ("\n" if labels.get(image_id) else ""), encoding="utf-8")

    return ConvertStats(
        split=split_name,
        images=len(images),
        annotations=len(data.get("annotations", [])),
        labels_written=sum(len(v) for v in labels.values()),
    )


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))
