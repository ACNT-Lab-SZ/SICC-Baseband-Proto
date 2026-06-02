import argparse
import shutil
from pathlib import Path
from typing import Optional


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=80)
    parser.add_argument("--class-name", default="car")
    args = parser.parse_args()

    src = Path(args.source_root)
    out = Path(args.output_root)
    _copy_split(src, out, "train", args.train)
    val_split = "val" if (src / "images" / "val").exists() else "test"
    _copy_split(src, out, val_split, args.val, out_name="val")
    yaml = out / "viso_car_subset.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                "val: images/val",
                "test: images/val",
                "names:",
                f"  0: {args.class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[RS-AI] subset={out}")
    print(f"[RS-AI] yaml={yaml}")
    return 0


def _copy_split(src: Path, out: Path, split: str, limit: int, out_name: Optional[str] = None) -> None:
    out_split = out_name or split
    out_img = out / "images" / out_split
    out_lab = out / "labels" / out_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    images = sorted((src / "images" / split).glob("*"))[:limit]
    for image in images:
        label = src / "labels" / split / (image.stem + ".txt")
        dst_img = out_img / image.name
        dst_lab = out_lab / (image.stem + ".txt")
        if not dst_img.exists():
            try:
                dst_img.symlink_to(image.resolve())
            except OSError:
                shutil.copy2(image, dst_img)
        if label.exists() and not dst_lab.exists():
            shutil.copy2(label, dst_lab)
        elif not dst_lab.exists():
            dst_lab.write_text("", encoding="utf-8")
    print(f"[RS-AI] subset_split={out_split} images={len(images)}")


if __name__ == "__main__":
    raise SystemExit(main())
