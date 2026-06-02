import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        fraction=args.fraction,
        val=args.val,
        amp=args.amp,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"[RS-AI] best={best} exists={best.exists()}")
    return 0 if best.exists() else 3


if __name__ == "__main__":
    raise SystemExit(main())
