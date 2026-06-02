import argparse

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    model = YOLO(args.model).model
    print(f"model_class: {type(model).__name__}")
    print(f"layers: {len(model.model)}")
    print(f"save: {list(model.save)}")
    print(f"names: {model.names}")
    for layer in model.model:
        print(f"{layer.i:02d} from={layer.f} type={layer.type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

