import argparse
import importlib.util
import json


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail fast if the RS-AI runtime cannot use CUDA.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    status = {
        "device": args.device,
        "numpy": has_module("numpy"),
        "cv2": has_module("cv2"),
        "torch": has_module("torch"),
        "ultralytics": has_module("ultralytics"),
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
    }

    if not all(status[name] for name in ("numpy", "cv2", "torch", "ultralytics")):
        print(json.dumps(status, ensure_ascii=False, indent=2))
        missing = [name for name in ("numpy", "cv2", "torch", "ultralytics") if not status[name]]
        raise SystemExit(f"missing Python modules: {', '.join(missing)}")

    import torch

    status["torch_version"] = torch.__version__
    status["cuda_available"] = bool(torch.cuda.is_available())
    status["cuda_device_count"] = int(torch.cuda.device_count())
    status["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not args.device.lower().startswith("cuda"):
        raise SystemExit(f"device must be a CUDA device for full-GPU mode, got: {args.device}")
    if not status["cuda_available"]:
        raise SystemExit("torch.cuda.is_available() is false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
