import importlib.util


for name in ["numpy", "cv2", "torch", "ultralytics"]:
    print(f"{name}: {'yes' if importlib.util.find_spec(name) else 'no'}")

try:
    import torch

    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"cuda_device_{idx}: {torch.cuda.get_device_name(idx)}")
except Exception as exc:
    print(f"torch_error: {type(exc).__name__}: {exc}")

