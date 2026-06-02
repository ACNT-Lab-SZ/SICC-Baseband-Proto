import importlib


for mod_name in [
    "ultralytics.utils.ops",
    "ultralytics.utils.nms",
    "ultralytics.models.yolo.detect.predict",
]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        print(f"{mod_name}: error {type(exc).__name__}: {exc}")
        continue
    print(
        f"{mod_name}: "
        f"non_max_suppression={hasattr(mod, 'non_max_suppression')} "
        f"scale_boxes={hasattr(mod, 'scale_boxes')}"
    )

