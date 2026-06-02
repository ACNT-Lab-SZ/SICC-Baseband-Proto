#!/usr/bin/env python3
from __future__ import annotations

import json
import os


def main() -> int:
    import torch

    result = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "is_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    for idx in range(torch.cuda.device_count()):
        item = {"index": idx}
        try:
            props = torch.cuda.get_device_properties(idx)
            item.update(
                {
                    "name": props.name,
                    "major": props.major,
                    "minor": props.minor,
                    "total_memory": props.total_memory,
                }
            )
        except Exception as exc:
            item["error"] = repr(exc)
        result["devices"].append(item)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
