#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import h5py


def main() -> int:
    root = Path("/mnt/sda/heqing/QuickQuakeBuildings/earthquake_building_dataset")
    files = list(root.glob("damaged/*_opt.mat"))[:3]
    files += list(root.glob("intact/*_opt.mat"))[:3]
    files += list(root.glob("damaged/*_SAR.mat"))[:2]
    for path in files:
        print("==", path)
        with h5py.File(path, "r") as h5:
            def visit(name: str, obj: object) -> None:
                if isinstance(obj, h5py.Dataset):
                    value = obj[()]
                    minmax = None
                    if value.size:
                        minmax = (float(value.min()), float(value.max()))
                    print(name, value.shape, value.dtype, minmax)
            h5.visititems(visit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
