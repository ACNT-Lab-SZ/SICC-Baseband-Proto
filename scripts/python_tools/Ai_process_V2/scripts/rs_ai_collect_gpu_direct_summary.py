from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summarize(root: Path, scheme: str) -> dict:
    path = root / f"{scheme}_roundtrip.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    detections = sum(len(frame.get("detections", [])) for frame in data.get("frames", []))
    return {
        "frames": data["summary"]["frames"],
        "fps": data["summary"]["fps"],
        "avg_payload_nbytes": data["summary"]["avg_payload_nbytes"],
        "detections": detections,
        "roundtrip_json": str(path),
        "manifest_json": str(root / f"{scheme}_manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect GPU-direct two-scheme validation summary.")
    parser.add_argument("--root", default="logs/gpu_direct_two_scheme")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    root = Path(args.root)
    out = {
        "uniform": _summarize(root, "uniform"),
        "roi-layered": _summarize(root, "roi-layered"),
    }
    uniform_payload = out["uniform"]["avg_payload_nbytes"]
    layered_payload = out["roi-layered"]["avg_payload_nbytes"]
    out["comparison"] = {
        "roi_layered_payload_ratio": layered_payload / uniform_payload if uniform_payload else None,
        "roi_layered_payload_saving": 1.0 - layered_payload / uniform_payload if uniform_payload else None,
        "detection_delta": out["roi-layered"]["detections"] - out["uniform"]["detections"],
    }
    output = Path(args.output_json) if args.output_json else root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
