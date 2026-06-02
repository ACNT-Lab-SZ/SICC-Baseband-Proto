from __future__ import annotations

import argparse
import json
from pathlib import Path

from rs_ai_link.viso import evaluate_detections, read_mot_like_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate detection JSON against a MOT/VISO annotation file.")
    parser.add_argument("--detections-json", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    detections = json.loads(Path(args.detections_json).read_text(encoding="utf-8"))
    ground_truth = read_mot_like_annotations(args.annotation)
    det_map = {
        int(frame.get("frame_id", idx)): list(frame.get("detections", []))
        for idx, frame in enumerate(detections.get("frames", []))
    }
    metrics = evaluate_detections(det_map, ground_truth, iou_threshold=args.iou)
    payload = {
        "detections_json": args.detections_json,
        "annotation": args.annotation,
        "metrics": metrics,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
