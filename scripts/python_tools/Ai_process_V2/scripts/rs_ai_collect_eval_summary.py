from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_run(root: Path) -> dict[str, Any]:
    detections = _load_json(root / "detections.json")
    ui_metrics = _load_json(root / "ui" / "ui_metrics.json")
    eval_metrics = _load_json(root / "eval.json").get("metrics", {})
    if not eval_metrics:
        eval_metrics = ui_metrics.get("evaluation", {})

    video = root / "ui" / "annotated.mp4"
    tx = root / "tx.rsbf"
    rx = root / "rx.rsbf"
    return {
        "tx_bytes": os.path.getsize(tx) if tx.exists() else None,
        "rx_bytes": os.path.getsize(rx) if rx.exists() else None,
        "frames": detections.get("summary", {}).get("frames"),
        "detections": detections.get("summary", {}).get("detections"),
        "decode_fps": detections.get("summary", {}).get("fps"),
        "tracks": ui_metrics.get("tracks", {}).get("count") or ui_metrics.get("track_count"),
        "precision": eval_metrics.get("precision"),
        "recall": eval_metrics.get("recall"),
        "f1": eval_metrics.get("f1"),
        "annotated_video": str(video),
        "annotated_video_exists": video.exists(),
        "annotated_video_bytes": os.path.getsize(video) if video.exists() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect RS-AI link test outputs into one JSON summary.")
    parser.add_argument("--uniform-root", default="logs/adaptive_eval/uniform")
    parser.add_argument("--baseline-root", default="logs/roi_layered_eval/baseline")
    parser.add_argument("--layered-root", default="logs/roi_layered_eval/layered")
    parser.add_argument("--output-json", default="logs/roi_layered_eval/test_pass_summary.json")
    args = parser.parse_args()

    summary = {
        "uniform_current": _summarize_run(Path(args.uniform_root)),
        "baseline_eval": _summarize_run(Path(args.baseline_root)),
        "roi_layered": _summarize_run(Path(args.layered_root)),
    }
    baseline = summary["baseline_eval"]
    layered = summary["roi_layered"]
    if baseline.get("tx_bytes") and layered.get("tx_bytes"):
        summary["comparison"] = {
            "roi_layered_vs_baseline_tx_ratio": layered["tx_bytes"] / baseline["tx_bytes"],
            "roi_layered_tx_saving": 1.0 - layered["tx_bytes"] / baseline["tx_bytes"],
            "detection_delta": layered.get("detections", 0) - baseline.get("detections", 0),
            "precision_delta": layered.get("precision", 0.0) - baseline.get("precision", 0.0),
            "recall_delta": layered.get("recall", 0.0) - baseline.get("recall", 0.0),
            "f1_delta": layered.get("f1", 0.0) - baseline.get("f1", 0.0),
        }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
