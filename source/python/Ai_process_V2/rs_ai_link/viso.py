from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


@dataclass
class VisoSequence:
    name: str
    root: Path
    frames_dir: Path | None = None
    video_file: Path | None = None
    annotation_file: Path | None = None
    frame_count: int = 0
    meta: dict = field(default_factory=dict)


def discover_viso_sequences(root: str | Path, max_depth: int = 4) -> list[VisoSequence]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    sequences: dict[Path, VisoSequence] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        depth = len(path.relative_to(root).parts)
        if depth > max_depth + 1:
            continue
        suffix = path.suffix.lower()
        parent = path.parent
        seq = sequences.get(parent)
        if seq is None:
            seq = VisoSequence(name=parent.name, root=parent)
            sequences[parent] = seq
        if suffix in IMAGE_EXTENSIONS:
            seq.frames_dir = parent
        elif suffix in VIDEO_EXTENSIONS and seq.video_file is None:
            seq.video_file = path
        elif suffix in {".txt", ".csv"} and _looks_like_annotation(path):
            seq.annotation_file = path

    out = []
    for seq in sequences.values():
        if seq.frames_dir:
            seq.frame_count = len(sorted(p for p in seq.frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))
        if seq.frames_dir or seq.video_file:
            out.append(seq)
    return sorted(out, key=lambda s: str(s.root))


def read_mot_like_annotations(path: str | Path) -> dict[int, list[dict]]:
    """Read MOT/VISO style annotations.

    Supported rows:
    frame,id,x,y,w,h,score,class,...
    frame,x,y,w,h,score,class,...
    x,y,w,h,score,class,... (single-frame fallback)
    """

    annotations: dict[int, list[dict]] = {}
    path = Path(path)
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace(" ", ",").split(",") if p.strip()]
        nums = []
        for part in parts:
            try:
                nums.append(float(part))
            except ValueError:
                break
        if len(nums) >= 8:
            frame_id = int(nums[0])
            track_id = int(nums[1])
            x, y, w, h = nums[2:6]
            score = nums[6]
            class_id = int(nums[7])
        elif len(nums) >= 7:
            frame_id = int(nums[0])
            track_id = -1
            x, y, w, h = nums[1:5]
            score = nums[5]
            class_id = int(nums[6])
        elif len(nums) >= 6:
            frame_id = 0
            track_id = -1
            x, y, w, h = nums[0:4]
            score = nums[4]
            class_id = int(nums[5])
        if class_id < 0:
            class_id = 0
        else:
            continue
        annotations.setdefault(frame_id, []).append(
            {
                "frame_id": frame_id,
                "track_id": track_id if track_id >= 0 else None,
                "class_id": class_id,
                "confidence": float(score),
                "xyxy": [float(x), float(y), float(x + w), float(y + h)],
            }
        )
    return annotations


def evaluate_detections(
    detections: dict[int, list[dict]],
    ground_truth: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict:
    tp = 0
    fp = 0
    fn = 0
    for frame_id, gt_items in ground_truth.items():
        det_items = list(detections.get(frame_id, []))
        matched_det: set[int] = set()
        for gt in gt_items:
            best_iou = 0.0
            best_idx = -1
            for idx, det in enumerate(det_items):
                if idx in matched_det:
                    continue
                if _class_id(det) != _class_id(gt):
                    continue
                iou = box_iou(det["xyxy"], gt["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                tp += 1
                matched_det.add(best_idx)
            else:
                fn += 1
        fp += max(0, len(det_items) - len(matched_det))
    for frame_id, det_items in detections.items():
        if frame_id not in ground_truth:
            fp += len(det_items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou_threshold": iou_threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def assign_iou_tracks(frames: list[dict], iou_threshold: float = 0.35, max_age: int = 3) -> list[dict]:
    next_track_id = 1
    tracks: list[dict] = []
    for frame in frames:
        frame_id = int(frame.get("frame_id", 0))
        detections = sorted(frame.get("detections", []), key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        used_tracks: set[int] = set()
        for det in detections:
            best_iou = 0.0
            best_track = None
            for track in tracks:
                if track["id"] in used_tracks:
                    continue
                if frame_id - track["last_frame"] > max_age:
                    continue
                if _class_id(det) != track["class_id"]:
                    continue
                iou = box_iou(det["xyxy"], track["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track is not None and best_iou >= iou_threshold:
                det["track_id"] = best_track["id"]
                best_track["xyxy"] = det["xyxy"]
                best_track["last_frame"] = frame_id
                best_track["history"].append(_box_center(det["xyxy"]))
                used_tracks.add(best_track["id"])
            else:
                det["track_id"] = next_track_id
                tracks.append(
                    {
                        "id": next_track_id,
                        "class_id": _class_id(det),
                        "xyxy": det["xyxy"],
                        "last_frame": frame_id,
                        "history": [_box_center(det["xyxy"])],
                    }
                )
                used_tracks.add(next_track_id)
                next_track_id += 1
        frame["detections"] = detections
    return frames


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _box_center(box: list[float]) -> list[float]:
    return [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]


def _class_id(item: dict) -> int:
    if "class_id" in item:
        return int(item["class_id"])
    return int(item.get("category_id", -1))


def _looks_like_annotation(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ["ann", "gt", "label", "mot", "track", "bbox"])
