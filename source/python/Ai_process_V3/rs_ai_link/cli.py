from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlretrieve

from .bitstream import FeatureFrame, TensorBlob, inspect_stream, read_stream, write_stream
from .viso import assign_iou_tracks, discover_viso_sequences, evaluate_detections, read_mot_like_annotations
from .viso_coco import convert_viso_coco_to_yolo
from .yolo_split import GpuDirectFrame, ImportanceAwareSplitYoloFeatureCodec, LayeredRoiSplitYoloFeatureCodec, SplitYoloFeatureCodec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rs_ai_link", description="Remote-sensing AI feature bitstream tools.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode-video", help="Extract split YOLO features from a video into an RSBF bitstream.")
    _add_common_encode_args(enc, include_video=True)
    _add_visual_preview_args(enc)

    enc_img = sub.add_parser("encode-images", help="Extract split YOLO features from satellite images into an RSBF bitstream.")
    _add_common_encode_args(enc_img, include_images=True)
    _add_visual_preview_args(enc_img)

    enc_imp = sub.add_parser("encode-video-importance", help="Encode a video with ROI-priority task-driven feature transmission.")
    _add_common_encode_args(enc_imp, include_video=True)
    _add_importance_args(enc_imp)

    enc_img_imp = sub.add_parser("encode-images-importance", help="Encode images with ROI-priority task-driven feature transmission.")
    _add_common_encode_args(enc_img_imp, include_images=True)
    _add_importance_args(enc_img_imp)

    viso_scan = sub.add_parser("scan-viso", help="Scan a VISO-style dataset root for videos or frame sequences.")
    viso_scan.add_argument("--root", required=True)
    viso_scan.add_argument("--output-json", default="")

    enc_viso = sub.add_parser("encode-viso", help="Encode a VISO video or frame sequence into an RSBF feature bitstream.")
    _add_viso_encode_args(enc_viso)
    _add_visual_preview_args(enc_viso)

    enc_viso_imp = sub.add_parser("encode-viso-importance", help="Encode VISO with ROI-priority task-driven feature transmission.")
    _add_viso_encode_args(enc_viso_imp)
    _add_importance_args(enc_viso_imp)

    enc_layer = sub.add_parser("encode-video-roi-layered", help="Encode a video with coarse-background and full-resolution ROI enhancement.")
    _add_common_encode_args(enc_layer, include_video=True)
    _add_layered_args(enc_layer)

    enc_img_layer = sub.add_parser("encode-images-roi-layered", help="Encode images with coarse-background and full-resolution ROI enhancement.")
    _add_common_encode_args(enc_img_layer, include_images=True)
    _add_layered_args(enc_img_layer)

    enc_viso_layer = sub.add_parser("encode-viso-roi-layered", help="Encode VISO with coarse-background and full-resolution ROI enhancement.")
    _add_viso_encode_args(enc_viso_layer)
    _add_layered_args(enc_viso_layer)

    conv_viso = sub.add_parser("convert-viso-coco", help="Convert VISO COCO detection annotations to YOLO labels.")
    conv_viso.add_argument("--coco-root", required=True)
    conv_viso.add_argument("--output-root", required=True)
    conv_viso.add_argument("--class-name", default="car")
    conv_viso.add_argument("--copy-images", action="store_true")

    dec = sub.add_parser("decode-bitstream", help="Run the YOLO suffix and NMS on a received RSBF bitstream.")
    _add_decode_args(dec)

    dec_imp = sub.add_parser("decode-importance-bitstream", help="Decode an ROI-priority task-driven RSBF bitstream.")
    _add_decode_args(dec_imp)

    gpu_direct = sub.add_parser("gpu-direct-roundtrip", help="Run a GPU-resident feature payload roundtrip without writing RSBF.")
    _add_gpu_direct_args(gpu_direct)

    insp = sub.add_parser("inspect-bitstream", help="Inspect an RSBF bitstream without ML dependencies.")
    insp.add_argument("--input", required=True)
    insp.add_argument("--max-frames", type=int, default=5)

    mock = sub.add_parser("mock-bitstream", help="Create a small synthetic RSBF file for local link testing.")
    mock.add_argument("--output", required=True)
    mock.add_argument("--frames", type=int, default=4)

    boot = sub.add_parser("bootstrap-vhr10-demo", help="Download open VHR-10 samples and a VHR-10 YOLOv8n model.")
    boot.add_argument("--output-dir", default="open_data/vhr10_demo")
    boot.add_argument("--images", type=int, default=8)

    render = sub.add_parser("render-detections", help="Render UI-ready annotated images/video and metrics JSON.")
    render.add_argument("--detections-json", required=True)
    render.add_argument("--output-dir", required=True)
    render.add_argument("--image-dir", default="")
    render.add_argument("--video", default="")
    render.add_argument("--pattern", default="*.jpg")
    render.add_argument("--output-video", default="")
    render.add_argument("--metrics-json", default="")
    render.add_argument("--tx-bitstream", default="")
    render.add_argument("--rx-bitstream", default="")
    render.add_argument("--source-bitstream", default="", help="Optional RSBF file that carries embedded visual preview layers.")
    render.add_argument("--fps", type=float, default=2.0)
    render.add_argument("--track", action="store_true", help="Assign lightweight IoU track_id values before rendering.")
    render.add_argument("--track-iou", type=float, default=0.35)
    render.add_argument("--gt-annotation", default="", help="Optional MOT/VISO annotation file for precision/recall/F1.")
    render.add_argument("--eval-iou", type=float, default=0.5)

    args = parser.parse_args(argv)
    if args.cmd == "encode-video":
        return _encode_video(args)
    if args.cmd == "encode-images":
        return _encode_images(args)
    if args.cmd == "encode-video-importance":
        return _encode_video_importance(args)
    if args.cmd == "encode-images-importance":
        return _encode_images_importance(args)
    if args.cmd == "scan-viso":
        return _scan_viso(args)
    if args.cmd == "encode-viso":
        return _encode_viso(args)
    if args.cmd == "encode-viso-importance":
        return _encode_viso_importance(args)
    if args.cmd == "encode-video-roi-layered":
        return _encode_video_roi_layered(args)
    if args.cmd == "encode-images-roi-layered":
        return _encode_images_roi_layered(args)
    if args.cmd == "encode-viso-roi-layered":
        return _encode_viso_roi_layered(args)
    if args.cmd == "convert-viso-coco":
        info = convert_viso_coco_to_yolo(args.coco_root, args.output_root, class_name=args.class_name, copy_images=args.copy_images)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "decode-bitstream":
        return _decode_bitstream(args)
    if args.cmd == "decode-importance-bitstream":
        return _decode_bitstream(args, importance=True)
    if args.cmd == "gpu-direct-roundtrip":
        return _gpu_direct_roundtrip(args)
    if args.cmd == "inspect-bitstream":
        print(json.dumps(inspect_stream(args.input, args.max_frames), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "mock-bitstream":
        _write_mock(args.output, args.frames)
        return 0
    if args.cmd == "bootstrap-vhr10-demo":
        _bootstrap_vhr10_demo(args.output_dir, args.images)
        return 0
    if args.cmd == "render-detections":
        _render_detections(args)
        return 0
    raise AssertionError(args.cmd)


def _add_common_encode_args(parser: argparse.ArgumentParser, *, include_video: bool = False, include_images: bool = False) -> None:
    if include_video:
        parser.add_argument("--video", required=True)
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument("--max-frames", type=int, default=0)
    if include_images:
        parser.add_argument("--image-dir", required=True)
        parser.add_argument("--pattern", default="*.jpg")
        parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-layer", type=int, default=22)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tensor-codec", choices=["int8", "float16", "float32"], default="int8")
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--half", action="store_true")


def _add_importance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--roi-conf", type=float, default=0.20)
    parser.add_argument("--roi-margin", type=float, default=0.10)
    parser.add_argument("--bg-keep-ratio", type=float, default=0.15)
    parser.add_argument("--bg-scale-boost", type=float, default=4.0)


def _add_visual_preview_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embed-visual-preview", action="store_true")
    parser.add_argument("--visual-downsample", type=int, default=1)
    parser.add_argument("--visual-quality", type=int, default=85)


def _add_layered_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--roi-conf", type=float, default=0.20)
    parser.add_argument("--roi-margin", type=float, default=0.10)
    parser.add_argument("--bg-downsample", type=int, default=4)


def _add_viso_encode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequence", default="", help="Sequence root. Used to auto-discover frames/video.")
    parser.add_argument("--frames-dir", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--annotation", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--split-layer", type=int, default=22)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tensor-codec", choices=["int8", "float16", "float32"], default="int8")
    parser.add_argument("--zlib-level", type=int, default=1)
    parser.add_argument("--half", action="store_true")


def _add_decode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split-layer", type=int, default=0, help="Override stream split layer. Default: read from stream metadata.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tensor-codec", choices=["int8", "float16", "float32"], default="int8")
    parser.add_argument("--half", action="store_true")


def _add_gpu_direct_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", default="")
    parser.add_argument("--frames-dir", default="")
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scheme", choices=["uniform", "roi-layered"], default="uniform")
    parser.add_argument("--split-layer", type=int, default=22)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tensor-codec", choices=["int8", "float16", "float32"], default="int8")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--roi-conf", type=float, default=0.20)
    parser.add_argument("--roi-margin", type=float, default=0.10)
    parser.add_argument("--bg-downsample", type=int, default=4)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--half", action="store_true")


def _encode_video(args: argparse.Namespace) -> int:
    import cv2

    codec = _make_codec(args)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = max(1, int(args.stride))
    max_frames = int(args.max_frames)

    def frames():
        emitted = 0
        source_idx = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_idx % stride != 0:
                source_idx += 1
                continue
            pts_ms = (source_idx * 1000.0 / fps) if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC))
            yield codec.extract_feature_frame(frame, emitted, pts_ms)
            emitted += 1
            source_idx += 1
            if max_frames > 0 and emitted >= max_frames:
                break
        elapsed = max(time.time() - t0, 1.0e-9)
        print(f"[RS-AI] encoded_frames={emitted} fps={emitted / elapsed:.2f}")

    write_stream(
        args.output,
        frames(),
        {
            "model": args.model,
            "split_layer": args.split_layer,
            "imgsz": args.imgsz,
            "tensor_codec": args.tensor_codec,
            "video": str(Path(args.video)),
            "stride": stride,
            "resource_mode": _resource_mode_for_args(args),
        },
    )
    cap.release()
    print(f"[RS-AI] bitstream={args.output}")
    return 0


def _gpu_direct_roundtrip(args: argparse.Namespace) -> int:
    import cv2

    if not args.video and not args.frames_dir:
        raise RuntimeError("gpu-direct-roundtrip requires --video or --frames-dir")
    if args.scheme == "roi-layered":
        codec = LayeredRoiSplitYoloFeatureCodec(
            model_path=args.model,
            split_layer=args.split_layer,
            device=args.device,
            imgsz=args.imgsz,
            tensor_codec=args.tensor_codec,
            half=args.half,
            roi_conf=args.roi_conf,
            roi_margin=args.roi_margin,
            bg_downsample=args.bg_downsample,
        )
    else:
        codec = SplitYoloFeatureCodec(
            model_path=args.model,
            split_layer=args.split_layer,
            device=args.device,
            imgsz=args.imgsz,
            tensor_codec=args.tensor_codec,
            half=args.half,
        )
    manifests: list[dict] = []
    outputs: list[dict] = []
    frame_count = 0
    t0 = time.time()

    def handle_frame(frame_bgr: Any, frame_id: int, pts_ms: float) -> None:
        nonlocal frame_count
        gpu_frame: GpuDirectFrame = codec.extract_gpu_direct_frame(frame_bgr, frame_id, pts_ms)
        manifests.append(gpu_frame.manifest())
        detections = codec.detect_from_gpu_direct_frame(gpu_frame, conf=args.conf, iou=args.iou)
        outputs.append(
            {
                "frame_id": int(frame_id),
                "pts_ms": float(pts_ms),
                "payload_nbytes": gpu_frame.payload_nbytes,
                "detections": [
                    {
                        "class_id": int(det.class_id),
                        "class_name": det.class_name,
                        "confidence": float(det.confidence),
                        "xyxy": [float(v) for v in det.xyxy],
                    }
                    for det in detections
                ],
            }
        )
        frame_count += 1

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pts_ms = (frame_count * 1000.0 / fps) if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC))
            handle_frame(frame, frame_count, pts_ms)
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
        cap.release()
    else:
        frame_paths = sorted(Path(args.frames_dir).glob(args.pattern))
        if args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]
        for path in frame_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read frame: {path}")
            handle_frame(frame, frame_count, frame_count * 1000.0)

    elapsed = max(time.time() - t0, 1.0e-9)
    summary = {
        "protocol": "rs-ai-gpu-direct-v1",
        "scheme": args.scheme,
        "frames": frame_count,
        "fps": frame_count / elapsed,
        "avg_payload_nbytes": (sum(item["payload_nbytes"] for item in outputs) / frame_count) if frame_count else 0.0,
    }
    result = {"summary": summary, "frames": outputs}
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.manifest_json:
        Path(args.manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_json).write_text(json.dumps({"protocol": "rs-ai-gpu-direct-v1", "scheme": args.scheme, "frames": manifests}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RS-AI] gpu_direct_frames={frame_count} fps={frame_count / elapsed:.2f}")
    return 0


def _encode_images(args: argparse.Namespace) -> int:
    import cv2

    image_dir = Path(args.image_dir)
    images = sorted(p for p in image_dir.glob(args.pattern) if p.is_file())
    if not images:
        raise RuntimeError(f"no images matched {image_dir / args.pattern}")
    if args.max_images > 0:
        images = images[: args.max_images]
    codec = _make_codec(args)

    def frames():
        emitted = 0
        t0 = time.time()
        for image_path in images:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read image: {image_path}")
            yield codec.extract_feature_frame(frame, emitted, emitted * 1000.0)
            emitted += 1
        elapsed = max(time.time() - t0, 1.0e-9)
        print(f"[RS-AI] encoded_images={emitted} fps={emitted / elapsed:.2f}")

    write_stream(
        args.output,
        frames(),
        {
            "model": args.model,
            "split_layer": args.split_layer,
            "imgsz": args.imgsz,
            "tensor_codec": args.tensor_codec,
            "image_dir": str(image_dir),
            "pattern": args.pattern,
            "resource_mode": _resource_mode_for_args(args),
        },
    )
    print(f"[RS-AI] bitstream={args.output}")
    return 0


def _scan_viso(args: argparse.Namespace) -> int:
    sequences = discover_viso_sequences(args.root)
    payload = {
        "root": args.root,
        "sequences": [
            {
                "name": seq.name,
                "root": str(seq.root),
                "frames_dir": str(seq.frames_dir) if seq.frames_dir else None,
                "video_file": str(seq.video_file) if seq.video_file else None,
                "annotation_file": str(seq.annotation_file) if seq.annotation_file else None,
                "frame_count": seq.frame_count,
            }
            for seq in sequences
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _encode_viso(args: argparse.Namespace) -> int:
    source = _resolve_viso_source(args)
    if source["kind"] == "frames":
        image_args = argparse.Namespace(**_copy_common_encode_kwargs(args))
        image_args.image_dir = source["path"]
        image_args.pattern = args.pattern
        image_args.max_images = args.max_frames
        ret = _encode_images(image_args)
    else:
        video_args = argparse.Namespace(**_copy_common_encode_kwargs(args))
        video_args.video = source["path"]
        video_args.max_frames = args.max_frames
        video_args.stride = 1
        ret = _encode_video(video_args)
    manifest = {
        "dataset": "VISO",
        "source_kind": source["kind"],
        "source_path": source["path"],
        "annotation": args.annotation or source.get("annotation"),
        "model": args.model,
        "split_layer": args.split_layer,
        "bitstream": args.output,
        "max_frames": args.max_frames,
        "resource_mode": _resource_mode_for_args(args),
    }
    if args.manifest_json:
        Path(args.manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RS-AI] viso_manifest={args.manifest_json}")
    return ret


def _resolve_viso_source(args: argparse.Namespace) -> dict:
    if args.frames_dir:
        return {"kind": "frames", "path": args.frames_dir, "annotation": args.annotation or None}
    if args.video:
        return {"kind": "video", "path": args.video, "annotation": args.annotation or None}
    if args.sequence:
        seqs = discover_viso_sequences(args.sequence, max_depth=2)
        if not seqs:
            raise RuntimeError(f"no VISO frame/video sequence found under {args.sequence}")
        seq = seqs[0]
        if seq.frames_dir:
            return {"kind": "frames", "path": str(seq.frames_dir), "annotation": str(seq.annotation_file) if seq.annotation_file else args.annotation or None}
        if seq.video_file:
            return {"kind": "video", "path": str(seq.video_file), "annotation": str(seq.annotation_file) if seq.annotation_file else args.annotation or None}
    raise RuntimeError("encode-viso requires --frames-dir, --video, or --sequence")


def _decode_bitstream(args: argparse.Namespace, importance: bool = False) -> int:
    meta, frames = read_stream(args.input)
    split_layer = int(args.split_layer or meta.get("split_layer") or 22)
    resource_mode = str(meta.get("resource_mode", ""))
    use_importance = importance or resource_mode == "importance_aware_v1"
    use_layered = resource_mode == "roi_layered_v1"
    codec = _make_codec(args, split_layer=split_layer, force_importance=use_importance, force_layered=use_layered)
    results = []
    frame_count = 0
    t0 = time.time()
    for frame in frames:
        detections = codec.detect_from_feature_frame(frame, conf=args.conf, iou=args.iou, max_det=args.max_det)
        results.append(
            {
                "frame_id": frame.frame_id,
                "pts_ms": frame.pts_ms,
                "source_size": list(frame.source_size),
                "detections": [asdict(det) for det in detections],
            }
        )
        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
    elapsed = max(time.time() - t0, 1.0e-9)
    output = {
        "input_bitstream": args.input,
        "model": args.model,
        "split_layer": split_layer,
        "frames": results,
        "summary": {
            "frames": frame_count,
            "detections": sum(len(item["detections"]) for item in results),
            "fps": frame_count / elapsed,
        },
        "resource_mode": resource_mode or ("roi_layered_v1" if use_layered else ("importance_aware_v1" if use_importance else "uniform_v1")),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RS-AI] decoded_frames={frame_count} detections={output['summary']['detections']} fps={output['summary']['fps']:.2f}")
    print(f"[RS-AI] detections={args.output_json}")
    return 0


def _encode_video_importance(args: argparse.Namespace) -> int:
    return _encode_video(args)


def _encode_images_importance(args: argparse.Namespace) -> int:
    return _encode_images(args)


def _encode_viso_importance(args: argparse.Namespace) -> int:
    return _encode_viso(args)


def _encode_video_roi_layered(args: argparse.Namespace) -> int:
    return _encode_video(args)


def _encode_images_roi_layered(args: argparse.Namespace) -> int:
    return _encode_images(args)


def _encode_viso_roi_layered(args: argparse.Namespace) -> int:
    return _encode_viso(args)


def _is_importance_args(args: argparse.Namespace) -> bool:
    return hasattr(args, "bg_keep_ratio")


def _is_layered_args(args: argparse.Namespace) -> bool:
    return hasattr(args, "bg_downsample")


def _resource_mode_for_args(args: argparse.Namespace) -> str:
    if _is_layered_args(args):
        return "roi_layered_v1"
    if _is_importance_args(args):
        return "importance_aware_v1"
    return "uniform_v1"


def _make_codec(args: argparse.Namespace, split_layer: int | None = None, force_importance: bool = False, force_layered: bool = False):
    kwargs = {
        "model_path": args.model,
        "split_layer": int(args.split_layer if split_layer is None else split_layer),
        "device": args.device,
        "imgsz": args.imgsz,
        "tensor_codec": args.tensor_codec,
        "zlib_level": getattr(args, "zlib_level", 1),
        "half": args.half,
    }
    if force_layered or _is_layered_args(args):
        return LayeredRoiSplitYoloFeatureCodec(
            **kwargs,
            roi_conf=getattr(args, "roi_conf", 0.20),
            roi_margin=getattr(args, "roi_margin", 0.10),
            bg_downsample=getattr(args, "bg_downsample", 4),
            visual_downsample=getattr(args, "visual_downsample", 4),
            visual_base_quality=getattr(args, "visual_quality", 55),
        )
    if force_importance or _is_importance_args(args):
        return ImportanceAwareSplitYoloFeatureCodec(
            **kwargs,
            roi_conf=getattr(args, "roi_conf", 0.20),
            roi_margin=getattr(args, "roi_margin", 0.10),
            bg_keep_ratio=getattr(args, "bg_keep_ratio", 0.15),
            bg_scale_boost=getattr(args, "bg_scale_boost", 4.0),
        )
    return SplitYoloFeatureCodec(
        **kwargs,
        embed_visual_preview=getattr(args, "embed_visual_preview", False),
        visual_downsample=getattr(args, "visual_downsample", 1),
        visual_quality=getattr(args, "visual_quality", 85),
    )


def _copy_common_encode_kwargs(args: argparse.Namespace) -> dict:
    keys = [
        "model",
        "output",
        "split_layer",
        "device",
        "imgsz",
        "tensor_codec",
        "zlib_level",
        "half",
        "roi_conf",
        "roi_margin",
        "bg_keep_ratio",
        "bg_scale_boost",
        "bg_downsample",
        "embed_visual_preview",
        "visual_downsample",
        "visual_quality",
    ]
    out = {}
    for key in keys:
        if hasattr(args, key):
            out[key] = getattr(args, key)
    return out


def _write_mock(output: str, frames: int) -> None:
    def gen():
        for i in range(frames):
            payload = bytes((i + j) % 256 for j in range(256))
            yield FeatureFrame(
                frame_id=i,
                pts_ms=i * 40.0,
                source_size=(720, 1280),
                network_size=(640, 640),
                split_layer=22,
                tensors=[
                    TensorBlob(
                        name="mock_feature",
                        shape=[1, 1, 16, 16],
                        dtype="uint8",
                        encoding="raw",
                        data=payload,
                    )
                ],
                meta={"mock": True},
            )

    write_stream(output, gen(), {"mock": True, "split_layer": 22, "tensor_codec": "raw"})
    print(f"[RS-AI] mock_bitstream={output}")


def _bootstrap_vhr10_demo(output_dir: str, images: int) -> None:
    out = Path(output_dir)
    image_dir = out / "images"
    model_dir = out / "models"
    image_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_url = "https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10/resolve/main/best.pt"
    model_path = model_dir / "vhr10_yolov8n_best.pt"
    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        print(f"[RS-AI] downloading_model={model_url}")
        urlretrieve(model_url, model_path)

    base = "https://huggingface.co/datasets/satellite-image-deep-learning/VHR-10/resolve/main/VHR-10/positive image set"
    count = max(1, int(images))
    downloaded = []
    for idx in range(1, count + 1):
        name = f"{idx:03d}.jpg"
        path = image_dir / name
        if not path.exists() or path.stat().st_size == 0:
            url = f"{base}/{quote(name)}"
            url = url.replace("positive image set", "positive%20image%20set")
            print(f"[RS-AI] downloading_image={url}")
            urlretrieve(url, path)
        downloaded.append(str(path))

    manifest = {
        "task": "NWPU VHR-10 satellite remote-sensing object detection",
        "model_repo": "bluelabel/satellite-equipment-detection-yolov8n-vhr10",
        "dataset_repo": "satellite-image-deep-learning/VHR-10",
        "model_path": str(model_path),
        "image_dir": str(image_dir),
        "images": downloaded,
        "classes": [
            "airplane",
            "ship",
            "storage_tank",
            "baseball_diamond",
            "tennis_court",
            "basketball_court",
            "ground_track_field",
            "harbor",
            "bridge",
            "vehicle",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RS-AI] vhr10_demo={out}")
    print(f"[RS-AI] model={model_path}")
    print(f"[RS-AI] image_dir={image_dir}")


def _render_detections(args: argparse.Namespace) -> None:
    import cv2

    data = json.loads(Path(args.detections_json).read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if args.track:
        frames = assign_iou_tracks(frames, iou_threshold=args.track_iou)
        data["frames"] = frames
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_render_sources(args, len(frames))
    annotated = []
    for item in frames:
        frame_id = int(item["frame_id"])
        if frame_id not in sources:
            continue
        image = sources[frame_id].copy()
        detections = item.get("detections", [])
        _draw_detections(image, detections)
        out_path = output_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(out_path), image)
        annotated.append(str(out_path))

    if args.output_video:
        _write_video(args.output_video, [Path(p) for p in annotated], args.fps)

    metrics = _build_ui_metrics(data, annotated, args)
    if args.gt_annotation:
        gt = read_mot_like_annotations(args.gt_annotation)
        det_by_frame = {int(item.get("frame_id", 0)): item.get("detections", []) for item in frames}
        metrics["evaluation"] = evaluate_detections(det_by_frame, gt, iou_threshold=args.eval_iou)
    metrics_path = Path(args.metrics_json) if args.metrics_json else output_dir / "ui_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RS-AI] annotated_frames={len(annotated)} output_dir={output_dir}")
    print(f"[RS-AI] ui_metrics={metrics_path}")
    if args.output_video:
        print(f"[RS-AI] annotated_video={args.output_video}")


def _load_render_sources(args: argparse.Namespace, frame_count: int) -> dict[int, object]:
    import cv2

    sources = {}
    if args.image_dir:
        image_paths = sorted(Path(args.image_dir).glob(args.pattern))
        for idx, path in enumerate(image_paths[:frame_count]):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read image: {path}")
            sources[idx] = image
        return sources
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {args.video}")
        idx = 0
        while idx < frame_count:
            ok, image = cap.read()
            if not ok:
                break
            sources[idx] = image
            idx += 1
        cap.release()
        return sources
    if args.source_bitstream:
        return _load_visual_sources_from_bitstream(args.source_bitstream, frame_count)
    raise RuntimeError("render-detections requires --image-dir or --video")


def _load_visual_sources_from_bitstream(bitstream_path: str, frame_count: int) -> dict[int, object]:
    import cv2
    import numpy as np

    _meta, frames = read_stream(bitstream_path)
    sources: dict[int, object] = {}
    for idx, frame in enumerate(frames):
        if idx >= frame_count:
            break
        blobs = {blob.name: blob for blob in frame.tensors if blob.meta.get("auxiliary")}
        full_blob = blobs.get("__visual_full_jpeg")
        if full_blob is not None:
            image = cv2.imdecode(np.frombuffer(full_blob.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            src_h, src_w = [int(v) for v in full_blob.meta.get("source_shape", [image.shape[0], image.shape[1]])]
            if image.shape[0] != src_h or image.shape[1] != src_w:
                image = cv2.resize(image, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            sources[int(frame.frame_id)] = image
            continue
        base_blob = blobs.get("__visual_base_jpeg")
        roi_blob = blobs.get("__visual_roi_jpeg") or blobs.get("__visual_roi_png")
        mask_blob = blobs.get("__visual_roi_mask")
        if base_blob is None or roi_blob is None or mask_blob is None:
            continue
        base = cv2.imdecode(np.frombuffer(base_blob.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        roi = cv2.imdecode(np.frombuffer(roi_blob.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if base is None or roi is None:
            continue
        h, w = int(mask_blob.shape[0]), int(mask_blob.shape[1])
        raw = zlib.decompress(mask_blob.data)
        packed = np.frombuffer(raw, dtype=np.uint8)
        mask = np.unpackbits(packed, bitorder="little")[: h * w].reshape((h, w)).astype(bool, copy=False)
        base_up = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
        restored = base_up.copy()
        restored[mask] = roi[mask]
        sources[int(frame.frame_id)] = restored
    return sources


def _draw_detections(image: object, detections: list[dict]) -> None:
    import cv2

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        class_name = str(det.get("class_name", det.get("class_id", "object")))
        confidence = float(det.get("confidence", 0.0))
        color = _class_color(class_name)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        track = det.get("track_id")
        label = f"{class_name} {confidence:.2f}" if track is None else f"#{track} {class_name} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_label = max(0, y1 - th - baseline - 4)
        cv2.rectangle(image, (x1, y_label), (x1 + tw + 8, y_label + th + baseline + 6), color, -1)
        cv2.putText(image, label, (x1 + 4, y_label + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _class_color(name: str) -> tuple[int, int, int]:
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
        (22, 160, 133),
        (41, 128, 185),
    ]
    return palette[sum(name.encode("utf-8")) % len(palette)]


def _write_video(output_video: str, frame_paths: list[Path], fps: float) -> None:
    import cv2

    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read annotated frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 0.1), (w, h))
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()


def _build_ui_metrics(data: dict, annotated: list[str], args: argparse.Namespace) -> dict:
    frames = data.get("frames", [])
    class_counts: dict[str, int] = {}
    confidences = []
    per_frame = []
    track_ids = set()
    for item in frames:
        detections = item.get("detections", [])
        per_frame.append({"frame_id": item.get("frame_id"), "detections": len(detections)})
        for det in detections:
            class_name = str(det.get("class_name", det.get("class_id", "object")))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(float(det.get("confidence", 0.0)))
            if det.get("track_id") is not None:
                track_ids.add(int(det["track_id"]))
    tx_bytes = _file_size_or_none(args.tx_bitstream)
    rx_bytes = _file_size_or_none(args.rx_bitstream)
    return {
        "task": "remote_sensing_object_detection",
        "model": data.get("model"),
        "split_layer": data.get("split_layer"),
        "frames": len(frames),
        "detections": sum(class_counts.values()),
        "class_counts": class_counts,
        "confidence": {
            "avg": sum(confidences) / len(confidences) if confidences else 0.0,
            "max": max(confidences) if confidences else 0.0,
            "min": min(confidences) if confidences else 0.0,
        },
        "per_frame": per_frame,
        "tracks": {
            "enabled": len(track_ids) > 0,
            "count": len(track_ids),
        },
        "link_payload": {
            "tx_bitstream": args.tx_bitstream or None,
            "rx_bitstream": args.rx_bitstream or None,
            "tx_bytes": tx_bytes,
            "rx_bytes": rx_bytes,
            "byte_delta": (rx_bytes - tx_bytes) if tx_bytes is not None and rx_bytes is not None else None,
        },
        "assets": {
            "annotated_frames": annotated,
            "annotated_video": args.output_video or None,
            "detections_json": args.detections_json,
        },
    }


def _file_size_or_none(path: str) -> int | None:
    if not path:
        return None
    p = Path(path)
    return p.stat().st_size if p.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
