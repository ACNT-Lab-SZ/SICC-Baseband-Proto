from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

from .bitstream import FeatureFrame, TensorBlob


@dataclass
class Detection:
    frame_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]


@dataclass
class GpuTensorPacket:
    name: str
    shape: list[int]
    dtype: str
    encoding: str
    offset: int
    nbytes: int
    scale: float | None = None
    zero_point: int | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class GpuDirectFrame:
    frame_id: int
    pts_ms: float
    source_size: tuple[int, int]
    network_size: tuple[int, int]
    split_layer: int
    payload: Any
    packets: list[GpuTensorPacket]
    meta: dict = field(default_factory=dict)

    @property
    def payload_nbytes(self) -> int:
        return int(self.payload.numel())

    def manifest(self) -> dict:
        return {
            "protocol": "rs-ai-gpu-direct-v1",
            "frame_id": int(self.frame_id),
            "pts_ms": float(self.pts_ms),
            "source_size": [int(self.source_size[0]), int(self.source_size[1])],
            "network_size": [int(self.network_size[0]), int(self.network_size[1])],
            "split_layer": int(self.split_layer),
            "payload_nbytes": self.payload_nbytes,
            "packets": [
                {
                    "name": p.name,
                    "shape": [int(v) for v in p.shape],
                    "dtype": p.dtype,
                    "encoding": p.encoding,
                    "offset": int(p.offset),
                    "nbytes": int(p.nbytes),
                    "scale": None if p.scale is None else float(p.scale),
                    "zero_point": None if p.zero_point is None else int(p.zero_point),
                    "meta": p.meta,
                }
                for p in self.packets
            ],
            "meta": self.meta,
        }


def _missing_ml_error() -> RuntimeError:
    return RuntimeError(
        "YOLO split inference requires numpy, opencv-python, torch, and ultralytics. "
        "Install them on the 4090 server, then run this command again."
    )


class SplitYoloFeatureCodec:
    """Split an Ultralytics YOLO model into TX feature extraction and RX detection."""

    def __init__(
        self,
        model_path: str,
        split_layer: int,
        device: str = "cuda:0",
        imgsz: int = 640,
        tensor_codec: str = "int8",
        zlib_level: int = 1,
        half: bool = False,
        embed_visual_preview: bool = False,
        visual_downsample: int = 1,
        visual_quality: int = 85,
    ) -> None:
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
            import torch
            from ultralytics import YOLO
            from ultralytics.utils.ops import scale_boxes
            try:
                from ultralytics.utils.ops import non_max_suppression
            except ImportError:
                from ultralytics.utils.nms import non_max_suppression
        except Exception as exc:  # pragma: no cover - exercised on GPU host
            raise _missing_ml_error() from exc

        self.torch = torch
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.model_path = model_path
        self.split_layer = int(split_layer)
        self.device = torch.device(device)
        self.imgsz = int(imgsz)
        self.tensor_codec = tensor_codec.lower()
        self.zlib_level = int(zlib_level)
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model.to(self.device).eval()
        self.names = getattr(self.model, "names", getattr(self.yolo, "names", {}))
        if half:
            self.model.half()
        self.half = bool(half)
        self.layers = list(self.model.model)
        if self.split_layer <= 0 or self.split_layer >= len(self.layers):
            raise ValueError(f"split_layer must be in [1, {len(self.layers) - 1}]")
        if self.tensor_codec not in {"int8", "float16", "float32"}:
            raise ValueError("tensor_codec must be int8, float16, or float32")
        self.embed_visual_preview = bool(embed_visual_preview)
        self.visual_downsample = max(1, int(visual_downsample))
        self.visual_quality = max(5, min(95, int(visual_quality)))

    def preprocess_bgr(self, frame_bgr: Any) -> tuple[Any, dict]:
        import cv2
        import numpy as np

        h0, w0 = frame_bgr.shape[:2]
        scale = min(self.imgsz / h0, self.imgsz / w0)
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).contiguous()
        tensor = tensor.unsqueeze(0).float().div_(255.0)
        if self.half:
            tensor = tensor.half()
        meta = {
            "original_shape": [int(h0), int(w0)],
            "letterbox": {
                "scale": float(scale),
                "pad_xy": [int(pad_x), int(pad_y)],
                "resized_shape": [int(new_h), int(new_w)],
            },
        }
        return tensor, meta

    def extract_feature_frame(self, frame_bgr: Any, frame_id: int, pts_ms: float) -> FeatureFrame:
        x, prep_meta = self.preprocess_bgr(frame_bgr)
        x, y = self._run_prefix(x)
        tensors = [self._pack_tensor("x", x)]
        for idx, saved in enumerate(y):
            if saved is not None:
                tensors.append(self._pack_tensor(f"y_{idx}", saved))
        if self.embed_visual_preview:
            tensors.extend(self._pack_uniform_visual_preview(frame_bgr))
        source_h, source_w = prep_meta["original_shape"]
        return FeatureFrame(
            frame_id=int(frame_id),
            pts_ms=float(pts_ms),
            source_size=(source_h, source_w),
            network_size=(self.imgsz, self.imgsz),
            split_layer=self.split_layer,
            tensors=tensors,
            meta={
                "model": self.model_path,
                "tensor_codec": self.tensor_codec,
                "preprocess": prep_meta,
                "visual_preview": {
                    "enabled": self.embed_visual_preview,
                    "visual_downsample": self.visual_downsample,
                    "visual_quality": self.visual_quality,
                },
            },
        )

    def detect_from_feature_frame(
        self,
        frame: FeatureFrame,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
    ) -> list[Detection]:
        if frame.split_layer != self.split_layer:
            raise ValueError(f"bitstream split_layer={frame.split_layer}, decoder split_layer={self.split_layer}")
        tensor_map = {t.name: self._unpack_tensor(t) for t in frame.tensors if not t.meta.get("auxiliary")}
        if "x" not in tensor_map:
            raise ValueError("feature frame does not contain current tensor 'x'")
        x = tensor_map["x"]
        y: list[Any] = [None] * self.split_layer
        for name, tensor in tensor_map.items():
            if name.startswith("y_"):
                y[int(name[2:])] = tensor
        x, y = self._run_suffix(x, y)
        pred = x[0] if isinstance(x, (tuple, list)) else x
        nms = self.non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=max_det)
        det = nms[0]
        if det is None or len(det) == 0:
            return []
        original_shape = tuple(frame.meta.get("preprocess", {}).get("original_shape", list(frame.source_size)))
        det[:, :4] = self.scale_boxes((self.imgsz, self.imgsz), det[:, :4], original_shape).round()
        out: list[Detection] = []
        for row in det.detach().cpu().tolist():
            x1, y1, x2, y2, score, cls = row[:6]
            class_id = int(cls)
            out.append(
                Detection(
                    frame_id=frame.frame_id,
                    class_id=class_id,
                    class_name=str(self.names.get(class_id, class_id) if isinstance(self.names, dict) else class_id),
                    confidence=float(score),
                    xyxy=[float(x1), float(y1), float(x2), float(y2)],
                )
            )
        return out

    def extract_gpu_direct_frame(self, frame_bgr: Any, frame_id: int, pts_ms: float) -> GpuDirectFrame:
        x, prep_meta = self.preprocess_bgr(frame_bgr)
        x, y = self._run_prefix(x)
        packets: list[GpuTensorPacket] = []
        payload_parts: list[Any] = []
        offset = 0

        def add_packet(name: str, tensor: Any) -> None:
            nonlocal offset
            packet, raw = self._pack_tensor_device(name, tensor)
            packet.offset = offset
            offset += packet.nbytes
            packets.append(packet)
            payload_parts.append(raw)

        add_packet("x", x)
        for idx, saved in enumerate(y):
            if saved is not None:
                add_packet(f"y_{idx}", saved)

        payload = self.torch.cat(payload_parts, dim=0) if payload_parts else self.torch.empty((0,), dtype=self.torch.uint8, device=self.device)
        source_h, source_w = prep_meta["original_shape"]
        return GpuDirectFrame(
            frame_id=int(frame_id),
            pts_ms=float(pts_ms),
            source_size=(source_h, source_w),
            network_size=(self.imgsz, self.imgsz),
            split_layer=self.split_layer,
            payload=payload,
            packets=packets,
            meta={
                "model": self.model_path,
                "tensor_codec": self.tensor_codec,
                "preprocess": prep_meta,
                "resource_mode": "gpu_direct_v1",
            },
        )

    def detect_from_gpu_direct_frame(
        self,
        frame: GpuDirectFrame,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
    ) -> list[Detection]:
        if frame.split_layer != self.split_layer:
            raise ValueError(f"gpu-direct split_layer={frame.split_layer}, decoder split_layer={self.split_layer}")
        tensor_map = {packet.name: self._unpack_tensor_device(frame.payload, packet) for packet in frame.packets}
        if "x" not in tensor_map:
            raise ValueError("gpu-direct frame does not contain current tensor 'x'")
        x = tensor_map["x"]
        y: list[Any] = [None] * self.split_layer
        for name, tensor in tensor_map.items():
            if name.startswith("y_"):
                y[int(name[2:])] = tensor
        x, y = self._run_suffix(x, y)
        pred = x[0] if isinstance(x, (tuple, list)) else x
        nms = self.non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=max_det)
        det = nms[0]
        if det is None or len(det) == 0:
            return []
        original_shape = tuple(frame.meta.get("preprocess", {}).get("original_shape", list(frame.source_size)))
        det[:, :4] = self.scale_boxes((self.imgsz, self.imgsz), det[:, :4], original_shape).round()
        out: list[Detection] = []
        for row in det.detach().cpu().tolist():
            x1, y1, x2, y2, score, cls = row[:6]
            class_id = int(cls)
            out.append(
                Detection(
                    frame_id=frame.frame_id,
                    class_id=class_id,
                    class_name=str(self.names.get(class_id, class_id) if isinstance(self.names, dict) else class_id),
                    confidence=float(score),
                    xyxy=[float(x1), float(y1), float(x2), float(y2)],
                )
            )
        return out

    def gpu_direct_to_feature_frame(self, frame: GpuDirectFrame) -> FeatureFrame:
        tensors: list[TensorBlob] = []
        for packet in frame.packets:
            start = int(packet.offset)
            stop = start + int(packet.nbytes)
            raw = frame.payload[start:stop].detach().cpu().numpy().tobytes()
            if packet.encoding == "raw+symmetric_int8":
                arr = self.torch.frombuffer(raw, dtype=self.torch.int8).clone().cpu().numpy()
                data = zlib.compress(arr.tobytes(order="C"), self.zlib_level)
                tensors.append(
                    TensorBlob(
                        name=packet.name,
                        shape=packet.shape,
                        dtype="int8",
                        encoding="zlib+symmetric_int8",
                        data=data,
                        scale=packet.scale,
                        zero_point=packet.zero_point,
                        meta=dict(packet.meta),
                    )
                )
            elif packet.encoding == "raw+float16":
                arr = frame.payload[start:stop].view(self.torch.float16).detach().cpu().numpy()
                data = zlib.compress(arr.tobytes(order="C"), self.zlib_level)
                tensors.append(TensorBlob(name=packet.name, shape=packet.shape, dtype="float16", encoding="zlib+raw", data=data, meta=dict(packet.meta)))
            elif packet.encoding == "raw+float32":
                arr = frame.payload[start:stop].view(self.torch.float32).detach().cpu().numpy()
                data = zlib.compress(arr.tobytes(order="C"), self.zlib_level)
                tensors.append(TensorBlob(name=packet.name, shape=packet.shape, dtype="float32", encoding="zlib+raw", data=data, meta=dict(packet.meta)))
            else:
                raise ValueError(f"unsupported gpu-direct packet encoding: {packet.encoding}")
        return FeatureFrame(
            frame_id=frame.frame_id,
            pts_ms=frame.pts_ms,
            source_size=frame.source_size,
            network_size=frame.network_size,
            split_layer=frame.split_layer,
            tensors=tensors,
            meta=dict(frame.meta),
        )

    def _run_prefix(self, x: Any) -> tuple[Any, list[Any]]:
        y: list[Any] = []
        with self.torch.no_grad():
            for layer in self.layers[: self.split_layer]:
                if layer.f != -1:
                    x = self._layer_input(layer.f, x, y)
                x = layer(x)
                y.append(x if layer.i in self.model.save else None)
        return x, y

    def _run_suffix(self, x: Any, y: list[Any]) -> tuple[Any, list[Any]]:
        with self.torch.no_grad():
            for layer in self.layers[self.split_layer :]:
                if layer.f != -1:
                    x = self._layer_input(layer.f, x, y)
                x = layer(x)
                y.append(x if layer.i in self.model.save else None)
        return x, y

    def _layer_input(self, f: Any, x: Any, y: list[Any]) -> Any:
        if isinstance(f, int):
            return x if f == -1 else y[f]
        return [x if j == -1 else y[j] for j in f]

    def _pack_tensor(self, name: str, tensor: Any) -> TensorBlob:
        arr = tensor.detach().float().cpu().contiguous().numpy()
        if self.tensor_codec == "int8":
            max_abs = float(abs(arr).max()) if arr.size else 0.0
            scale = max(max_abs / 127.0, 1.0e-12)
            q = (arr / scale).round().clip(-127, 127).astype("int8")
            data = zlib.compress(q.tobytes(order="C"), self.zlib_level)
            return TensorBlob(name=name, shape=list(arr.shape), dtype="int8", encoding="zlib+symmetric_int8", data=data, scale=scale, zero_point=0)
        if self.tensor_codec == "float16":
            data = zlib.compress(arr.astype("float16").tobytes(order="C"), self.zlib_level)
            return TensorBlob(name=name, shape=list(arr.shape), dtype="float16", encoding="zlib+raw", data=data)
        data = zlib.compress(arr.astype("float32").tobytes(order="C"), self.zlib_level)
        return TensorBlob(name=name, shape=list(arr.shape), dtype="float32", encoding="zlib+raw", data=data)

    def _pack_tensor_device(self, name: str, tensor: Any) -> tuple[GpuTensorPacket, Any]:
        arr = tensor.detach().float().contiguous()
        if self.tensor_codec == "int8":
            max_abs = float(arr.abs().amax().item()) if arr.numel() else 0.0
            scale = max(max_abs / 127.0, 1.0e-12)
            q = (arr / scale).round().clamp_(-127, 127).to(self.torch.int8).contiguous()
            raw = q.view(self.torch.uint8).reshape(-1)
            return (
                GpuTensorPacket(
                    name=name,
                    shape=[int(v) for v in arr.shape],
                    dtype="int8",
                    encoding="raw+symmetric_int8",
                    offset=0,
                    nbytes=int(raw.numel()),
                    scale=scale,
                    zero_point=0,
                ),
                raw,
            )
        if self.tensor_codec == "float16":
            q = arr.to(self.torch.float16).contiguous()
            raw = q.view(self.torch.uint8).reshape(-1)
            return (
                GpuTensorPacket(
                    name=name,
                    shape=[int(v) for v in arr.shape],
                    dtype="float16",
                    encoding="raw+float16",
                    offset=0,
                    nbytes=int(raw.numel()),
                ),
                raw,
            )
        q = arr.to(self.torch.float32).contiguous()
        raw = q.view(self.torch.uint8).reshape(-1)
        return (
            GpuTensorPacket(
                name=name,
                shape=[int(v) for v in arr.shape],
                dtype="float32",
                encoding="raw+float32",
                offset=0,
                nbytes=int(raw.numel()),
            ),
            raw,
        )

    def _unpack_tensor(self, blob: TensorBlob) -> Any:
        import numpy as np

        if not blob.encoding.startswith("zlib+"):
            raise ValueError(f"unsupported tensor encoding: {blob.encoding}")
        raw = zlib.decompress(blob.data)
        arr = np.frombuffer(raw, dtype=np.dtype(blob.dtype)).reshape(tuple(blob.shape))
        if blob.encoding == "zlib+symmetric_int8":
            if blob.scale is None:
                raise ValueError(f"int8 tensor {blob.name} is missing scale")
            arr = arr.astype("float32") * float(blob.scale)
        else:
            arr = arr.astype("float32", copy=False)
        tensor = self.torch.from_numpy(arr.copy()).to(self.device)
        if self.half:
            tensor = tensor.half()
        return tensor

    def _unpack_tensor_device(self, payload: Any, packet: GpuTensorPacket) -> Any:
        start = int(packet.offset)
        stop = start + int(packet.nbytes)
        raw = payload[start:stop].contiguous()
        if packet.encoding == "raw+symmetric_int8":
            if packet.scale is None:
                raise ValueError(f"gpu-direct int8 tensor {packet.name} is missing scale")
            tensor = raw.view(self.torch.int8).to(self.torch.float32).reshape(packet.shape) * float(packet.scale)
        elif packet.encoding == "raw+float16":
            tensor = raw.view(self.torch.float16).reshape(packet.shape).to(self.torch.float32)
        elif packet.encoding == "raw+float32":
            tensor = raw.view(self.torch.float32).reshape(packet.shape)
        else:
            raise ValueError(f"unsupported gpu-direct packet encoding: {packet.encoding}")
        if self.half:
            tensor = tensor.half()
        return tensor

    def _pack_uniform_visual_preview(self, frame_bgr: Any) -> list[TensorBlob]:
        import cv2

        h, w = frame_bgr.shape[:2]
        if self.visual_downsample > 1:
            small_w = max(1, int(round(w / self.visual_downsample)))
            small_h = max(1, int(round(h / self.visual_downsample)))
            preview = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            preview = frame_bgr
            small_h, small_w = h, w
        ok, preview_buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), self.visual_quality])
        if not ok:
            raise RuntimeError("failed to encode uniform visual preview")
        return [
            TensorBlob(
                name="__visual_full_jpeg",
                shape=[int(small_h), int(small_w), 3],
                dtype="uint8",
                encoding="raw+jpeg",
                data=bytes(preview_buf),
                meta={"auxiliary": True, "role": "visual_full", "source_shape": [int(h), int(w)]},
            )
        ]


class ImportanceAwareSplitYoloFeatureCodec(SplitYoloFeatureCodec):
    """Task-driven split codec with ROI-priority transmission."""

    def __init__(
        self,
        model_path: str,
        split_layer: int,
        device: str = "cuda:0",
        imgsz: int = 640,
        tensor_codec: str = "int8",
        zlib_level: int = 1,
        half: bool = False,
        roi_conf: float = 0.20,
        roi_margin: float = 0.10,
        bg_keep_ratio: float = 0.15,
        bg_scale_boost: float = 4.0,
    ) -> None:
        super().__init__(
            model_path=model_path,
            split_layer=split_layer,
            device=device,
            imgsz=imgsz,
            tensor_codec=tensor_codec,
            zlib_level=zlib_level,
            half=half,
        )
        self.roi_conf = float(roi_conf)
        self.roi_margin = max(0.0, float(roi_margin))
        self.bg_keep_ratio = min(1.0, max(0.0, float(bg_keep_ratio)))
        self.bg_scale_boost = max(1.0, float(bg_scale_boost))

    def extract_feature_frame(self, frame_bgr: Any, frame_id: int, pts_ms: float) -> FeatureFrame:
        dense = super().extract_feature_frame(frame_bgr, frame_id, pts_ms)
        detections = super().detect_from_feature_frame(dense, conf=self.roi_conf)
        roi_boxes = [det.xyxy for det in detections]
        packed = [item for blob in dense.tensors for item in self._pack_importance_tensor(blob, roi_boxes, dense.source_size)]
        roi_pixels = self._roi_pixel_count(roi_boxes, dense.source_size)
        frame_meta = dict(dense.meta)
        frame_meta.update(
            {
                "resource_mode": "importance_aware_v1",
                "roi_generation": {
                    "source": "self_detection",
                    "roi_conf": self.roi_conf,
                    "roi_margin": self.roi_margin,
                    "bg_keep_ratio": self.bg_keep_ratio,
                    "bg_scale_boost": self.bg_scale_boost,
                    "roi_boxes": [[float(v) for v in box] for box in roi_boxes],
                    "roi_pixels": roi_pixels,
                },
                "resource_summary": self._summarize_importance_tensors(packed),
            }
        )
        return FeatureFrame(
            frame_id=dense.frame_id,
            pts_ms=dense.pts_ms,
            source_size=dense.source_size,
            network_size=dense.network_size,
            split_layer=dense.split_layer,
            tensors=packed,
            meta=frame_meta,
        )

    def detect_from_feature_frame(
        self,
        frame: FeatureFrame,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
    ) -> list[Detection]:
        if frame.meta.get("resource_mode") != "importance_aware_v1":
            return super().detect_from_feature_frame(frame, conf=conf, iou=iou, max_det=max_det)
        restored = FeatureFrame(
            frame_id=frame.frame_id,
            pts_ms=frame.pts_ms,
            source_size=frame.source_size,
            network_size=frame.network_size,
            split_layer=frame.split_layer,
            tensors=self._restore_dense_tensors(frame.tensors),
            meta=frame.meta,
        )
        return super().detect_from_feature_frame(restored, conf=conf, iou=iou, max_det=max_det)

    def _restore_dense_tensors(self, tensors: list[TensorBlob]) -> list[TensorBlob]:
        import numpy as np

        groups: dict[str, dict[str, TensorBlob]] = {}
        for blob in tensors:
            group = str(blob.meta.get("group", blob.name))
            role = str(blob.meta.get("role", "dense"))
            groups.setdefault(group, {})[role] = blob
        restored: list[TensorBlob] = []
        for name in sorted(groups):
            bundle = groups[name]
            if "dense" in bundle:
                restored.append(bundle["dense"])
                continue
            if not {"roi_idx", "roi_val", "bg_idx", "bg_val"} <= set(bundle):
                raise ValueError(f"incomplete importance-aware tensor bundle for {name}")
            roi_idx = self._decode_indices(bundle["roi_idx"])
            bg_idx = self._decode_indices(bundle["bg_idx"])
            roi_val = self._decode_values(bundle["roi_val"], scale_boost=1.0)
            bg_val = self._decode_values(bundle["bg_val"], scale_boost=float(bundle["bg_val"].meta.get("scale_boost", 1.0)))
            shape = tuple(int(v) for v in bundle["roi_val"].meta["full_shape"])
            flat = np.zeros(int(np.prod(shape)), dtype=np.float32)
            flat[roi_idx] = roi_val
            flat[bg_idx] = bg_val
            arr = flat.reshape(shape)
            restored.append(self._pack_tensor(name, self.torch.from_numpy(arr).to(self.device)))
        return restored

    def _pack_importance_tensor(self, blob: TensorBlob, roi_boxes: list[list[float]], source_size: tuple[int, int]) -> list[TensorBlob]:
        import numpy as np

        arr = self._unpack_tensor(blob).detach().float().cpu().numpy()
        if arr.ndim != 4 or arr.shape[0] != 1:
            dense_blob = TensorBlob(name=blob.name, shape=blob.shape, dtype=blob.dtype, encoding=blob.encoding, data=blob.data, scale=blob.scale, zero_point=blob.zero_point, meta={"group": blob.name, "role": "dense"})
            return [dense_blob]
        mask = self._build_roi_mask(arr.shape[2], arr.shape[3], roi_boxes, source_size)
        expanded = np.broadcast_to(mask[None, None, :, :], arr.shape).reshape(-1)
        flat = arr.reshape(-1)
        roi_idx = np.flatnonzero(expanded)
        bg_idx_all = np.flatnonzero(~expanded)
        if roi_idx.size == 0:
            roi_idx = np.arange(flat.size, dtype=np.int64)
            bg_idx_all = np.empty(0, dtype=np.int64)
        bg_idx = self._select_background_indices(flat, bg_idx_all)
        roi_val = flat[roi_idx]
        bg_val = flat[bg_idx]
        shape_list = [int(v) for v in arr.shape]
        return [
            self._encode_indices(f"{blob.name}__roi_idx", blob.name, "roi_idx", roi_idx),
            self._encode_values(f"{blob.name}__roi_val", blob.name, "roi_val", roi_val, shape_list, scale_boost=1.0, priority="high"),
            self._encode_indices(f"{blob.name}__bg_idx", blob.name, "bg_idx", bg_idx),
            self._encode_values(f"{blob.name}__bg_val", blob.name, "bg_val", bg_val, shape_list, scale_boost=self.bg_scale_boost, priority="low"),
        ]

    def _build_roi_mask(self, h: int, w: int, roi_boxes: list[list[float]], source_size: tuple[int, int]) -> Any:
        import numpy as np

        src_h, src_w = source_size
        mask = np.zeros((h, w), dtype=bool)
        if not roi_boxes:
            mask[:, :] = True
            return mask
        margin_x = self.roi_margin * src_w
        margin_y = self.roi_margin * src_h
        for box in roi_boxes:
            x1, y1, x2, y2 = box
            x1 = max(0.0, x1 - margin_x)
            y1 = max(0.0, y1 - margin_y)
            x2 = min(float(src_w), x2 + margin_x)
            y2 = min(float(src_h), y2 + margin_y)
            fx1 = int(max(0, min(w - 1, round((x1 / max(src_w, 1)) * (w - 1)))))
            fx2 = int(max(0, min(w, round((x2 / max(src_w, 1)) * w))))
            fy1 = int(max(0, min(h - 1, round((y1 / max(src_h, 1)) * (h - 1)))))
            fy2 = int(max(0, min(h, round((y2 / max(src_h, 1)) * h))))
            mask[fy1:max(fy1 + 1, fy2), fx1:max(fx1 + 1, fx2)] = True
        if not mask.any():
            mask[:, :] = True
        return mask

    def _select_background_indices(self, flat: Any, bg_idx_all: Any) -> Any:
        import numpy as np

        if bg_idx_all.size == 0 or self.bg_keep_ratio >= 1.0:
            return bg_idx_all.astype(np.int64, copy=False)
        keep = max(1, int(round(bg_idx_all.size * self.bg_keep_ratio)))
        scores = np.abs(flat[bg_idx_all])
        if keep >= bg_idx_all.size:
            return bg_idx_all.astype(np.int64, copy=False)
        topk_local = np.argpartition(scores, -keep)[-keep:]
        return np.sort(bg_idx_all[topk_local].astype(np.int64, copy=False))

    def _encode_indices(self, name: str, group: str, role: str, indices: Any) -> TensorBlob:
        import numpy as np

        raw = zlib.compress(np.asarray(indices, dtype=np.uint32).tobytes(order="C"), self.zlib_level)
        return TensorBlob(
            name=name,
            shape=[int(len(indices))],
            dtype="uint32",
            encoding="zlib+raw",
            data=raw,
            meta={"group": group, "role": role},
        )

    def _encode_values(
        self,
        name: str,
        group: str,
        role: str,
        values: Any,
        full_shape: list[int],
        scale_boost: float,
        priority: str,
    ) -> TensorBlob:
        import numpy as np

        arr = np.asarray(values, dtype=np.float32)
        meta = {"group": group, "role": role, "full_shape": full_shape, "priority": priority}
        if self.tensor_codec == "int8":
            max_abs = float(np.abs(arr).max()) if arr.size else 0.0
            scale = max((max_abs / 127.0) * scale_boost, 1.0e-12)
            q = (arr / scale).round().clip(-127, 127).astype("int8")
            data = zlib.compress(q.tobytes(order="C"), self.zlib_level)
            meta["scale_boost"] = float(scale_boost)
            return TensorBlob(name=name, shape=[int(arr.size)], dtype="int8", encoding="zlib+symmetric_int8", data=data, scale=scale, zero_point=0, meta=meta)
        if self.tensor_codec == "float16":
            data = zlib.compress(arr.astype("float16").tobytes(order="C"), self.zlib_level)
            return TensorBlob(name=name, shape=[int(arr.size)], dtype="float16", encoding="zlib+raw", data=data, meta=meta)
        data = zlib.compress(arr.astype("float32").tobytes(order="C"), self.zlib_level)
        return TensorBlob(name=name, shape=[int(arr.size)], dtype="float32", encoding="zlib+raw", data=data, meta=meta)

    def _decode_indices(self, blob: TensorBlob) -> Any:
        import numpy as np

        raw = zlib.decompress(blob.data)
        return np.frombuffer(raw, dtype=np.uint32).astype(np.int64, copy=False)

    def _decode_values(self, blob: TensorBlob, scale_boost: float) -> Any:
        import numpy as np

        raw = zlib.decompress(blob.data)
        arr = np.frombuffer(raw, dtype=np.dtype(blob.dtype))
        if blob.encoding == "zlib+symmetric_int8":
            if blob.scale is None:
                raise ValueError(f"importance tensor {blob.name} is missing scale")
            arr = arr.astype("float32") * float(blob.scale)
        else:
            arr = arr.astype("float32", copy=False)
        return arr

    def _roi_pixel_count(self, roi_boxes: list[list[float]], source_size: tuple[int, int]) -> int:
        src_h, src_w = source_size
        total = 0.0
        for box in roi_boxes:
            x1, y1, x2, y2 = box
            total += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return int(min(total, float(src_h * src_w)))

    def _summarize_importance_tensors(self, tensors: list[TensorBlob]) -> dict:
        roi_values = 0
        bg_values = 0
        roi_bytes = 0
        bg_bytes = 0
        for blob in tensors:
            role = str(blob.meta.get("role", ""))
            if role == "roi_val":
                roi_values += int(blob.shape[0]) if blob.shape else 0
                roi_bytes += len(blob.data)
            elif role == "bg_val":
                bg_values += int(blob.shape[0]) if blob.shape else 0
                bg_bytes += len(blob.data)
        total_values = roi_values + bg_values
        return {
            "roi_values": roi_values,
            "background_values": bg_values,
            "background_keep_ratio_effective": (bg_values / total_values) if total_values else 0.0,
            "roi_bytes": roi_bytes,
            "background_bytes": bg_bytes,
            "payload_bytes": roi_bytes + bg_bytes,
        }


class LayeredRoiSplitYoloFeatureCodec(SplitYoloFeatureCodec):
    """Task-driven file codec with coarse global background and full-resolution ROI enhancement."""

    def __init__(
        self,
        model_path: str,
        split_layer: int,
        device: str = "cuda:0",
        imgsz: int = 640,
        tensor_codec: str = "int8",
        zlib_level: int = 1,
        half: bool = False,
        roi_conf: float = 0.20,
        roi_margin: float = 0.10,
        bg_downsample: int = 4,
        visual_downsample: int = 4,
        visual_base_quality: int = 55,
    ) -> None:
        super().__init__(
            model_path=model_path,
            split_layer=split_layer,
            device=device,
            imgsz=imgsz,
            tensor_codec=tensor_codec,
            zlib_level=zlib_level,
            half=half,
        )
        self.roi_conf = float(roi_conf)
        self.roi_margin = max(0.0, float(roi_margin))
        self.bg_downsample = max(1, int(bg_downsample))
        self.visual_downsample = max(1, int(visual_downsample))
        self.visual_base_quality = max(5, min(95, int(visual_base_quality)))
        self.visual_roi_quality = max(self.visual_base_quality, 88)

    def extract_feature_frame(self, frame_bgr: Any, frame_id: int, pts_ms: float) -> FeatureFrame:
        dense = super().extract_feature_frame(frame_bgr, frame_id, pts_ms)
        detections = super().detect_from_feature_frame(dense, conf=self.roi_conf)
        roi_boxes = [det.xyxy for det in detections]
        packed = [item for blob in dense.tensors for item in self._pack_layered_tensor(blob, roi_boxes, dense.source_size)]
        packed.extend(self._pack_visual_preview(frame_bgr, roi_boxes))
        frame_meta = dict(dense.meta)
        frame_meta.update(
            {
                "resource_mode": "roi_layered_v1",
                "roi_generation": {
                    "source": "self_detection",
                    "roi_conf": self.roi_conf,
                    "roi_margin": self.roi_margin,
                    "bg_downsample": self.bg_downsample,
                    "visual_downsample": self.visual_downsample,
                    "roi_boxes": [[float(v) for v in box] for box in roi_boxes],
                },
                "resource_summary": self._summarize_layered_tensors(packed),
            }
        )
        return FeatureFrame(
            frame_id=dense.frame_id,
            pts_ms=dense.pts_ms,
            source_size=dense.source_size,
            network_size=dense.network_size,
            split_layer=dense.split_layer,
            tensors=packed,
            meta=frame_meta,
        )

    def extract_gpu_direct_frame(self, frame_bgr: Any, frame_id: int, pts_ms: float) -> GpuDirectFrame:
        x, prep_meta = self.preprocess_bgr(frame_bgr)
        x, y = self._run_prefix(x)
        dense_gpu = self._make_dense_gpu_frame(
            tensors=[("x", x)] + [(f"y_{idx}", saved) for idx, saved in enumerate(y) if saved is not None],
            frame_id=frame_id,
            pts_ms=pts_ms,
            prep_meta=prep_meta,
            resource_mode="gpu_direct_v1",
        )
        detections = SplitYoloFeatureCodec.detect_from_gpu_direct_frame(self, dense_gpu, conf=self.roi_conf)
        roi_boxes = [det.xyxy for det in detections]
        packets: list[GpuTensorPacket] = []
        payload_parts: list[Any] = []
        offset = 0

        def add_packet(packet: GpuTensorPacket, raw: Any) -> None:
            nonlocal offset
            packet.offset = offset
            offset += packet.nbytes
            packets.append(packet)
            payload_parts.append(raw)

        for name, tensor in [("x", x)] + [(f"y_{idx}", saved) for idx, saved in enumerate(y) if saved is not None]:
            for packet, raw in self._pack_layered_tensor_device(name, tensor, roi_boxes, tuple(prep_meta["original_shape"])):
                add_packet(packet, raw)

        payload = self.torch.cat(payload_parts, dim=0) if payload_parts else self.torch.empty((0,), dtype=self.torch.uint8, device=self.device)
        source_h, source_w = prep_meta["original_shape"]
        return GpuDirectFrame(
            frame_id=int(frame_id),
            pts_ms=float(pts_ms),
            source_size=(source_h, source_w),
            network_size=(self.imgsz, self.imgsz),
            split_layer=self.split_layer,
            payload=payload,
            packets=packets,
            meta={
                "model": self.model_path,
                "tensor_codec": self.tensor_codec,
                "preprocess": prep_meta,
                "resource_mode": "gpu_direct_roi_layered_v1",
                "roi_generation": {
                    "source": "self_detection",
                    "roi_conf": self.roi_conf,
                    "roi_margin": self.roi_margin,
                    "bg_downsample": self.bg_downsample,
                    "roi_boxes": [[float(v) for v in box] for box in roi_boxes],
                },
            },
        )

    def detect_from_feature_frame(
        self,
        frame: FeatureFrame,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
    ) -> list[Detection]:
        if frame.meta.get("resource_mode") != "roi_layered_v1":
            return super().detect_from_feature_frame(frame, conf=conf, iou=iou, max_det=max_det)
        restored = FeatureFrame(
            frame_id=frame.frame_id,
            pts_ms=frame.pts_ms,
            source_size=frame.source_size,
            network_size=frame.network_size,
            split_layer=frame.split_layer,
            tensors=self._restore_layered_tensors(frame.tensors),
            meta=frame.meta,
        )
        return super().detect_from_feature_frame(restored, conf=conf, iou=iou, max_det=max_det)

    def detect_from_gpu_direct_frame(
        self,
        frame: GpuDirectFrame,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
    ) -> list[Detection]:
        if frame.meta.get("resource_mode") != "gpu_direct_roi_layered_v1":
            return super().detect_from_gpu_direct_frame(frame, conf=conf, iou=iou, max_det=max_det)
        restored_tensors = self._restore_layered_gpu_tensors(frame)
        dense_gpu = self._make_dense_gpu_frame(
            tensors=restored_tensors,
            frame_id=frame.frame_id,
            pts_ms=frame.pts_ms,
            prep_meta=frame.meta.get("preprocess", {}),
            resource_mode="gpu_direct_v1",
        )
        return SplitYoloFeatureCodec.detect_from_gpu_direct_frame(self, dense_gpu, conf=conf, iou=iou, max_det=max_det)

    def _make_dense_gpu_frame(
        self,
        tensors: list[tuple[str, Any]],
        frame_id: int,
        pts_ms: float,
        prep_meta: dict,
        resource_mode: str,
    ) -> GpuDirectFrame:
        packets: list[GpuTensorPacket] = []
        payload_parts: list[Any] = []
        offset = 0
        for name, tensor in tensors:
            packet, raw = self._pack_tensor_device(name, tensor)
            packet.offset = offset
            offset += packet.nbytes
            packets.append(packet)
            payload_parts.append(raw)
        payload = self.torch.cat(payload_parts, dim=0) if payload_parts else self.torch.empty((0,), dtype=self.torch.uint8, device=self.device)
        source_h, source_w = prep_meta.get("original_shape", [0, 0])
        return GpuDirectFrame(
            frame_id=int(frame_id),
            pts_ms=float(pts_ms),
            source_size=(int(source_h), int(source_w)),
            network_size=(self.imgsz, self.imgsz),
            split_layer=self.split_layer,
            payload=payload,
            packets=packets,
            meta={
                "model": self.model_path,
                "tensor_codec": self.tensor_codec,
                "preprocess": prep_meta,
                "resource_mode": resource_mode,
            },
        )

    def _pack_layered_tensor(self, blob: TensorBlob, roi_boxes: list[list[float]], source_size: tuple[int, int]) -> list[TensorBlob]:
        import numpy as np

        arr = self._unpack_tensor(blob).detach().float().cpu().numpy()
        if arr.ndim != 4 or arr.shape[0] != 1:
            dense_blob = TensorBlob(
                name=blob.name,
                shape=blob.shape,
                dtype=blob.dtype,
                encoding=blob.encoding,
                data=blob.data,
                scale=blob.scale,
                zero_point=blob.zero_point,
                meta={"group": blob.name, "role": "dense"},
            )
            return [dense_blob]

        _, _, h, w = arr.shape
        mask = self._build_roi_mask(h, w, roi_boxes, source_size)
        base_small = self._downsample_feature(arr)
        roi_dense = arr * mask[None, None, :, :].astype(np.float32)
        shape_list = [int(v) for v in arr.shape]
        return [
            self._encode_dense_array(f"{blob.name}__base", blob.name, "base", base_small, priority="low", extra_meta={"full_shape": shape_list, "bg_downsample": self.bg_downsample}),
            self._encode_dense_array(f"{blob.name}__roi", blob.name, "roi", roi_dense, priority="high", extra_meta={"full_shape": shape_list}),
            self._encode_mask(f"{blob.name}__mask", blob.name, "mask", mask),
        ]

    def _pack_layered_tensor_device(
        self,
        name: str,
        tensor: Any,
        roi_boxes: list[list[float]],
        source_size: tuple[int, int],
    ) -> list[tuple[GpuTensorPacket, Any]]:
        import torch.nn.functional as F

        values = tensor.detach().float().contiguous()
        if values.ndim != 4 or values.shape[0] != 1:
            packet, raw = self._pack_tensor_device(name, values)
            packet.meta = {"group": name, "role": "dense", "priority": "normal"}
            return [(packet, raw)]

        _, _, h, w = values.shape
        mask_np = self._build_roi_mask(int(h), int(w), roi_boxes, source_size)
        mask = self.torch.from_numpy(mask_np).to(self.device)
        out_h = max(1, int(h) // self.bg_downsample)
        out_w = max(1, int(w) // self.bg_downsample)
        base_small = F.interpolate(values, size=(out_h, out_w), mode="bilinear", align_corners=False)
        mask_flat = mask.reshape(-1)
        roi_values = values.reshape(values.shape[0], values.shape[1], -1)[:, :, mask_flat]

        base_packet, base_raw = self._pack_tensor_device(f"{name}__base", base_small)
        base_packet.meta = {
            "group": name,
            "role": "base",
            "priority": "low",
            "full_shape": [int(v) for v in values.shape],
            "bg_downsample": self.bg_downsample,
        }
        roi_packet, roi_raw = self._pack_tensor_device(f"{name}__roi", roi_values)
        roi_packet.meta = {
            "group": name,
            "role": "roi",
            "priority": "high",
            "full_shape": [int(v) for v in values.shape],
            "roi_compact": True,
        }
        mask_raw = mask.to(self.torch.uint8).contiguous().reshape(-1)
        mask_packet = GpuTensorPacket(
            name=f"{name}__mask",
            shape=[int(h), int(w)],
            dtype="uint8",
            encoding="raw+mask_u8",
            offset=0,
            nbytes=int(mask_raw.numel()),
            meta={"group": name, "role": "mask", "priority": "control"},
        )
        return [(base_packet, base_raw), (roi_packet, roi_raw), (mask_packet, mask_raw)]

    def _restore_layered_gpu_tensors(self, frame: GpuDirectFrame) -> list[tuple[str, Any]]:
        import torch.nn.functional as F

        groups: dict[str, dict[str, GpuTensorPacket]] = {}
        for packet in frame.packets:
            group = str(packet.meta.get("group", packet.name))
            role = str(packet.meta.get("role", "dense"))
            groups.setdefault(group, {})[role] = packet

        restored: list[tuple[str, Any]] = []
        for name in sorted(groups):
            bundle = groups[name]
            if "dense" in bundle:
                restored.append((name, self._unpack_tensor_device(frame.payload, bundle["dense"])))
                continue
            if not {"base", "roi", "mask"} <= set(bundle):
                raise ValueError(f"incomplete gpu-direct roi-layered tensor bundle for {name}")
            base_small = self._unpack_tensor_device(frame.payload, bundle["base"])
            roi_values = self._unpack_tensor_device(frame.payload, bundle["roi"])
            mask_packet = bundle["mask"]
            start = int(mask_packet.offset)
            stop = start + int(mask_packet.nbytes)
            h, w = int(mask_packet.shape[0]), int(mask_packet.shape[1])
            mask = frame.payload[start:stop].contiguous().view(self.torch.uint8).reshape((h, w)).to(self.device).bool()
            base_up = F.interpolate(base_small.float(), size=(h, w), mode="bilinear", align_corners=False)
            fused_flat = base_up.reshape(base_up.shape[0], base_up.shape[1], -1).clone()
            fused_flat[:, :, mask.reshape(-1)] = roi_values.float()
            fused = fused_flat.reshape(base_up.shape)
            restored.append((name, fused))
        return restored

    def _restore_layered_tensors(self, tensors: list[TensorBlob]) -> list[TensorBlob]:
        import numpy as np

        groups: dict[str, dict[str, TensorBlob]] = {}
        for blob in tensors:
            if blob.meta.get("auxiliary"):
                continue
            group = str(blob.meta.get("group", blob.name))
            role = str(blob.meta.get("role", "dense"))
            groups.setdefault(group, {})[role] = blob
        restored: list[TensorBlob] = []
        for name in sorted(groups):
            bundle = groups[name]
            if "dense" in bundle:
                restored.append(bundle["dense"])
                continue
            if not {"base", "roi", "mask"} <= set(bundle):
                raise ValueError(f"incomplete roi-layered tensor bundle for {name}")
            base_small = self._decode_dense_array(bundle["base"])
            roi_dense = self._decode_dense_array(bundle["roi"])
            mask = self._decode_mask(bundle["mask"])
            full_shape = tuple(int(v) for v in bundle["roi"].meta["full_shape"])
            base_up = self._upsample_feature(base_small, full_shape[2], full_shape[3])
            mask4 = mask[None, None, :, :].astype(np.float32)
            fused = base_up * (1.0 - mask4) + roi_dense
            restored.append(self._pack_tensor(name, self.torch.from_numpy(fused.astype(np.float32, copy=False)).to(self.device)))
        return restored

    def _build_roi_mask(self, h: int, w: int, roi_boxes: list[list[float]], source_size: tuple[int, int]) -> Any:
        import numpy as np

        src_h, src_w = source_size
        mask = np.zeros((h, w), dtype=bool)
        if not roi_boxes:
            mask[:, :] = True
            return mask
        margin_x = self.roi_margin * src_w
        margin_y = self.roi_margin * src_h
        for box in roi_boxes:
            x1, y1, x2, y2 = box
            x1 = max(0.0, x1 - margin_x)
            y1 = max(0.0, y1 - margin_y)
            x2 = min(float(src_w), x2 + margin_x)
            y2 = min(float(src_h), y2 + margin_y)
            fx1 = int(max(0, min(w - 1, round((x1 / max(src_w, 1)) * (w - 1)))))
            fx2 = int(max(0, min(w, round((x2 / max(src_w, 1)) * w))))
            fy1 = int(max(0, min(h - 1, round((y1 / max(src_h, 1)) * (h - 1)))))
            fy2 = int(max(0, min(h, round((y2 / max(src_h, 1)) * h))))
            mask[fy1:max(fy1 + 1, fy2), fx1:max(fx1 + 1, fx2)] = True
        if not mask.any():
            mask[:, :] = True
        return mask

    def _downsample_feature(self, arr: Any) -> Any:
        if self.bg_downsample <= 1:
            return arr.astype("float32", copy=False)
        import torch.nn.functional as F

        tensor = self.torch.from_numpy(arr).float()
        h, w = int(arr.shape[2]), int(arr.shape[3])
        out_h = max(1, h // self.bg_downsample)
        out_w = max(1, w // self.bg_downsample)
        down = F.interpolate(tensor, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return down.cpu().numpy().astype("float32", copy=False)

    def _upsample_feature(self, arr: Any, out_h: int, out_w: int) -> Any:
        import torch.nn.functional as F

        tensor = self.torch.from_numpy(arr).float()
        up = F.interpolate(tensor, size=(int(out_h), int(out_w)), mode="bilinear", align_corners=False)
        return up.cpu().numpy().astype("float32", copy=False)

    def _encode_dense_array(
        self,
        name: str,
        group: str,
        role: str,
        arr: Any,
        priority: str,
        extra_meta: dict | None = None,
    ) -> TensorBlob:
        import numpy as np

        values = np.asarray(arr, dtype=np.float32)
        meta = {"group": group, "role": role, "priority": priority}
        if extra_meta:
            meta.update(extra_meta)
        if self.tensor_codec == "int8":
            max_abs = float(np.abs(values).max()) if values.size else 0.0
            scale = max(max_abs / 127.0, 1.0e-12)
            q = (values / scale).round().clip(-127, 127).astype("int8")
            data = zlib.compress(q.tobytes(order="C"), self.zlib_level)
            return TensorBlob(name=name, shape=[int(v) for v in values.shape], dtype="int8", encoding="zlib+symmetric_int8", data=data, scale=scale, zero_point=0, meta=meta)
        if self.tensor_codec == "float16":
            data = zlib.compress(values.astype("float16").tobytes(order="C"), self.zlib_level)
            return TensorBlob(name=name, shape=[int(v) for v in values.shape], dtype="float16", encoding="zlib+raw", data=data, meta=meta)
        data = zlib.compress(values.astype("float32").tobytes(order="C"), self.zlib_level)
        return TensorBlob(name=name, shape=[int(v) for v in values.shape], dtype="float32", encoding="zlib+raw", data=data, meta=meta)

    def _decode_dense_array(self, blob: TensorBlob) -> Any:
        import numpy as np

        raw = zlib.decompress(blob.data)
        arr = np.frombuffer(raw, dtype=np.dtype(blob.dtype)).reshape(tuple(int(v) for v in blob.shape))
        if blob.encoding == "zlib+symmetric_int8":
            if blob.scale is None:
                raise ValueError(f"roi-layered tensor {blob.name} is missing scale")
            arr = arr.astype("float32") * float(blob.scale)
        else:
            arr = arr.astype("float32", copy=False)
        return arr

    def _encode_mask(self, name: str, group: str, role: str, mask: Any) -> TensorBlob:
        import numpy as np

        packed = np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1), bitorder="little")
        data = zlib.compress(packed.tobytes(order="C"), self.zlib_level)
        return TensorBlob(
            name=name,
            shape=[int(mask.shape[0]), int(mask.shape[1])],
            dtype="uint8",
            encoding="zlib+packbits",
            data=data,
            meta={"group": group, "role": role},
        )

    def _decode_mask(self, blob: TensorBlob) -> Any:
        import numpy as np

        raw = zlib.decompress(blob.data)
        packed = np.frombuffer(raw, dtype=np.uint8)
        h, w = int(blob.shape[0]), int(blob.shape[1])
        mask = np.unpackbits(packed, bitorder="little")[: h * w].reshape((h, w))
        return mask.astype(bool, copy=False)

    def _summarize_layered_tensors(self, tensors: list[TensorBlob]) -> dict:
        base_bytes = 0
        roi_bytes = 0
        mask_bytes = 0
        visual_bytes = 0
        for blob in tensors:
            if blob.meta.get("auxiliary"):
                visual_bytes += len(blob.data)
                continue
            role = str(blob.meta.get("role", ""))
            if role == "base":
                base_bytes += len(blob.data)
            elif role == "roi":
                roi_bytes += len(blob.data)
            elif role == "mask":
                mask_bytes += len(blob.data)
        return {
            "base_bytes": base_bytes,
            "roi_bytes": roi_bytes,
            "mask_bytes": mask_bytes,
            "visual_bytes": visual_bytes,
            "payload_bytes": base_bytes + roi_bytes + mask_bytes + visual_bytes,
            "bg_downsample": self.bg_downsample,
        }

    def _pack_visual_preview(self, frame_bgr: Any, roi_boxes: list[list[float]]) -> list[TensorBlob]:
        import cv2
        import numpy as np

        h, w = frame_bgr.shape[:2]
        small_w = max(1, int(round(w / self.visual_downsample)))
        small_h = max(1, int(round(h / self.visual_downsample)))
        base = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        ok, base_buf = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), self.visual_base_quality])
        if not ok:
            raise RuntimeError("failed to encode visual base preview")

        mask = np.zeros((h, w), dtype=np.uint8)
        if roi_boxes:
            margin_x = self.roi_margin * w
            margin_y = self.roi_margin * h
            for box in roi_boxes:
                x1, y1, x2, y2 = box
                x1 = max(0, int(round(x1 - margin_x)))
                y1 = max(0, int(round(y1 - margin_y)))
                x2 = min(w, int(round(x2 + margin_x)))
                y2 = min(h, int(round(y2 + margin_y)))
                mask[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)] = 255
        else:
            mask[:, :] = 255
        roi = np.zeros_like(frame_bgr)
        roi[mask > 0] = frame_bgr[mask > 0]
        ok, roi_buf = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), self.visual_roi_quality])
        if not ok:
            raise RuntimeError("failed to encode visual roi preview")
        packed_mask = np.packbits((mask.reshape(-1) > 0).astype(np.uint8), bitorder="little")

        return [
            TensorBlob(
                name="__visual_base_jpeg",
                shape=[small_h, small_w, 3],
                dtype="uint8",
                encoding="raw+jpeg",
                data=bytes(base_buf),
                meta={"auxiliary": True, "role": "visual_base", "source_shape": [h, w]},
            ),
            TensorBlob(
                name="__visual_roi_jpeg",
                shape=[h, w, 3],
                dtype="uint8",
                encoding="raw+jpeg",
                data=bytes(roi_buf),
                meta={"auxiliary": True, "role": "visual_roi", "source_shape": [h, w], "visual_quality": self.visual_roi_quality},
            ),
            TensorBlob(
                name="__visual_roi_mask",
                shape=[h, w],
                dtype="uint8",
                encoding="raw+packbits",
                data=zlib.compress(packed_mask.tobytes(order="C"), self.zlib_level),
                meta={"auxiliary": True, "role": "visual_mask", "source_shape": [h, w]},
            ),
        ]
