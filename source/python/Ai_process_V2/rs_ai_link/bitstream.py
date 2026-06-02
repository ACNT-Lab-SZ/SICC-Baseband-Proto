from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Sequence


STREAM_MAGIC = b"RSBF"
FRAME_MAGIC = b"RSFR"
VERSION = 1

_STREAM_HEADER = struct.Struct("<4sHHII")
_FRAME_HEADER = struct.Struct("<4sQQIIII")


class BitstreamError(RuntimeError):
    """Raised when an RSBF feature bitstream is malformed."""


@dataclass
class TensorBlob:
    name: str
    shape: Sequence[int]
    dtype: str
    encoding: str
    data: bytes
    scale: float | None = None
    zero_point: int | None = None
    meta: dict = field(default_factory=dict)

    def to_meta(self, offset: int) -> dict:
        meta = {
            "name": self.name,
            "shape": [int(v) for v in self.shape],
            "dtype": self.dtype,
            "encoding": self.encoding,
            "offset": int(offset),
            "nbytes": len(self.data),
        }
        if self.scale is not None:
            meta["scale"] = float(self.scale)
        if self.zero_point is not None:
            meta["zero_point"] = int(self.zero_point)
        if self.meta:
            meta["meta"] = self.meta
        return meta


@dataclass
class FeatureFrame:
    frame_id: int
    pts_ms: float
    source_size: tuple[int, int]
    network_size: tuple[int, int]
    split_layer: int
    tensors: list[TensorBlob]
    meta: dict = field(default_factory=dict)


def bytes_to_bits_msb(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_bytes_msb(bits: Sequence[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit & 1:
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


def write_stream(path: str | Path, frames: Iterable[FeatureFrame], stream_meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        _write_stream_header(f, stream_meta or {})
        for frame in frames:
            _write_frame(f, frame)


def read_stream(path: str | Path) -> tuple[dict, Iterator[FeatureFrame]]:
    path = Path(path)
    f = path.open("rb")
    try:
        meta = _read_stream_header(f)
    except Exception:
        f.close()
        raise

    def iterator() -> Iterator[FeatureFrame]:
        with f:
            while True:
                frame = _read_frame_or_none(f)
                if frame is None:
                    break
                yield frame

    return meta, iterator()


def inspect_stream(path: str | Path, max_frames: int | None = None) -> dict:
    meta, frames = read_stream(path)
    count = 0
    tensor_bytes = 0
    first_frame = None
    for frame in frames:
        if first_frame is None:
            first_frame = {
                "frame_id": frame.frame_id,
                "pts_ms": frame.pts_ms,
                "source_size": frame.source_size,
                "network_size": frame.network_size,
                "split_layer": frame.split_layer,
                "tensors": [
                    {
                        "name": t.name,
                        "shape": list(t.shape),
                        "dtype": t.dtype,
                        "encoding": t.encoding,
                        "nbytes": len(t.data),
                        "meta": t.meta,
                    }
                    for t in frame.tensors
                ],
            }
        tensor_bytes += sum(len(t.data) for t in frame.tensors)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    return {
        "stream": meta,
        "frames_seen": count,
        "tensor_payload_bytes_seen": tensor_bytes,
        "first_frame": first_frame,
    }


def _json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_stream_header(f: BinaryIO, meta: dict) -> None:
    header = {
        "format": "remote-sensing-feature-bitstream",
        "version": VERSION,
        **meta,
    }
    header_bytes = _json_bytes(header)
    f.write(_STREAM_HEADER.pack(STREAM_MAGIC, VERSION, 0, len(header_bytes), zlib.crc32(header_bytes) & 0xFFFFFFFF))
    f.write(header_bytes)


def _read_stream_header(f: BinaryIO) -> dict:
    raw = f.read(_STREAM_HEADER.size)
    if len(raw) != _STREAM_HEADER.size:
        raise BitstreamError("missing RSBF stream header")
    magic, version, _reserved, header_len, header_crc = _STREAM_HEADER.unpack(raw)
    if magic != STREAM_MAGIC:
        raise BitstreamError("not an RSBF stream")
    if version != VERSION:
        raise BitstreamError(f"unsupported RSBF version: {version}")
    header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
        raise BitstreamError("truncated RSBF stream header")
    if (zlib.crc32(header_bytes) & 0xFFFFFFFF) != header_crc:
        raise BitstreamError("RSBF stream header CRC mismatch")
    return json.loads(header_bytes.decode("utf-8"))


def _write_frame(f: BinaryIO, frame: FeatureFrame) -> None:
    payload_parts: list[bytes] = []
    tensor_meta: list[dict] = []
    offset = 0
    for tensor in frame.tensors:
        payload_parts.append(tensor.data)
        tensor_meta.append(tensor.to_meta(offset))
        offset += len(tensor.data)
    payload = b"".join(payload_parts)
    header = {
        "frame_id": int(frame.frame_id),
        "pts_ms": float(frame.pts_ms),
        "source_size": [int(frame.source_size[0]), int(frame.source_size[1])],
        "network_size": [int(frame.network_size[0]), int(frame.network_size[1])],
        "split_layer": int(frame.split_layer),
        "tensors": tensor_meta,
        "meta": frame.meta,
    }
    header_bytes = _json_bytes(header)
    f.write(
        _FRAME_HEADER.pack(
            FRAME_MAGIC,
            int(frame.frame_id),
            int(round(frame.pts_ms * 1000.0)),
            len(header_bytes),
            len(payload),
            zlib.crc32(header_bytes) & 0xFFFFFFFF,
            zlib.crc32(payload) & 0xFFFFFFFF,
        )
    )
    f.write(header_bytes)
    f.write(payload)


def _read_frame_or_none(f: BinaryIO) -> FeatureFrame | None:
    raw = f.read(_FRAME_HEADER.size)
    if not raw:
        return None
    if len(raw) != _FRAME_HEADER.size:
        raise BitstreamError("truncated RSBF frame header")
    magic, frame_id, pts_us, header_len, payload_len, header_crc, payload_crc = _FRAME_HEADER.unpack(raw)
    if magic != FRAME_MAGIC:
        raise BitstreamError("bad RSBF frame magic")
    header_bytes = f.read(header_len)
    payload = f.read(payload_len)
    if len(header_bytes) != header_len or len(payload) != payload_len:
        raise BitstreamError("truncated RSBF frame")
    if (zlib.crc32(header_bytes) & 0xFFFFFFFF) != header_crc:
        raise BitstreamError(f"RSBF frame {frame_id} header CRC mismatch")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != payload_crc:
        raise BitstreamError(f"RSBF frame {frame_id} payload CRC mismatch")
    header = json.loads(header_bytes.decode("utf-8"))
    tensors: list[TensorBlob] = []
    for meta in header["tensors"]:
        start = int(meta["offset"])
        stop = start + int(meta["nbytes"])
        if start < 0 or stop > len(payload):
            raise BitstreamError(f"RSBF frame {frame_id} tensor range is out of bounds")
        tensors.append(
            TensorBlob(
                name=str(meta["name"]),
                shape=[int(v) for v in meta["shape"]],
                dtype=str(meta["dtype"]),
                encoding=str(meta["encoding"]),
                data=payload[start:stop],
                scale=meta.get("scale"),
                zero_point=meta.get("zero_point"),
                meta=dict(meta.get("meta", {})),
            )
        )
    return FeatureFrame(
        frame_id=int(header.get("frame_id", frame_id)),
        pts_ms=float(header.get("pts_ms", pts_us / 1000.0)),
        source_size=(int(header["source_size"][0]), int(header["source_size"][1])),
        network_size=(int(header["network_size"][0]), int(header["network_size"][1])),
        split_layer=int(header["split_layer"]),
        tensors=tensors,
        meta=dict(header.get("meta", {})),
    )
