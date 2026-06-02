from pathlib import Path
import tempfile

from rs_ai_link.bitstream import (
    FeatureFrame,
    TensorBlob,
    bits_to_bytes_msb,
    bytes_to_bits_msb,
    inspect_stream,
    read_stream,
    write_stream,
)


def test_rsbf_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "features.rsbf"
        frames = [
            FeatureFrame(
                frame_id=7,
                pts_ms=123.5,
                source_size=(720, 1280),
                network_size=(640, 640),
                split_layer=22,
                tensors=[
                    TensorBlob(
                        name="x",
                        shape=[1, 2, 2, 2],
                        dtype="int8",
                        encoding="raw",
                        data=b"abcdefgh",
                        scale=0.125,
                        zero_point=0,
                        meta={"group": "x", "role": "dense"},
                    )
                ],
                meta={"unit": True},
            )
        ]
        write_stream(path, frames, {"model": "unit.pt", "split_layer": 22})
        meta, loaded = read_stream(path)
        loaded_frames = list(loaded)
        assert meta["model"] == "unit.pt"
        assert loaded_frames[0].frame_id == 7
        assert loaded_frames[0].tensors[0].data == b"abcdefgh"
        assert loaded_frames[0].tensors[0].meta["role"] == "dense"
        assert inspect_stream(path)["frames_seen"] == 1


def test_msb_bits_round_trip():
    data = b"\x80\x01\xfe"
    assert bits_to_bytes_msb(bytes_to_bits_msb(data)) == data
