from __future__ import annotations

import json
import os
import shutil
import time
import uuid
import zlib
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MAGIC = b"MUP1"
HEADER_LEN = 256


TASKS = {
    "flood": {
        "task_name": "洪水检测 Sen1Floods11",
        "model_name": "emergency_sen1floods11_flood_best.pt",
        "classes": ["flood_water"],
        "input_size": [640, 640],
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "version": "2026.05.demo",
        "metrics": {"mAP50": None, "summary": "flood_water detection demo"},
        "tx_video": ["tasks", "flood", "videos", "original_satellite_sequence.mp4"],
        "rx_video": ["tasks", "flood", "videos", "receiver_detected_sequence.mp4"],
        "display": {
            "input_title": "原始遥感序列",
            "output_title": "洪水区域检测结果",
            "style": "蓝色/青色标记洪水区域",
        },
    },
    "earthquake": {
        "task_name": "地震建筑损毁 QuickQuake",
        "model_name": "emergency_quickquake_damage_best.pt",
        "classes": ["intact_building", "damaged_building"],
        "input_size": [640, 640],
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "version": "2026.05.demo",
        "metrics": {"mAP50": 0.763, "summary": "intact vs damaged building assessment"},
        "tx_video": ["tasks", "earthquake", "videos", "original_earthquake_building_mosaic.mp4"],
        "rx_video": ["tasks", "earthquake", "videos", "receiver_detected_building_damage_mosaic.mp4"],
        "display": {
            "input_title": "灾后建筑遥感输入",
            "output_title": "建筑损毁检测结果",
            "style": "intact_building=绿色, damaged_building=红/橙色",
        },
    },
    "maritime_ship": {
        "task_name": "海上船只检测 SSDD",
        "model_name": "security_maritime_ship_best.pt",
        "classes": ["ship"],
        "input_size": [640, 640],
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "version": "2026.05.demo",
        "metrics": {"mAP50": 0.991, "summary": "SAR ship detection demo"},
        "tx_video": ["tasks", "maritime_ship", "videos", "original_maritime_ship_mosaic.mp4"],
        "rx_video": ["tasks", "maritime_ship", "videos", "receiver_detected_ship_mosaic.mp4"],
        "display": {
            "input_title": "原始海上 SAR 图像",
            "output_title": "海上船只检测结果",
            "style": "青色/黄色 ship 检测框",
        },
    },
}


def _json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pack_chunk_header(meta: dict) -> bytes:
    raw = _json_bytes(meta)
    if len(raw) > HEADER_LEN - len(MAGIC) - 4:
        raise ValueError(f"chunk header too large: {len(raw)} bytes")
    return MAGIC + len(raw).to_bytes(4, "big") + raw + bytes(HEADER_LEN - len(MAGIC) - 4 - len(raw))


def _unpack_chunk_header(blob: bytes) -> Tuple[dict, bytes]:
    if len(blob) < HEADER_LEN or blob[:4] != MAGIC:
        raise ValueError("invalid ModelUpdate chunk magic")
    n = int.from_bytes(blob[4:8], "big")
    if n <= 0 or n > HEADER_LEN - len(MAGIC) - 4:
        raise ValueError("invalid ModelUpdate chunk header length")
    meta = json.loads(blob[8:8 + n].decode("utf-8"))
    payload = blob[HEADER_LEN:HEADER_LEN + int(meta["payload_len"])]
    return meta, payload


@dataclass
class ModelUpdatePackage:
    update_id: str
    task_id: str
    manifest: dict
    package_file: Path
    manifest_file: Path
    labels_file: Path
    config_file: Path
    chunk_count: int
    model_bytes: int
    sha256: str


class ModelUpdatePacketizer:
    def __init__(self, ai_root: Path, work_root: Path, chunk_size: int = 4096):
        self.ai_root = Path(ai_root)
        self.work_root = Path(work_root)
        self.chunk_size = int(chunk_size)
        self.project_root = self._find_project_root(self.ai_root)
        self.models_root = self._first_existing(
            self.ai_root / "models",
            self.project_root / "data" / "models" / "Ai_process_V3" / "models",
        )
        self.media_root = self._first_existing(
            self.project_root / "data" / "media" / "Ai_process_V3",
            self.ai_root,
        )

    @staticmethod
    def _find_project_root(start_dir: Path) -> Path:
        for path in [start_dir, *start_dir.parents]:
            if (path / "requirements.txt").exists() and (path / "source").exists():
                return path
            if path.name == "organized_workspace_20260601":
                return path
        return start_dir

    @staticmethod
    def _first_existing(*paths: Path) -> Path:
        for path in paths:
            if path.exists():
                return path
        return paths[0]

    def task_info(self, task_id: str) -> dict:
        if task_id not in TASKS:
            raise ValueError(f"unsupported task_id: {task_id}")
        info = dict(TASKS[task_id])
        info["task_id"] = task_id
        info["model_path"] = self.models_root / info["model_name"]
        info["tx_video_path"] = self.media_root.joinpath(*info["tx_video"])
        info["rx_video_path"] = self.media_root.joinpath(*info["rx_video"])
        return info

    def build(self, task_id: str, update_id: str | None = None) -> ModelUpdatePackage:
        info = self.task_info(task_id)
        model_path = Path(info["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        update_id = update_id or f"{task_id}-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        out_dir = self.work_root / update_id
        out_dir.mkdir(parents=True, exist_ok=True)

        model_hash = _sha256_file(model_path)
        manifest = {
            "update_type": "model_update",
            "update_id": update_id,
            "task_id": task_id,
            "task_name": info["task_name"],
            "model_name": info["model_name"],
            "model_format": "pytorch_pt",
            "version": info["version"],
            "classes": info["classes"],
            "input_size": info["input_size"],
            "confidence_threshold": info["confidence_threshold"],
            "nms_threshold": info["nms_threshold"],
            "model_bytes": model_path.stat().st_size,
            "sha256": model_hash,
            "chunk_size": self.chunk_size,
            "chunk_count": 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_model_path": str(model_path),
            "display": info["display"],
            "metrics": info["metrics"],
        }
        labels = {"task_id": task_id, "classes": info["classes"]}
        infer_config = {
            "task_id": task_id,
            "input_size": info["input_size"],
            "confidence_threshold": info["confidence_threshold"],
            "nms_threshold": info["nms_threshold"],
            "model_format": "pytorch_pt",
        }

        files: List[Tuple[str, bytes, int]] = [
            ("labels", _json_bytes(labels), 0),
            ("config", _json_bytes(infer_config), 0),
            ("model", model_path.read_bytes(), 1),
        ]
        manifest_placeholder = _json_bytes(dict(manifest, chunk_count=0))
        files.insert(0, ("manifest", manifest_placeholder, 0))

        chunk_total = 0
        for _, data, _ in files:
            chunk_total += max(1, (len(data) + self.chunk_size - 1) // self.chunk_size)
        manifest["chunk_count"] = chunk_total
        files[0] = ("manifest", _json_bytes(manifest), 0)

        package_file = out_dir / f"{update_id}.mupd"
        chunk_id = 0
        with package_file.open("wb") as f:
            for file_type, data, priority in files:
                count_for_file = max(1, (len(data) + self.chunk_size - 1) // self.chunk_size)
                for file_chunk_id in range(count_for_file):
                    start = file_chunk_id * self.chunk_size
                    payload = data[start:start + self.chunk_size]
                    meta = {
                        "update_id": update_id,
                        "task_id": task_id,
                        "file_type": file_type,
                        "chunk_id": chunk_id,
                        "chunk_count": chunk_total,
                        "file_chunk_id": file_chunk_id,
                        "file_chunk_count": count_for_file,
                        "payload_len": len(payload),
                        "payload_crc32": zlib.crc32(payload) & 0xFFFFFFFF,
                        "priority": priority,
                    }
                    f.write(_pack_chunk_header(meta))
                    f.write(payload)
                    chunk_id += 1

        manifest_file = out_dir / "manifest.json"
        labels_file = out_dir / "labels.json"
        config_file = out_dir / "infer_config.json"
        manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        labels_file.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        config_file.write_text(json.dumps(infer_config, ensure_ascii=False, indent=2), encoding="utf-8")

        return ModelUpdatePackage(
            update_id=update_id,
            task_id=task_id,
            manifest=manifest,
            package_file=package_file,
            manifest_file=manifest_file,
            labels_file=labels_file,
            config_file=config_file,
            chunk_count=chunk_total,
            model_bytes=manifest["model_bytes"],
            sha256=model_hash,
        )


class ModelUpdateReassembler:
    def __init__(self, staging_root: Path):
        self.staging_root = Path(staging_root)

    def reassemble_file(self, package_file: Path) -> dict:
        package_file = Path(package_file)
        chunks: Dict[int, Tuple[dict, bytes]] = {}
        crc_errors = []
        with package_file.open("rb") as f:
            while True:
                header = f.read(HEADER_LEN)
                if not header:
                    break
                meta, payload = _unpack_chunk_header(header + f.read(0))
                payload = f.read(int(meta["payload_len"]))
                if len(payload) != int(meta["payload_len"]):
                    raise ValueError(f"truncated chunk {meta.get('chunk_id')}")
                if (zlib.crc32(payload) & 0xFFFFFFFF) != int(meta["payload_crc32"]):
                    crc_errors.append(int(meta["chunk_id"]))
                chunks[int(meta["chunk_id"])] = (meta, payload)

        if not chunks:
            raise ValueError("empty ModelUpdate package")
        first_meta = chunks[min(chunks.keys())][0]
        update_id = first_meta["update_id"]
        task_id = first_meta["task_id"]
        chunk_count = int(first_meta["chunk_count"])
        missing = [i for i in range(chunk_count) if i not in chunks]
        out_dir = self.staging_root / update_id
        out_dir.mkdir(parents=True, exist_ok=True)

        by_file: Dict[str, List[Tuple[int, bytes]]] = {}
        for _, (meta, payload) in chunks.items():
            by_file.setdefault(meta["file_type"], []).append((int(meta["file_chunk_id"]), payload))

        files_out = {}
        for file_type, parts in by_file.items():
            data = b"".join(payload for _, payload in sorted(parts))
            name = {
                "manifest": "manifest.json",
                "labels": "labels.json",
                "config": "infer_config.json",
                "model": "model.pt",
            }.get(file_type, f"{file_type}.bin")
            path = out_dir / name
            path.write_bytes(data)
            files_out[file_type] = path

        manifest = json.loads(files_out["manifest"].read_text(encoding="utf-8"))
        model_sha = _sha256_file(files_out["model"]) if "model" in files_out else ""
        sha_ok = model_sha == manifest.get("sha256")
        return {
            "update_id": update_id,
            "task_id": task_id,
            "chunk_count": chunk_count,
            "received_chunks": len(chunks),
            "missing_chunks": missing,
            "crc_errors": crc_errors,
            "sha256": model_sha,
            "sha256_ok": sha_ok,
            "manifest": manifest,
            "staging_dir": str(out_dir),
            "model_path": str(files_out.get("model", "")),
            "labels_path": str(files_out.get("labels", "")),
            "config_path": str(files_out.get("config", "")),
        }


class ModelRegistry:
    def __init__(self, registry_path: Path, active_root: Path):
        self.registry_path = Path(registry_path)
        self.active_root = Path(active_root)

    def load(self) -> dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {"active_models": {}, "last_update": {}}

    def save(self, registry: dict) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.registry_path)

    def smoke_test(self, model_path: Path, manifest: dict) -> Tuple[bool, str]:
        if not Path(model_path).exists():
            return False, "model file missing"
        if Path(model_path).stat().st_size != int(manifest.get("model_bytes", -1)):
            return False, "model size mismatch"
        try:
            import torch  # type: ignore
            _ = torch.load(str(model_path), map_location="cpu", weights_only=False)
            return True, "torch load smoke test passed"
        except ImportError:
            return True, "torch unavailable; integrity smoke test passed"
        except Exception as exc:
            return False, f"torch load failed: {exc}"

    def activate(self, reassembly: dict) -> dict:
        manifest = reassembly["manifest"]
        task_id = manifest["task_id"]
        model_path = Path(reassembly["model_path"])
        ok, message = self.smoke_test(model_path, manifest)
        if not reassembly.get("sha256_ok"):
            ok, message = False, "sha256 mismatch"
        registry = self.load()
        previous = registry.get("active_models", {}).get(task_id)
        result = {
            "task_id": task_id,
            "smoke_test_passed": ok,
            "smoke_message": message,
            "rolled_back": False,
        }
        if not ok:
            registry["last_update"] = {
                "task_id": task_id,
                "model_name": manifest.get("model_name"),
                "update_id": manifest.get("update_id"),
                "sha256_verified": bool(reassembly.get("sha256_ok")),
                "smoke_test_passed": False,
                "activated": False,
                "status": "failed",
                "reason": message,
                "rolled_back": bool(previous),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self.save(registry)
            result.update({"activated": False, "active_model": previous})
            return result

        task_active_dir = self.active_root / task_id
        task_active_dir.mkdir(parents=True, exist_ok=True)
        active_model = task_active_dir / manifest["model_name"]
        shutil.copy2(model_path, active_model)
        for key, name in (("labels_path", "labels.json"), ("config_path", "infer_config.json")):
            if reassembly.get(key):
                shutil.copy2(reassembly[key], task_active_dir / name)

        registry.setdefault("active_models", {})[task_id] = {
            "model_path": str(active_model),
            "version": manifest.get("version"),
            "classes": manifest.get("classes", []),
            "sha256": manifest.get("sha256"),
            "status": "active",
            "model_name": manifest.get("model_name"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        registry["last_update"] = {
            "task_id": task_id,
            "model_name": manifest.get("model_name"),
            "update_id": manifest.get("update_id"),
            "sha256_verified": True,
            "smoke_test_passed": True,
            "activated": True,
            "status": "active",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save(registry)
        result.update({"activated": True, "active_model": registry["active_models"][task_id]})
        return result
