from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any


PROTOCOL = "rs-ai-model-update-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.sat_root)
    model = Path(args.model)
    if not model.exists():
        raise FileNotFoundError(model)
    model_dst = root / "models" / args.task_id / args.version / model.name
    model_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, model_dst)
    active = {
        "protocol": PROTOCOL,
        "task_id": args.task_id,
        "version": args.version,
        "scene": args.scene,
        "target_classes": [c.strip() for c in args.classes.split(",") if c.strip()],
        "model_path": str(model_dst),
        "model_sha256": _sha256(model_dst),
        "split_layer": args.split_layer,
        "imgsz": args.imgsz,
        "tensor_codec": args.tensor_codec,
        "resource_scheme": args.resource_scheme,
        "activated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "initial_load",
    }
    _write_json(root / "active_task.json", active)
    print(json.dumps(active, ensure_ascii=False, indent=2))
    return 0


def cmd_make_update(args: argparse.Namespace) -> int:
    model = Path(args.model)
    if not model.exists():
        raise FileNotFoundError(model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL,
        "package_type": "model_update",
        "task_id": args.task_id,
        "version": args.version,
        "scene": args.scene,
        "target_classes": [c.strip() for c in args.classes.split(",") if c.strip()],
        "model_filename": model.name,
        "model_sha256": _sha256(model),
        "model_bytes": model.stat().st_size,
        "split_layer": args.split_layer,
        "imgsz": args.imgsz,
        "tensor_codec": args.tensor_codec,
        "resource_scheme": args.resource_scheme,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": args.notes,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model, f"models/{model.name}")
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"update_package": str(output), "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


def cmd_uplink(args: argparse.Namespace) -> int:
    package = Path(args.package)
    if not package.exists():
        raise FileNotFoundError(package)
    inbox = Path(args.sat_root) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    dst = inbox / package.name
    shutil.copy2(package, dst)
    print(json.dumps({"uplinked_package": str(dst), "bytes": dst.stat().st_size}, ensure_ascii=False, indent=2))
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    package = Path(args.package)
    if not package.exists():
        package = Path(args.sat_root) / "inbox" / args.package
    if not package.exists():
        raise FileNotFoundError(package)
    root = Path(args.sat_root)
    with zipfile.ZipFile(package, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("protocol") != PROTOCOL:
            raise ValueError(f"unsupported update protocol: {manifest.get('protocol')}")
        model_member = f"models/{manifest['model_filename']}"
        payload = zf.read(model_member)
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != manifest["model_sha256"]:
            raise ValueError("model sha256 mismatch; refusing to activate update")
        model_dst = root / "models" / manifest["task_id"] / manifest["version"] / manifest["model_filename"]
        model_dst.parent.mkdir(parents=True, exist_ok=True)
        model_dst.write_bytes(payload)
    active = {
        "protocol": manifest["protocol"],
        "task_id": manifest["task_id"],
        "version": manifest["version"],
        "scene": manifest["scene"],
        "target_classes": manifest["target_classes"],
        "model_path": str(model_dst),
        "model_sha256": manifest["model_sha256"],
        "split_layer": manifest["split_layer"],
        "imgsz": manifest["imgsz"],
        "tensor_codec": manifest["tensor_codec"],
        "resource_scheme": manifest["resource_scheme"],
        "activated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(package),
    }
    _write_json(root / "active_task.json", active)
    print(json.dumps(active, ensure_ascii=False, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.sat_root) / "active_task.json"
    if not path.exists():
        print(json.dumps({"active": None, "message": "no active task"}, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(_read_json(path), ensure_ascii=False, indent=2))
    return 0


def cmd_active_model(args: argparse.Namespace) -> int:
    active = _read_json(Path(args.sat_root) / "active_task.json")
    print(active["model_path"])
    return 0


def cmd_run_command(args: argparse.Namespace) -> int:
    active = _read_json(Path(args.sat_root) / "active_task.json")
    scheme = "roi-layered" if active.get("resource_scheme") == "roi-layered" else "uniform"
    command = [
        f'CONDA_ENV="{args.conda_env}"',
        f'VIDEO_FILE="{args.video}"',
        f'MODEL="{active["model_path"]}"',
        f'SCHEME="{scheme}"',
        f'MAX_FRAMES="{args.max_frames}"',
        f'./scripts/rs_ai_gpu_direct_viso_demo.sh',
    ]
    print(" ".join(command))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulate ground-to-satellite RS-AI task model updates.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_model(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--task-id", required=True)
        sp.add_argument("--version", required=True)
        sp.add_argument("--scene", required=True)
        sp.add_argument("--classes", required=True)
        sp.add_argument("--model", required=True)
        sp.add_argument("--split-layer", type=int, default=22)
        sp.add_argument("--imgsz", type=int, default=640)
        sp.add_argument("--tensor-codec", default="int8", choices=["int8", "float16", "float32"])
        sp.add_argument("--resource-scheme", default="roi-layered", choices=["uniform", "roi-layered"])

    sp = sub.add_parser("init", help="Initialize the satellite-side active task.")
    sp.add_argument("--sat-root", required=True)
    common_model(sp)
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("make-update", help="Create a ground-side model update package.")
    common_model(sp)
    sp.add_argument("--output", required=True)
    sp.add_argument("--notes", default="")
    sp.set_defaults(func=cmd_make_update)

    sp = sub.add_parser("uplink", help="Simulate uplinking a model package into the satellite inbox.")
    sp.add_argument("--package", required=True)
    sp.add_argument("--sat-root", required=True)
    sp.set_defaults(func=cmd_uplink)

    sp = sub.add_parser("apply", help="Validate and activate an uplinked model package.")
    sp.add_argument("--package", required=True)
    sp.add_argument("--sat-root", required=True)
    sp.set_defaults(func=cmd_apply)

    sp = sub.add_parser("status", help="Print the satellite active task.")
    sp.add_argument("--sat-root", required=True)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("active-model", help="Print the active model path only.")
    sp.add_argument("--sat-root", required=True)
    sp.set_defaults(func=cmd_active_model)

    sp = sub.add_parser("run-command", help="Print a GPU-direct command using the active task.")
    sp.add_argument("--sat-root", required=True)
    sp.add_argument("--video", required=True)
    sp.add_argument("--max-frames", type=int, default=120)
    sp.add_argument("--conda-env", default="djscc")
    sp.set_defaults(func=cmd_run_command)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
