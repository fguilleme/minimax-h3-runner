from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .images import aspect_preserving_canvas, image_dimensions


PHASES = ("encode", "denoise", "decode")


def input_signature(config_path: str | Path) -> str:
    config_path = Path(config_path)
    raw = config_path.read_bytes()
    data = json.loads(raw)
    digest = hashlib.sha256(raw)
    for key in ("first_frame", "last_frame"):
        if data.get(key) is None:
            continue
        path = Path(data[key])
        if not path.is_absolute():
            path = config_path.parent / path
        digest.update(key.encode())
        digest.update(path.resolve().as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def aspect_adjusted_config(data: dict, config_path: Path) -> dict:
    """Fit the keyframe ratio inside the requested bounded canvas.

    The requested dimensions are upper bounds. A portrait keyframe must not
    silently turn a modest canvas into a 768x1344 latent and OOM the job.
    """
    source = data.get("first_frame") or data.get("last_frame")
    if source is None:
        return data
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = config_path.parent / source_path
    source_width, source_height = image_dimensions(source_path)
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))
    if width <= 0 or height <= 0:
        return data
    width, height = aspect_preserving_canvas(source_width, source_height)
    adjusted = dict(data)
    adjusted["width"] = width
    adjusted["height"] = height
    return adjusted


def phase_plan(existing: set[str], force: bool) -> list[str]:
    if force:
        return list(PHASES)
    if not {"conditioning", "empty-latent"}.issubset(existing):
        return list(PHASES)
    if "sampled-latent" not in existing:
        return ["denoise", "decode"]
    if "output" not in existing:
        return ["decode"]
    return []


def artifact_names(work_dir: Path, output: Path) -> set[str]:
    names = set()
    for name in ("conditioning", "empty-latent", "sampled-latent"):
        if (work_dir / f"{name}.json").is_file() and (work_dir / f"{name}.safetensors").is_file():
            names.add(name)
    if output.is_file():
        names.add("output")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMax H3 in isolated headless phases")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument("--last-frame", type=Path)
    parser.add_argument(
        "--stop-comfy-service",
        action="store_true",
        help="Stop comfyui.service before running; it is intentionally not restarted.",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = args.config.resolve()
    data = json.loads(config.read_text())
    if args.first_frame is not None:
        data["first_frame"] = str(args.first_frame.resolve())
    if args.last_frame is not None:
        data["last_frame"] = str(args.last_frame.resolve())
    data = aspect_adjusted_config(data, config)
    if data != json.loads(config.read_text()):
        config = work_dir / "effective-config.json"
        config.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    if args.stop_comfy_service:
        subprocess.run(["systemctl", "--user", "stop", "comfyui.service"], check=True)

    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    existing = artifact_names(work_dir, output)
    signature = input_signature(config)
    signature_path = work_dir / "input-signature.txt"
    previous_signature = signature_path.read_text().strip() if signature_path.is_file() else None
    plan = phase_plan(existing, args.force or previous_signature != signature)
    print("phase plan:", ", ".join(plan) or "nothing to do", flush=True)
    for phase in plan:
        command = [
            sys.executable,
            "-m",
            f"h3runner.{phase}",
            "--config",
            str(config),
            "--work-dir",
            str(work_dir),
        ]
        if phase == "decode":
            command.extend(["--output", str(output)])
        print("running:", phase, flush=True)
        subprocess.run(command, env=env, check=True)
    if plan:
        signature_path.write_text(signature + "\n")


if __name__ == "__main__":
    main()
