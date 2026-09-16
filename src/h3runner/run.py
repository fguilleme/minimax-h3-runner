from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PHASES = ("encode", "denoise", "decode")


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
    parser.add_argument(
        "--stop-comfy-service",
        action="store_true",
        help="Stop comfyui.service before running; it is intentionally not restarted.",
    )
    args = parser.parse_args()

    config = args.config.resolve()
    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.stop_comfy_service:
        subprocess.run(["systemctl", "--user", "stop", "comfyui.service"], check=True)

    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    existing = artifact_names(work_dir, output)
    plan = phase_plan(existing, args.force)
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


if __name__ == "__main__":
    main()
