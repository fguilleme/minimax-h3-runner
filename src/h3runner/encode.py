from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import save_tree
from .config import load_config
from .latents import flatten_av_latent
from .runtime import bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode MiniMax H3 prompt without the ComfyUI server")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    runtime = bootstrap(config)

    with torch.inference_mode():
        clip = runtime.nodes.CLIPLoader().load_clip(config.clip_name, type="krea2", device="default")[0]
        clip = runtime.clipproj_nodes.ClipProjApply().apply(clip, config.projection_name)[0]
        output = runtime.h3_nodes.MiniMaxH3ImageToVideo.execute(
            clip=clip,
            vae=None,
            prompt=config.prompt,
            width=config.width,
            height=config.height,
            length=config.length,
        )
        conditioning, latent = output.args
        save_tree(args.work_dir / "conditioning", conditioning, artifact_type="conditioning")
        save_tree(args.work_dir / "empty-latent", flatten_av_latent(latent), artifact_type="empty-latent")

    report = {
        "phase": "encode",
        "conditioning": str(args.work_dir / "conditioning.safetensors"),
        "latent": str(args.work_dir / "empty-latent.safetensors"),
        "aligned_frames": config.aligned_frames,
    }
    (args.work_dir / "encode-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
