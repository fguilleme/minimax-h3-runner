from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import load_tree, save_tree
from .config import load_config
from .latents import flatten_av_latent, restore_av_latent
from .runtime import bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise MiniMax H3 latent without the ComfyUI server")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    conditioning, _ = load_tree(args.work_dir / "conditioning", expected_type="conditioning")
    flat_latent, _ = load_tree(args.work_dir / "empty-latent", expected_type="empty-latent")
    runtime = bootstrap(config)
    latent = restore_av_latent(flat_latent, runtime.nested_tensor.NestedTensor)

    with torch.inference_mode():
        model = runtime.nodes.UNETLoader().load_unet(config.unet_name, "default")[0]
        model = runtime.nodes.LoraLoaderModelOnly().load_lora_model_only(model, config.lora_name, 1.0)[0]
        guider = runtime.sampler_nodes.BasicGuider.execute(model, conditioning).args[0]
        noise = runtime.sampler_nodes.RandomNoise.execute(config.seed).args[0]
        sampler = runtime.sampler_nodes.KSamplerSelect.execute(config.sampler).args[0]
        sigmas = runtime.sampler_nodes.BasicScheduler.execute(
            model, config.scheduler, config.steps, 1.0
        ).args[0]
        sampled = runtime.sampler_nodes.SamplerCustomAdvanced.execute(
            noise, guider, sampler, sigmas, latent
        ).args[0]
        save_tree(args.work_dir / "sampled-latent", flatten_av_latent(sampled), artifact_type="sampled-latent")

    report = {
        "phase": "denoise",
        "latent": str(args.work_dir / "sampled-latent.safetensors"),
        "steps": config.steps,
        "seed": config.seed,
    }
    (args.work_dir / "denoise-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
