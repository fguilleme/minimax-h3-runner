from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .artifacts import save_tree
from .config import load_config
from .images import load_image_tensor
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

    def progress(message: str, started: float) -> None:
        print(f"encode: {message} elapsed={time.monotonic() - started:.1f}s", flush=True)

    with torch.inference_mode():
        started = time.monotonic()
        first_frame = load_image_tensor(config.first_frame) if config.first_frame is not None else None
        last_frame = load_image_tensor(config.last_frame) if config.last_frame is not None else None
        progress("images loaded", started)

        clip = runtime.nodes.CLIPLoader().load_clip(config.clip_name, type="krea2", device="default")[0]
        progress("CLIP loaded", started)
        clip = runtime.clipproj_nodes.ClipProjApply().apply(clip, config.projection_name)[0]
        progress("projection applied", started)

        latent, frame_count = runtime.h3_nodes._empty_av_latent(config.width, config.height, config.length)
        images = []
        keyframes = []
        if first_frame is not None:
            image = runtime.h3_nodes._resize(first_frame[:1], config.width, config.height, "disabled")
            images.append(image)
            keyframes.append({"resolved_frame_index": 0, "image": image})
        if last_frame is not None:
            image = runtime.h3_nodes._resize(last_frame[:1], config.width, config.height, "center")
            images.append(image)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": image})

        tokens = clip.tokenize(config.prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        progress("multimodal conditioning encoded", started)

        del clip
        runtime.clipproj_nodes.release_all()
        import comfy.model_management as model_management
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        gc.collect()
        progress("text encoder released", started)

        if keyframes:
            import node_helpers
            video_vae = runtime.nodes.VAELoader().load_vae(config.video_vae_name)[0]
            progress("video VAE loaded", started)
            for keyframe in keyframes:
                keyframe["latent"] = video_vae.encode(keyframe.pop("image"))
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"minimax_keyframes": keyframes}
            )
            progress("keyframes encoded", started)

        save_tree(args.work_dir / "conditioning", conditioning, artifact_type="conditioning")
        save_tree(args.work_dir / "empty-latent", flatten_av_latent(latent), artifact_type="empty-latent")
        progress("artifacts saved", started)

    report = {
        "phase": "encode",
        "conditioning": str(args.work_dir / "conditioning.safetensors"),
        "latent": str(args.work_dir / "empty-latent.safetensors"),
        "aligned_frames": config.aligned_frames,
        "mode": "first-last-frame" if first_frame is not None and last_frame is not None else "image-to-video" if first_frame is not None or last_frame is not None else "text-to-video",
    }
    (args.work_dir / "encode-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
