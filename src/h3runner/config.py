from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path


DEFAULT_COMFY = Path("/home/francois/comfy/ComfyUI")


def align_frame_count(length: int) -> int:
    if length < 5:
        length = 5
    while length % 17 != 5:
        length += 1
    return length


def temporal_shape(length: int) -> tuple[int, int, int]:
    frame_count = align_frame_count(length)
    latent_t = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    audio_t = round(frame_count / 24 * 40)
    return frame_count, latent_t, audio_t


@dataclass(frozen=True)
class H3Config:
    width: int = 608
    height: int = 352
    length: int = 50
    output_fps: float = 10.0
    audio_mode: str = "loop"
    steps: int = 8
    seed: int = 13092028
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    prompt: str = (
        "Realistic cinematic wildlife shot. A small red fox stands in a misty "
        "forest clearing at dawn, gently turns its head toward the camera, leaves "
        "moving softly in the breeze. Static camera, natural light, detailed fur, "
        "no text, no logo. Audio: quiet forest ambience, light wind through leaves, "
        "one distant bird call."
    )
    comfy_root: Path = DEFAULT_COMFY
    first_frame: Path | None = None
    last_frame: Path | None = None
    unet_name: str = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
    clip_name: str = "qwen3vl_4b_fp8_scaled.safetensors"
    projection_name: str = "mmh3-4b-ClipProj-v3.1.safetensors"
    video_vae_name: str = "minimax_h3_video_vae_fp16.safetensors"
    audio_vae_name: str = "minimax_h3_audio_vae_fp32.safetensors"
    lora_name: str = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

    @property
    def aligned_frames(self) -> int:
        return align_frame_count(self.length)

    @property
    def output_duration(self) -> float:
        return self.aligned_frames / self.output_fps

    def validate(self) -> None:
        if self.width <= 0 or self.width % 32:
            raise ValueError("width must be a positive multiple of 32")
        if self.height <= 0 or self.height % 32:
            raise ValueError("height must be a positive multiple of 32")
        if self.output_fps <= 0:
            raise ValueError("output_fps must be positive")
        if self.audio_mode not in {"loop", "stretch", "pad"}:
            raise ValueError("audio_mode must be one of: loop, stretch, pad")
        if self.steps <= 0:
            raise ValueError("steps must be positive")


def load_config(path: str | Path) -> H3Config:
    path = Path(path)
    data = json.loads(path.read_text())
    allowed = {field.name for field in fields(H3Config)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    if "comfy_root" in data:
        data["comfy_root"] = Path(data["comfy_root"])
    for key in ("first_frame", "last_frame"):
        if data.get(key) is not None:
            value = Path(data[key])
            data[key] = value if value.is_absolute() else path.parent / value
    config = H3Config(**data)
    config.validate()
    return config
