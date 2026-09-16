from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

from .artifacts import load_tree
from .config import load_config
from .runtime import bootstrap


def atempo_factors(speed: float) -> list[float]:
    if speed <= 0:
        raise ValueError("audio speed must be positive")
    factors = []
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    if abs(speed - 1.0) > 1e-12:
        factors.append(speed)
    return factors


def write_wav(path: Path, audio: dict) -> None:
    waveform = audio["waveform"].detach().to("cpu", torch.float32)
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError(f"unexpected audio shape: {tuple(waveform.shape)}")
    pcm = waveform[0].clamp(-1, 1).movedim(0, 1).numpy()
    pcm = (pcm * 32767.0).round().astype("<i2", copy=False)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(pcm.shape[1])
        output.setsampwidth(2)
        output.setframerate(int(audio["sample_rate"]))
        output.writeframes(pcm.tobytes())


def encode_mp4(images: torch.Tensor, audio: dict, fps: float, output_path: Path, native_fps: float = 24.0) -> None:
    images = images.detach().to("cpu", torch.float32)
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"unexpected video shape: {tuple(images.shape)}")
    frames, height, width, _ = images.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        tmp_dir = Path(tmp)
        wav_path = tmp_dir / "audio.wav"
        mp4_path = tmp_dir / "output.mp4"
        write_wav(wav_path, audio)
        audio_filter = ",".join(f"atempo={factor:.12g}" for factor in atempo_factors(fps / native_fps))
        audio_filter = f"{audio_filter},apad" if audio_filter else "apad"
        duration = frames / fps
        command = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-i", str(wav_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-filter:a", audio_filter,
            "-c:a", "aac", "-b:a", "128k",
            "-t", f"{duration:.9f}",
            "-movflags", "+faststart", str(mp4_path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        try:
            for index in range(frames):
                frame = images[index, :, :, :3].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        os.replace(mp4_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode MiniMax H3 AV latent without the ComfyUI server")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    flat, _ = load_tree(args.work_dir / "sampled-latent", expected_type="sampled-latent")
    runtime = bootstrap(config)

    with torch.inference_mode():
        video_vae = runtime.nodes.VAELoader().load_vae(config.video_vae_name)[0]
        video_samples = {"samples": flat["video"]}
        images = runtime.nodes.VAEDecodeTiled().decode(
            video_vae, video_samples, tile_size=512, overlap=64,
            temporal_size=64, temporal_overlap=8,
        )[0]
        del video_vae
        audio_vae = runtime.nodes.VAELoader().load_vae(config.audio_vae_name)[0]
        audio = runtime.audio_nodes.VAEDecodeAudio.execute(
            audio_vae, {"samples": flat["audio"]}
        ).args[0]
        encode_mp4(images, audio, config.output_fps, args.output)

    report = {
        "phase": "decode",
        "output": str(args.output),
        "frames": int(images.shape[0]),
        "fps": config.output_fps,
    }
    (args.work_dir / "decode-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
