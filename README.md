# MiniMax H3 Headless Runner

<!-- Versions -->
[English README](README.md) — [README français](README_FR.md)

Runner local MiniMax H3 split into three independent processes, without the ComfyUI HTTP server, Manager, web interface, or UI graph execution.

It reuses selected ComfyUI Python loaders and kernels because the W4A8 DiT and ClipProj models use their optimized formats. Model weights are neither copied nor downloaded by this project — you must fetch them first (step 2: **Roadmap**) .

## Pipeline

1. `h3runner.encode` loads Qwen3-VL-4B and ClipProj, optionally encodes a first and/or last keyframe with the video VAE, writes the conditioning and empty audiovisual latent, then exits.
2. `h3runner.denoise` loads only the W4A8 DiT and Turbo LoRA, performs the eight sampling steps, writes the generated audiovisual latent, then exits.
3. `h3runner.decode` loads the VAEs, performs temporally tiled decoding, and encodes the final MP4 with `ffmpeg`.

Each process boundary guarantees that the previous model is released before the next phase starts.

## Roadmap

From `git clone` to generating your first video.

1. **Prerequisites.** Linux + Python ≥ 3.10, `ffmpeg`, and `git`. Any ROCm AMD GPU in the gfx11xx family works (Radeon 8060 / 8070S = gfx1151 validated); ROCm ≥ 7.2; PyTorch for ROCm ≥ 2.6.

2. **Clone.**

```bash
git clone <URL-TO-this-repo> minmax-h3-runner
cd minmax-h3-runner
chmod +x run.sh run-long.sh
```

3. **Point to ComfyUI and download models.** Set `config.comfy_root` to your ComfyUI install (default `/home/francois/comfy/ComfyUI`). Place these in that root (`unet/`, `vae/`, `text_encoders/`, `lora/`, `custom_nodes/` following standard ComfyUI conventions). See the [Reused Models](#reused-models) table above for the exact names.

| Name | File | Location |
|---|---|---|
| UNet + video VAE (W4A8, mixed weights) | `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | `models/unet/` (or ComfyUI model path) |
| Text encoder | `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/` |
| Long clip projection (ClipProj) | `mmh3-4b-ClipProj-v3.1.safetensors` | `custom_nodes/ComfyUI-ClipProj/models/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |
| LoRA (8-step) | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `models/lora/` |

4. **Clone & setup the ClipProj custom node** (required by the text-encoding phase):

```bash
cd "$(python -c 'import toml,os;print(toml.load(config.json)["comfy_root"])')"
git clone <URL-TO-ClipProj-repo> custom_nodes/ComfyUI-ClipProj
# install the node's pip requirements into the same Python env as this runner
```

5. **Validate (smoke test) from inside the repo, using its real venv:**

```bash
cd /home/francois/projects/minimax-h3-runner
PYTHONPATH=src /home/francois/comfy/ComfyUI/.venv/bin/python -m unittest discover -s tests -v
```

If the suite passes, proceed to an actual generation:

```bash
cd /home/francois/projects/minimax-h3-runner
PYTHONPATH=src /home/francois/comfy/ComfyUI/.venv/bin/python -m run --config config.json --work-dir runs/fox-56f --output output/minimax-h3-fd560.mp4
```

## Validated Environment

- Radeon 8060S `gfx1151`, 32 GiB reserved VRAM and 32 GiB system RAM
- System ROCm 7.2.4 under `/opt/rocm`
- PyTorch `2.11.0+rocm7.13.0`, HIP runtime `7.13.99004`
- `comfy-kitchen 0.2.34`
- Python: `/home/francois/comfy/ComfyUI/.venv/bin/python`
- Docker remained active throughout the smoke tests
- `comfyui.service` remained stopped

## Reused Models

Model paths are resolved under `/home/francois/comfy/ComfyUI/models`:

- `diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`
- `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
- `clip_projections/mmh3-4b-ClipProj-v3.1.safetensors`
- `vae/minimax_h3_video_vae_fp16.safetensors`
- `vae/minimax_h3_audio_vae_fp32.safetensors`
- `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

## HTTP server and curl examples

The optional HTTP server queues jobs and returns immediately. It does not
launch the ComfyUI application; it launches `h3runner.longrun` in the same
ComfyUI virtual environment.

Start it from the project directory:

```bash
PYTHONPATH=src /home/francois/comfy/ComfyUI/.venv/bin/python \
  -m h3runner.server \
  --host 127.0.0.1 --port 8988 \
  --comfy-root /home/francois/comfy/ComfyUI \
  --runs-dir runs --log-dir log --pid runs/server.pid
```

Check readiness:

```bash
curl -s http://127.0.0.1:8988/healthz
```

### Example 1 — prompt only, 5 seconds at 10 fps

```bash
curl -sS -X POST http://127.0.0.1:8988/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A red fox walking through a sunlit forest clearing",
    "duration": 5,
    "fps": 10,
    "width": 384,
    "height": 224
  }'
```

### Example 2 — image plus prompt, 5 seconds at 10 fps

`first_frame` may be an absolute path or a path relative to the directory
from which the server was started.

```bash
curl -sS -X POST http://127.0.0.1:8988/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "The fox slowly turns its head while leaves move in the breeze",
    "first_frame": "input/fox-first.png",
    "duration": 5,
    "fps": 10,
    "width": 384,
    "height": 224
  }'
```

### Example 3 — long video

The server chains H3 segments until the requested duration. This example
uses one soundtrack policy and an initial image:

```bash
curl -sS -X POST http://127.0.0.1:8988/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A dancer performs a graceful energetic routine in a studio",
    "first_frame": "input/dancer-source.jpg",
    "duration": 30,
    "fps": 10,
    "width": 256,
    "height": 384,
    "audio": "first"
  }'
```

Each response contains a job id. Poll it until `state` is `done` or
`failed`:

```bash
curl -sS 'http://127.0.0.1:8988/status?job=JOB_ID'
```

The response's `output` field gives the generated MP4 path and `log_path`
gives the child-process log. Cancel a running job with:

```bash
curl -sS -X DELETE 'http://127.0.0.1:8988/status?job=JOB_ID'
```

## AI video post-processing: Real-ESRGAN and RIFE

For arbitrary input videos, use [Video2X](https://github.com/k4yt3x/video2x),
which provides Vulkan backends for [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
(super-resolution) and [RIFE](https://github.com/hzwer/ECCV2022-RIFE)
(frame interpolation). This is separate from MiniMax H3 generation. The
Linux AppImage was validated on an AMD Radeon 8060S with RADV/Vulkan.

Example: upscale a portrait video 2x, then interpolate 10 fps to 30 fps:

```bash
mkdir -p tools/video2x
curl -fL -o tools/video2x/Video2X-x86_64.AppImage \\
  https://github.com/k4yt3x/video2x/releases/download/6.4.0/Video2X-x86_64.AppImage
chmod +x tools/video2x/Video2X-x86_64.AppImage

# 2x super-resolution
./tools/video2x/Video2X-x86_64.AppImage --no-progress \\
  -i runs/input.mp4 -o runs/input-realesrgan-2x.mp4 \\
  -p realesrgan -s 2 --realesrgan-model realesr-animevideov3

# RIFE x3: 10 fps -> 30 fps
./tools/video2x/Video2X-x86_64.AppImage --no-progress \\
  -i runs/input-realesrgan-2x.mp4 -o runs/input-realesrgan-2x-rife-30fps.mp4 \\
  -p rife -m 3 --rife-model rife-v4.6
```

Use dimensions that are multiples of 32 for H3 generation. Real-ESRGAN
reconstructs plausible detail; it cannot recover detail absent from the source.
RIFE synthesizes intermediate frames and can produce artifacts around fast motion.

## Usage

```bash
cd /home/francois/projects/minimax-h3-runner
./run.sh
```

To explicitly stop the ComfyUI user service before execution:

```bash
./run.sh --stop-comfy-service
```

The service is intentionally not restarted. Docker containers are never stopped.

To recompute every phase even when valid artifacts already exist:

```bash
./run.sh --force
```

### Image-to-Video

Use a first frame:

```bash
./run.sh \
  --first-frame /path/to/start.png \
  --work-dir runs/my-i2v \
  --output output/my-i2v.mp4
```

Use both first and last frames:

```bash
./run.sh \
  --first-frame /path/to/start.png \
  --last-frame /path/to/end.png \
  --work-dir runs/my-bounded-i2v \
  --output output/my-bounded-i2v.mp4
```

RGB, RGBA, and grayscale images are normalized to floating-point RGB. MiniMax resizes them to the configured resolution. The first image is anchored to the first frame and the last image to the final frame. The same inputs can be set in `config.json` through `first_frame` and `last_frame`.

Choose width and height as multiples of 32. To preserve the source aspect ratio, select the nearest compatible dimensions. For example, an 853×1280 portrait is nearly 2:3 and is rendered without visible distortion at 384×576.

### Long Videos with Chained Segments

Long-video mode generates multiple H3 segments, extracts the exact final decoded frame of each segment, and uses it as the first frame of the next segment. The seed is incremented for every continuation. During assembly, one output-frame duration is removed from the beginning of each continuation to avoid duplicating the boundary keyframe.

By default, `--audio-policy first` keeps only the native soundtrack from the first segment and loops it across the complete video. This prevents the music from changing at every segment boundary. Video and audio are then trimmed to the exact requested duration.

```bash
./run-long.sh \
  --config config.json \
  --duration 30 \
  --work-dir runs/my-long-video-30s \
  --output output/my-long-video-30s.mp4
```

Each segment has its own resumable directory:

```text
runs/dancer-30s/
  chunk-000/{config.json,segment.mp4,artifacts/}
  chunk-001/{config.json,segment.mp4,artifacts/}
  ...
```

A repeated invocation skips valid phases and completed segments. Final concatenation is also skipped when its content signature has not changed. Use `--force` to regenerate every segment.

Long-video mode accepts an initial `first_frame`, but not `last_frame`, because every segment ending is reserved for automatic continuity. To restore independently generated H3 audio for each segment, explicitly pass:

```bash
--audio-policy segments
```

## Resuming Runs

Intermediate files are stored under the selected run directory, for example `runs/fox-56f/`:

- `conditioning.{json,safetensors}`
- `empty-latent.{json,safetensors}`
- `sampled-latent.{json,safetensors}`

`run.sh` resumes from the first missing phase. A content signature covers the configuration file and keyframe contents. Changing an image, prompt, or generation setting automatically invalidates the relevant run and executes the phases again.

Tensors are stored with Safetensors. A JSON manifest preserves lists, tuples, dictionaries, and scalar values without using pickle.

## Frame Rate and Audio

H3 generates at a native 24 FPS and aligns frame counts to `17k + 5`. A request for 50 frames therefore produces 56 frames. At an output rate of 10 FPS, the resulting segment lasts 5.6 seconds.

The default `audio_mode: "loop"` preserves the native H3 audio tempo and repeats it until the video duration is reached. Two alternatives remain available in the configuration:

- `"stretch"`: reproduces the previous temporal stretching behavior with FFmpeg `atempo`.
- `"pad"`: preserves native tempo and fills the remaining duration with silence.

For long videos, `--audio-policy first` additionally prevents a new soundtrack from being introduced by each generated segment.

## Smoke-Test Results

Single-segment test:

- Conditioning: `(1, 67, 5120)` FP32
- Video latent: `(1, 24, 17, 22, 38)` FP32
- Audio latent: `(1, 32, 2, 93)` FP32
- Diffusion: 8/8 steps in approximately 55–68 seconds
- Output: 56 frames, H.264 video
- Audio: stereo AAC at 32 kHz

Validated long-video test:

- Two chained 5.6-second H3 segments
- Final output: exactly 8.000 seconds
- 80 frames at 10 FPS
- 384×576 portrait resolution
- Video and audio both exactly 8.000 seconds
- One continuous first-segment soundtrack with measured loop correlation of `0.9993`
- Successful no-op resume for both segments and final concatenation

## Tests

```bash
PYTHONPATH=src /home/francois/comfy/ComfyUI/.venv/bin/python \
  -m unittest discover -s tests -v
```

## Limitations

- The runner imports internal ComfyUI Python modules but does not launch the ComfyUI application or HTTP server.
- A ComfyUI update may change internal APIs. Run the complete test suite and a smoke test after updating ComfyUI.
- Attention still covers the complete audiovisual sequence during diffusion. Temporal tiling is used for VAE decoding, not for the DiT.
- Long-video visual continuity is keyframe-based. Motion can still drift between independently sampled segments.
- `--audio-policy first` gives a consistent but repeated soundtrack. It does not generate a truly continuous long-form musical composition.
- The CUDA `flash-attn` package is not used. PyTorch reports that experimental AMD AOTriton attention can be enabled, but it remains disabled until validated on `gfx1151`.
