from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import align_frame_count, load_config


def parse_prompt_schedule(prompt: str) -> list[tuple[float, str]] | None:
    """Parse ``seconds:prompt|seconds:prompt`` or return None for plain text."""
    if ":" not in prompt and "|" not in prompt:
        return None
    if "|" not in prompt:
        prefix = prompt.split(":", 1)[0].strip()
        try:
            float(prefix)
        except ValueError:
            return None
    entries = prompt.split("|")
    parsed: list[tuple[float, str]] = []
    for entry in entries:
        duration_text, separator, text = entry.partition(":")
        if not separator:
            raise ValueError("each scheduled prompt must be duration:prompt")
        try:
            duration = float(duration_text.strip())
        except ValueError as exc:
            raise ValueError("scheduled prompt duration must be numeric") from exc
        text = text.strip()
        if duration <= 0:
            raise ValueError("scheduled prompt duration must be positive")
        if not text:
            raise ValueError("scheduled prompt text must not be empty")
        parsed.append((duration, text))
    return parsed


def build_scheduled_chunk_config(
    base: dict, index: int, continuation_frame: str | Path | None,
    duration: float, prompt: str,
) -> dict:
    config = build_chunk_config(base, index, continuation_frame)
    fps = float(base.get("output_fps", 0))
    if fps <= 0:
        raise ValueError("output_fps must be positive for scheduled prompts")
    # Continuation chunks begin with the previous chunk's last frame. Generate
    # one extra frame so deduplication preserves the requested duration.
    config["length"] = max(5, round(duration * fps) + (1 if index else 0))
    config["prompt"] = prompt
    return config


def extract_last_frame(video: str | Path, output: str | Path) -> Path:
    video = Path(video)
    output = Path(output)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nk=1:nw=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frame_count = int(probe.stdout.strip())
    if frame_count <= 0:
        raise ValueError(f"video has no decoded frames: {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video),
            "-vf", f"select=eq(n\\,{frame_count - 1})", "-frames:v", "1", str(output),
        ],
        check=True,
    )
    return output


def build_chunk_config(
    base: dict, index: int, continuation_frame: str | Path | None
) -> dict:
    if index < 0:
        raise ValueError("chunk index must be non-negative")
    config = dict(base)
    config["seed"] = int(base.get("seed", 0)) + index
    config["last_frame"] = None
    if continuation_frame is not None:
        config["first_frame"] = str(Path(continuation_frame).resolve())
    return config


def build_video_concat_filter(
    count: int,
    fps: float,
    target_duration: float,
    segment_durations: list[float] | None = None,
) -> str:
    if count <= 0:
        raise ValueError("segment count must be positive")
    if fps <= 0 or target_duration <= 0:
        raise ValueError("fps and target duration must be positive")
    if segment_durations is not None:
        if len(segment_durations) != count or any(value <= 0 for value in segment_durations):
            raise ValueError("segment durations must be positive and match segment count")
    frame_duration = 1.0 / fps
    filters: list[str] = []
    inputs: list[str] = []
    for index in range(count):
        if index == 0:
            trim = (
                f"trim=duration={segment_durations[index]:.9f},"
                if segment_durations is not None else ""
            )
            filters.append(f"[{index}:v]{trim}setpts=PTS-STARTPTS[v{index}]")
        else:
            duration = (
                f":duration={segment_durations[index]:.9f}"
                if segment_durations is not None else ""
            )
            filters.append(
                f"[{index}:v]trim=start={frame_duration:.9f}{duration},"
                f"setpts=PTS-STARTPTS[v{index}]"
            )
        inputs.append(f"[v{index}]")
    filters.append("".join(inputs) + f"concat=n={count}:v=1:a=0[vc]")
    filters.append(
        f"[vc]trim=duration={target_duration:.9f},setpts=PTS-STARTPTS[vout]"
    )
    return ";".join(filters)


def build_concat_filter(
    count: int,
    fps: float,
    target_duration: float,
    segment_durations: list[float] | None = None,
) -> str:
    if count <= 0:
        raise ValueError("segment count must be positive")
    if fps <= 0 or target_duration <= 0:
        raise ValueError("fps and target duration must be positive")
    if segment_durations is not None:
        if len(segment_durations) != count or any(value <= 0 for value in segment_durations):
            raise ValueError("segment durations must be positive and match segment count")
    frame_duration = 1.0 / fps
    filters: list[str] = []
    concat_inputs: list[str] = []
    for index in range(count):
        if index == 0:
            trim = (
                f"trim=duration={segment_durations[index]:.9f},"
                if segment_durations is not None else ""
            )
            atrim = (
                f"atrim=duration={segment_durations[index]:.9f},"
                if segment_durations is not None else ""
            )
            filters.append(f"[{index}:v]{trim}setpts=PTS-STARTPTS[v{index}]")
            filters.append(f"[{index}:a]{atrim}asetpts=PTS-STARTPTS[a{index}]")
        else:
            duration = (
                f":duration={segment_durations[index]:.9f}"
                if segment_durations is not None else ""
            )
            filters.append(
                f"[{index}:v]trim=start={frame_duration:.9f}{duration},"
                f"setpts=PTS-STARTPTS[v{index}]"
            )
            filters.append(
                f"[{index}:a]atrim=start={frame_duration:.9f}{duration},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        concat_inputs.extend((f"[v{index}]", f"[a{index}]"))
    filters.append(
        "".join(concat_inputs) + f"concat=n={count}:v=1:a=1[vc][ac]"
    )
    filters.append(
        f"[vc]trim=duration={target_duration:.9f},setpts=PTS-STARTPTS[vout]"
    )
    filters.append(
        f"[ac]atrim=duration={target_duration:.9f},asetpts=PTS-STARTPTS[aout]"
    )
    return ";".join(filters)


def segment_count(target_duration: float, segment_duration: float, fps: float) -> int:
    if target_duration <= 0:
        raise ValueError("target duration must be positive")
    if segment_duration <= 0:
        raise ValueError("segment duration must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if target_duration <= segment_duration:
        return 1
    continuation_duration = segment_duration - 1.0 / fps
    if continuation_duration <= 0:
        raise ValueError("segment duration must exceed one output frame")
    return 1 + math.ceil((target_duration - segment_duration) / continuation_duration)


def concat_segments(
    segments: list[Path],
    output: Path,
    fps: float,
    target_duration: float,
    audio_policy: str = "segments",
    native_audio_duration: float | None = None,
    segment_durations: list[float] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if audio_policy not in {"first", "segments"}:
        raise ValueError("audio_policy must be one of: first, segments")

    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        command = ["ffmpeg", "-y", "-v", "error"]
        for segment in segments:
            command.extend(["-i", str(segment)])

        if audio_policy == "first":
            if native_audio_duration is None or native_audio_duration <= 0:
                raise ValueError("native_audio_duration must be positive for first audio policy")
            native_audio = Path(tmp) / "native-audio.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-i", str(segments[0]),
                    "-vn", "-t", f"{native_audio_duration:.9f}",
                    "-c:a", "pcm_s16le", str(native_audio),
                ],
                check=True,
            )
            command.extend(["-stream_loop", "-1", "-i", str(native_audio)])
            graph = build_video_concat_filter(
                len(segments), fps=fps, target_duration=target_duration,
                segment_durations=segment_durations,
            )
            audio_map = f"{len(segments)}:a:0"
        else:
            graph = build_concat_filter(
                len(segments), fps=fps, target_duration=target_duration,
                segment_durations=segment_durations,
            )
            audio_map = "[aout]"

        command.extend(
            [
                "-filter_complex", graph, "-map", "[vout]", "-map", audio_map,
                "-r", f"{fps:.9f}", "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-b:a", "192k", "-ar", "32000", "-t", f"{target_duration:.9f}",
                "-movflags", "+faststart", str(output),
            ]
        )
        subprocess.run(command, check=True)


def segment_signature(
    segments: list[Path], fps: float, target_duration: float,
    audio_policy: str = "segments", segment_durations: list[float] | None = None,
) -> str:
    digest = hashlib.sha256(
        f"{fps:.9f}\0{target_duration:.9f}\0{audio_policy}\0{segment_durations}".encode()
    )
    for segment in segments:
        digest.update(segment.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a longer MiniMax H3 video as chained resumable segments"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--audio-policy", choices=("first", "segments"), default="first",
        help="first keeps one continuous soundtrack; segments uses each H3 segment audio.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    if config.last_frame is not None:
        raise ValueError("long-run mode does not support last_frame")
    schedule = parse_prompt_schedule(config.prompt)
    target_duration = sum(duration for duration, _ in schedule) if schedule else args.duration
    count = len(schedule) if schedule else segment_count(
        target_duration, config.output_duration, config.output_fps
    )
    print(
        f"long run: target={target_duration:.3f}s segments={count} "
        f"segment={config.output_duration:.3f}s"
        + (" prompt_schedule=enabled" if schedule else ""),
        flush=True,
    )

    base = json.loads(config_path.read_text())
    if config.first_frame is not None:
        base["first_frame"] = str(config.first_frame.resolve())
    base["last_frame"] = None
    source_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = source_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    segments: list[Path] = []
    continuation: Path | None = None
    for index in range(count):
        chunk_dir = work_dir / f"chunk-{index:03d}"
        artifact_dir = chunk_dir / "artifacts"
        segment = chunk_dir / "segment.mp4"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        if schedule:
            prompt_duration, prompt_text = schedule[index]
            chunk_data = build_scheduled_chunk_config(
                base, index=index, continuation_frame=continuation,
                duration=prompt_duration, prompt=prompt_text,
            )
        else:
            chunk_data = build_chunk_config(base, index=index, continuation_frame=continuation)
        chunk_config = chunk_dir / "config.json"
        chunk_config.write_text(json.dumps(chunk_data, indent=2, ensure_ascii=False) + "\n")
        command = [
            sys.executable, "-m", "h3runner.run", "--config", str(chunk_config),
            "--work-dir", str(artifact_dir), "--output", str(segment),
        ]
        if args.force:
            command.append("--force")
        print(f"chunk {index + 1}/{count}: {segment}", flush=True)
        subprocess.run(command, env=env, check=True)
        segments.append(segment)

        if index + 1 < count:
            continuation = extract_last_frame(segment, chunk_dir / "last-frame.png")

    scheduled_durations = [duration for duration, _ in schedule] if schedule else None
    signature = segment_signature(
        segments, config.output_fps, target_duration, args.audio_policy,
        scheduled_durations,
    )
    signature_path = work_dir / "concat-signature.txt"
    previous = signature_path.read_text().strip() if signature_path.is_file() else None
    if args.force or previous != signature or not output.is_file():
        print(f"concatenating {count} segments -> {output}", flush=True)
        concat_segments(
            segments,
            output,
            config.output_fps,
            target_duration,
            audio_policy=args.audio_policy,
            native_audio_duration=(
                align_frame_count(max(5, round(schedule[0][0] * config.output_fps)))
                / config.output_fps
                if schedule else config.aligned_frames / 24.0
            ),
            segment_durations=scheduled_durations,
        )
        signature_path.write_text(signature + "\n")
    else:
        print("concatenation: nothing to do", flush=True)

    report = {
        "phase": "longrun",
        "output": str(output),
        "target_duration": target_duration,
        "segments": [str(segment) for segment in segments],
        "segment_count": count,
        "prompt_schedule": schedule,
        "audio_policy": args.audio_policy,
    }
    (work_dir / "longrun-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
