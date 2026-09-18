"""HTTP interface so a video can be generated from one curl POST.

Stdlib-only; runs in the ComfyUI venv without extra installs. Each POST to
``/generate`` writes a config under ``runs/<job-id>/config.json``, spawns the
existing ``h3runner.longrun`` entrypoint as a background process, records the
PID in the pid-file, then ``GET /status?job=<id>`` is polled down to the final
MP4 at ``runs/<output-id>.mp4``.

Usage (no systemd required):

    PYTHONPATH=src /home/francois/comfy/ComfyUI/.venv/bin/python \
        -m h3runner.server --host 127.0.0.1 --port 8988 \
        --comfy-root /home/francois/comfy/ComfyUI \
        --runs-dir runs --log-dir log --pid runs/server.pid

Curl example with real defaults:

    curl -s "http://localhost:8988/generate" \\
        -H 'content-type: application/json' \\
        -d '{"prompt":"A fox in a forest", "duration":5, "fps":10}'
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

DEFAULT_FRAMES = 50
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 10.0
DEFAULT_DURATION_SECONDS = 6.0
DEFAULT_STEPS = 8
DEFAULT_SEED = 0

DEFAp_PROMPT = (
    "A stylish young woman with long dark wavy hair performs a graceful "
    "dance in soft studio light, plain grey background, no text and no logo, "
    "upbeat rhythmic electronic dance music."
)

AUDIO_POLICIES = ("first", "segments")

SEGMENT_MODELS = {
    "unet_name": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
    "projection_name": "mmh3-4b-ClipProj-v3.1.safetensors",
    "video_vae_name": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae_name": "minimax_h3_audio_vae_fp32.safetensors",
    "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
}


@dataclass
class Job:
    id: str
    state: str
    work_dir: Path
    output: Path
    started: float = field(default_factory=time.monotonic)
    pid: int | None = None
    log_path: Path | None = None
    finished: float | None = None
    error: str | None = None
    result: dict | None = None

    @property
    def done(self) -> bool:
        return self.state in ("done", "failed")

    def to_dict(self) -> dict:
        data = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        data.update({"work_dir": str(self.work_dir), "output": str(self.output)})
        return data

    def refresh(self) -> None:
        """Re-read child state from its work-dir after it exits."""
        if self.done:
            return
        alive = bool(self.pid and _alive(self.pid))
        if not alive:
            report = self.work_dir / "longrun-result.json"
            error_file = self.work_dir / "log.txt"
            if report.is_file():
                try:
                    self.result = json.loads(report.read_text())
                except (OSError, ValueError):
                    pass
                self.state, self.finished = "done", time.monotonic()
            else:
                self.state, self.finished = "failed", time.monotonic()
                self.error = _error_text(error_file) or "process exited without a result"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # ``kill(pid, 0)`` succeeds for a zombie until its parent reaps it.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (OSError, IndexError):
        return True
    return state != "Z"


def _fresh_pid_dir(base: Path, existing: set[str]) -> Path:
    for _ in range(16):
        candidate = base / f"job-{uuid.uuid4().hex[:8]}"
        if candidate.name not in existing:
            return candidate
    return base / f"job-{uuid.uuid4().hex[:8]}"


class Manager:
    def __init__(self, comfy_root: Path, runs_root: Path, pid_file: Path, log_dir: Path):
        self.comfy_root = Path(comfy_root)
        self.runs_root = Path(runs_root)
        self.pid_file = Path(pid_file)
        self.log_dir = Path(log_dir)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _python(self) -> Path:
        python = self.comfy_root / ".venv" / "bin" / "python"
        return python if python.is_file() else Path("python3")

    def _source(self) -> Path:
        src = self.runs_root.parent / "src"
        return src if src.is_dir() else Path.cwd() / "src"

    def submit(self, params: dict[str, object]) -> Job:
        first_frame = params.get("firstframe")
        if first_frame is None:
            first_frame = params.get("first_frame")
        prompt = str(params.get("prompt") or DEFAp_PROMPT)
        width = int(float(params.get("width", params.get("w") or DEFAULT_WIDTH)))
        height = int(float(params.get("height", params.get("h") or DEFAULT_HEIGHT)))
        fps = float(params.get("fps", params.get("framerate") or DEFAULT_FPS))
        duration = float(params.get("duration", params.get("secs") or DEFAULT_DURATION_SECONDS))
        steps = int(float(params.get("steps", DEFAULT_STEPS)))
        seed = int(float(params.get("seed", DEFAULT_SEED)))
        audio_policy = str(params.get("audio", "first"))
        if audio_policy not in AUDIO_POLICIES:
            raise ValueError(f"audio must be one of: {AUDIO_POLICIES}")

        work_dir = _fresh_pid_dir(self.runs_root, {c.name for c in self.runs_root.iterdir() if c.is_dir()})
        work_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:8]
        output = self.runs_root / f"{job_id}.mp4"

        cfg: dict[str, object] = {
            "width": width,
            "height": height,
            "length": DEFAULT_FRAMES,
            "output_fps": fps,
            "audio_mode": "loop",
            "steps": steps,
            "seed": seed,
            "sampler": str(params.get("sampler", "res_multistep")),
            "scheduler": str(params.get("scheduler", "simple")),
            "prompt": prompt,
            "comfy_root": str(self.comfy_root),
        }
        cfg.update(SEGMENT_MODELS)
        if first_frame:
            _ff = Path(first_frame).expanduser()
            if not _ff.is_absolute():
                _ff = (Path.cwd() / _ff)
            cfg["first_frame"] = str(_ff)
        cfg["last_frame"] = None

        config_path = work_dir / "config.json"
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

        python_path = str(self._source())
        env = dict(os.environ)
        env["PYTHONPATH"] = python_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        log_path = work_dir / "log.txt"
        log_file = open(log_path, "wb", buffering=0)
        proc = subprocess.Popen(
            [
                str(self._python()), "-m", "h3runner.longrun",
                "--config", str(config_path),
                "--work-dir", str(work_dir),
                "--output", str(output),
                "--duration", str(duration),
                "--audio-policy", audio_policy,
            ],
            stdout=log_file, stderr=subprocess.STDOUT, env=env, close_fds=True,
        )

        job = Job(
            id=job_id, state="running", work_dir=work_dir, output=output,
            log_path=log_path, pid=proc.pid,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            job.refresh()
        return job

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._jobs[i].to_dict() for i in sorted(self._jobs)]


class Handler(BaseHTTPRequestHandler):
    manager: Manager | None = None

    # ---- helpers ----------
    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def _send_file(self, path: Path, content_type: str, filename: str) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            self._send(404, {"error": "video file not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        if self.command != "HEAD":
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)

    def _job_payload(self, job: Job) -> dict:
        payload = job.to_dict()
        payload["video_url"] = f"/video?job={job.id}"
        return payload

    # ---- endpoints --------
    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "endpoint not found"}); return
        try:
            raw = self._read_body()
            params = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            self._send(400, {"error": "invalid JSON body"}); return
        if not isinstance(params, dict):
            self._send(400, {"error": "body must be a JSON object"}); return
        try:
            job = self.manager.submit(params)
        except (OSError, ValueError) as exc:
            self._send(400, {"error": str(exc)}); return
        self._send(201, {"job": self._job_payload(job), "message": "job queued"})

    def do_GET(self):
        query = parse_qs(self.path.split("?", 1)[-1]) if "?" in self.path else {}
        if self.path.split("?", 1)[0] == "/healthz":
            self._send(200, {"ok": True}); return
        if self.path.split("?", 1)[0] == "/video":
            job_id = query.get("job", [None])[0]
            if not job_id:
                self._send(400, {"error": "query param 'job' is required"})
                return
            job = self.manager.get(job_id)
            if not job:
                self._send(404, {"error": f"unknown job {job_id!r}"})
                return
            if job.state != "done":
                self._send(409, {"error": "video is not ready", "state": job.state})
                return
            self._send_file(job.output, "video/mp4", f"minimax-h3-{job.id}.mp4")
            return
        job_id = query.get("job", [None])[0]
        if not job_id:
            self._send(400, {"error": "query param 'job' is required"})
            return
        job = self.manager.get(job_id)
        if not job:
            self._send(404, {"error": f"unknown job {job_id!r}"})
            return
        self._send(200, {"job": self._job_payload(job)})

    def do_DELETE(self):
        query = parse_qs(self.path.split("?", 1)[-1]) if "?" in self.path else {}
        job_id = query.get("job", [None])[0]
        if not job_id:
            self._send(400, {"error": "query param 'job' is required"})
            return
        with self.manager._lock:
            job = self.manager._jobs.pop(job_id, None)
        if job and job.state == "running" and job.pid:
            try:
                os.kill(job.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        self._send(200, {"job": job.to_dict() if job else None, "message": "cancelled" if job else "not found"})

    def log_message(self, fmt, *args):
        import sys
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def _error_text(path: Path | None) -> str | None:
    if not path or not path.is_file():
        return None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    return text.strip()[-512:]


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP launcher for the MiniMax H3 runner")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8988)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default="runs")
    parser.add_argument("--log-dir", type=Path, default="log")
    parser.add_argument("--pid", type=str, default=None)
    args = parser.parse_args()

    pid_file = Path(args.pid) if args.pid else Path(None).parent / (args.runs_dir / "server.pid".rstrip("/"))
    manager = Manager(
        comfy_root=args.comfy_root,
        runs_root=args.runs_dir,
        pid_file=pid_file,
        log_dir=args.log_dir,
    )
    Handler.manager = manager
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")
    print(f"minimax-h3 HTTP server on {args.host}:{args.port}", flush=True)
    print("POST /generate  GET /status?job=ID  GET /video?job=ID  DELETE /status?job=ID  GET /healthz", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
