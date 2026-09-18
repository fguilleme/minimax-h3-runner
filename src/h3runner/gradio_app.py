"""Gradio UI for the headless MiniMax H3 runner.

The UI uses Manager directly; the H3 HTTP server is not required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .server import AUDIO_POLICIES, Manager


def format_job_status(job) -> str:
    if job is None:
        return "Aucun job en cours."
    payload = job.to_dict()
    payload["video_url"] = f"/video?job={job.id}"
    return json.dumps(payload, ensure_ascii=False, indent=2)


def submit_from_ui(manager: Manager, first_frame, prompt: str, duration, fps, steps, seed, audio):
    if not prompt or not prompt.strip():
        raise ValueError("Le prompt est obligatoire")
    params = {
        "prompt": prompt.strip(),
        "duration": float(duration),
        "fps": float(fps),
        "steps": int(steps),
        "seed": int(seed),
        "audio": audio,
    }
    if first_frame:
        params["first_frame"] = str(first_frame)
    job = manager.submit(params)
    return job.id, format_job_status(job), None


def poll_from_ui(manager: Manager, job_id: str | None):
    if not job_id:
        return "Aucun job en cours.", None
    job = manager.get(job_id)
    if job is None:
        return f"Job introuvable : {job_id}", None
    if job.state == "done" and job.output.is_file():
        return format_job_status(job), str(job.output)
    if job.state == "failed":
        return format_job_status(job), None
    return format_job_status(job), None


def cancel_from_ui(manager: Manager, job_id: str | None):
    if not job_id:
        return "Aucun job à annuler.", None, None
    with manager._lock:
        job = manager._jobs.pop(job_id, None)
    if job is None:
        return f"Job introuvable : {job_id}", None, None
    if job.state == "running" and job.pid:
        try:
            os.kill(job.pid, 15)
        except (ProcessLookupError, PermissionError):
            pass
    return f"Job annulé : {job_id}", None, None


def create_demo(manager: Manager):
    import gradio as gr

    with gr.Blocks(title="MiniMax H3") as demo:
        gr.Markdown(
            "# MiniMax H3\n"
            "Génération image-to-video locale, sans serveur HTTP ComfyUI.\n\n"
            "Timeline : `2:the woman walks|3:the woman looks left and right slowly`"
        )
        job_state = gr.State(value=None)
        with gr.Row():
            with gr.Column():
                first_frame = gr.Image(label="Image de départ", type="filepath")
                prompt = gr.Textbox(
                    label="Prompt ou timeline",
                    lines=5,
                    placeholder="2:the woman walks|3:the woman looks left and right slowly",
                )
                with gr.Row():
                    duration = gr.Number(label="Durée (s)", value=6, minimum=0.2)
                    fps = gr.Number(label="FPS de sortie", value=10, minimum=1)
                with gr.Row():
                    steps = gr.Number(label="Steps", value=8, minimum=1, precision=0)
                    seed = gr.Number(label="Seed", value=0, precision=0)
                audio = gr.Dropdown(
                    choices=list(AUDIO_POLICIES), value="first", label="Audio"
                )
                with gr.Row():
                    generate = gr.Button("Générer", variant="primary")
                    cancel = gr.Button("Annuler")
            with gr.Column():
                status = gr.Code(label="Statut", language="json", lines=16)
                video = gr.Video(label="Vidéo générée", autoplay=False)

        timer = gr.Timer(value=2.0)
        generate.click(
            fn=lambda image, text, seconds, output_fps, nsteps, job_seed, audio_mode: submit_from_ui(
                manager, image, text, seconds, output_fps, nsteps, job_seed, audio_mode
            ),
            inputs=[first_frame, prompt, duration, fps, steps, seed, audio],
            outputs=[job_state, status, video],
        )
        cancel.click(
            fn=lambda job_id: cancel_from_ui(manager, job_id),
            inputs=job_state,
            outputs=[status, job_state, video],
        )
        timer.tick(
            fn=lambda job_id: poll_from_ui(manager, job_id),
            inputs=job_state,
            outputs=[status, video],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax H3 Gradio UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--comfy-root", type=Path, default=Path("/home/francois/comfy/ComfyUI"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/gradio"))
    parser.add_argument("--log-dir", type=Path, default=Path("log/gradio"))
    parser.add_argument("--pid", type=Path, default=Path("runs/gradio/server.pid"))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not (
        os.environ.get("H3_GRADIO_USER") and os.environ.get("H3_GRADIO_PASSWORD")
    ):
        raise SystemExit("Refus de bind réseau sans H3_GRADIO_USER/H3_GRADIO_PASSWORD")

    manager = Manager(args.comfy_root, args.runs_dir, args.pid, args.log_dir)
    demo = create_demo(manager)
    user = os.environ.get("H3_GRADIO_USER")
    password = os.environ.get("H3_GRADIO_PASSWORD")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        auth=(user, password) if user and password else None,
        show_error=True,
    )


if __name__ == "__main__":
    main()
