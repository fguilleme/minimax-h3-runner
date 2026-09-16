from __future__ import annotations

from typing import Any, Callable


def flatten_av_latent(latent: dict[str, Any]) -> dict[str, Any]:
    samples = latent.get("samples")
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError("MiniMax H3 latent must be a nested audio/video latent")
    streams = tuple(samples.unbind())
    if len(streams) != 2:
        raise ValueError("MiniMax H3 latent must contain exactly two streams")
    extras = {key: value for key, value in latent.items() if key != "samples"}
    return {"video": streams[0], "audio": streams[1], "extras": extras}


def restore_av_latent(flat: dict[str, Any], nested_factory: Callable[[tuple[Any, Any]], Any]) -> dict[str, Any]:
    if "video" not in flat or "audio" not in flat:
        raise ValueError("latent artifact is missing video or audio")
    latent = dict(flat.get("extras", {}))
    latent["samples"] = nested_factory((flat["video"], flat["audio"]))
    return latent
