from __future__ import annotations

import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import H3Config


@dataclass(frozen=True)
class ComfyRuntime:
    nodes: object
    h3_nodes: object
    sampler_nodes: object
    audio_nodes: object
    nested_tensor: object
    clipproj_nodes: object


def configure_memory_mode(args: Any) -> None:
    """Keep full-resolution H3 keyframes under the host memory ceiling."""
    args.disable_async_offload = True
    args.disable_pinned_memory = True
    args.lowvram = True
    args.novram = False
    args.highvram = False


def bootstrap(config: H3Config) -> ComfyRuntime:
    root = config.comfy_root.resolve()
    if not (root / "nodes.py").is_file():
        raise FileNotFoundError(f"ComfyUI not found at {root}")
    os.chdir(root)
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None

    import comfy.options

    comfy.options.args_parsing = False
    from comfy.cli_args import args

    configure_memory_mode(args)

    import nodes
    import comfy.nested_tensor as nested_tensor
    from comfy_extras import nodes_audio, nodes_custom_sampler, nodes_minimax_h3

    clipproj_root = root / "custom_nodes" / "ComfyUI-ClipProj"
    package_name = "h3runner_clipproj"
    spec = importlib.util.spec_from_file_location(
        package_name,
        clipproj_root / "__init__.py",
        submodule_search_locations=[str(clipproj_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load ClipProj from {clipproj_root}")
    clipproj_package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = clipproj_package
    spec.loader.exec_module(clipproj_package)
    clipproj_nodes = sys.modules[f"{package_name}.clipproj_nodes"]

    return ComfyRuntime(
        nodes=nodes,
        h3_nodes=nodes_minimax_h3,
        sampler_nodes=nodes_custom_sampler,
        audio_nodes=nodes_audio,
        nested_tensor=nested_tensor,
        clipproj_nodes=clipproj_nodes,
    )
