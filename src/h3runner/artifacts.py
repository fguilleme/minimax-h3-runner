from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file


SCHEMA_VERSION = 1


def _pack(value: Any, tensors: dict[str, Any]) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        name = f"tensor_{len(tensors):06d}"
        tensors[name] = value.detach().to("cpu").contiguous()
        return {"kind": "tensor", "name": name}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("unsupported dictionary key type")
        return {"kind": "dict", "items": {key: _pack(item, tensors) for key, item in value.items()}}
    if isinstance(value, list):
        return {"kind": "list", "items": [_pack(item, tensors) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_pack(item, tensors) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def _unpack(node: Any, tensors: dict[str, Any]) -> Any:
    kind = node["kind"]
    if kind == "tensor":
        return tensors[node["name"]]
    if kind == "dict":
        return {key: _unpack(item, tensors) for key, item in node["items"].items()}
    if kind == "list":
        return [_unpack(item, tensors) for item in node["items"]]
    if kind == "tuple":
        return tuple(_unpack(item, tensors) for item in node["items"])
    if kind == "scalar":
        return node["value"]
    raise ValueError(f"unknown artifact node kind: {kind}")


def _paths(prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    return prefix.with_suffix(".safetensors"), prefix.with_suffix(".json")


def save_tree(prefix: str | Path, value: Any, artifact_type: str) -> dict[str, Any]:
    tensor_path, manifest_path = _paths(prefix)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    tree = _pack(value, tensors)
    if not tensors:
        raise ValueError("artifact must contain at least one tensor")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "tensor_file": tensor_path.name,
        "tree": tree,
    }

    tensor_fd, tensor_tmp_name = tempfile.mkstemp(prefix=tensor_path.name + ".", dir=tensor_path.parent)
    manifest_fd, manifest_tmp_name = tempfile.mkstemp(prefix=manifest_path.name + ".", dir=manifest_path.parent)
    os.close(tensor_fd)
    os.close(manifest_fd)
    tensor_tmp = Path(tensor_tmp_name)
    manifest_tmp = Path(manifest_tmp_name)
    try:
        save_file(tensors, tensor_tmp)
        manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        os.replace(tensor_tmp, tensor_path)
        os.replace(manifest_tmp, manifest_path)
    finally:
        tensor_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return manifest


def load_tree(prefix: str | Path, expected_type: str | None = None) -> tuple[Any, dict[str, Any]]:
    tensor_path, manifest_path = _paths(prefix)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact schema: {manifest.get('schema_version')}")
    if expected_type is not None and manifest.get("artifact_type") != expected_type:
        raise ValueError(
            f"artifact type is {manifest.get('artifact_type')!r}, expected {expected_type!r}"
        )
    tensors = load_file(tensor_path, device="cpu")
    return _unpack(manifest["tree"], tensors), manifest
