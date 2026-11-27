"""Versioned architecture-bound checkpoint I/O, with no implicit loading."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import torch

SCHEMA = "rvloc_seven_groups_single_query_depth_v3"
ARCHITECTURE = "single_query_depth_v3"


def save_checkpoint(model, path, *, optimizer=None, metadata=None):
    """Write only when explicitly called; construction never loads weights."""
    if model.config.get("architecture") != ARCHITECTURE:
        raise ValueError("Only single_query_depth_v3 models may use this checkpoint schema")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "architecture": ARCHITECTURE, "group": model.group, "config": model.config,
               "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "metadata": dict(metadata or {})}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, *, device="cpu", expected_group=None):
    """Read an explicitly named code-version-compatible checkpoint."""
    from . import build_model
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported checkpoint schema: v1/v2 weights are not interchangeable with single_query_depth_v3")
    if payload.get("architecture") != ARCHITECTURE or payload.get("config", {}).get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint architecture must be single_query_depth_v3")
    if expected_group is not None:
        expected = build_model(expected_group, payload["config"]).group
        if payload.get("group") != expected:
            raise ValueError("Checkpoint group does not match requested experiment")
    model = build_model(payload["group"], payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    model.core.trained = payload.get("metadata", {}).get("optimizer_updates", 0) > 0
    return model, payload
