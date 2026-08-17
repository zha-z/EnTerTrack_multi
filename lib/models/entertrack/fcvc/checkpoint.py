"""Backward-compatible FCVC checkpoint loading without key renaming."""

from pathlib import Path

import torch


def normalize_fcvc_state_dict(checkpoint):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("FCVC checkpoint must be a mapping")
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "student" in checkpoint or "teacher" in checkpoint:
        state = {}
        state.update(checkpoint.get("student", {}))
        state.update(checkpoint.get("teacher", {}))
    else:
        state = checkpoint
    normalized = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        name = name[5:] if name.startswith("fcvc.") else name
        normalized[name] = value
    return normalized


def load_fcvc_checkpoint(model, checkpoint, strict=True):
    return model.load_state_dict(
        normalize_fcvc_state_dict(checkpoint), strict=strict
    )
