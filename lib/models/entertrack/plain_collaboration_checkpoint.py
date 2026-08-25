"""Strict B0 checkpoint loading for Plain Collaboration V1."""

import torch


def _state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    state = checkpoint.get("net", None)
    if state is None:
        state = checkpoint.get("model", None)
    return state


def load_plain_collaboration_initialization(model, checkpoint_path):
    """Load every B0 tensor exactly and preserve only fresh adapter tensors."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _state_dict(checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError(
            "Plain collaboration checkpoint has no state dict: {}".format(
                checkpoint_path))
    if state and all(key.startswith("module.") for key in state):
        state = {
            key[len("module."):]: value for key, value in state.items()
        }
    inherited_adapter = sorted(
        key for key in state if key.startswith("plain_collaboration."))
    if inherited_adapter:
        raise RuntimeError(
            "V1 initialization refuses inherited adapter tensors: {}".format(
                inherited_adapter[:20]))

    model_state = model.state_dict()
    local_keys = sorted(
        key for key in model_state
        if not key.startswith("plain_collaboration."))
    missing_local = sorted(set(local_keys) - set(state))
    source_only = sorted(set(state) - set(local_keys))
    shape_mismatches = sorted(
        (key, tuple(state[key].shape), tuple(model_state[key].shape))
        for key in set(local_keys).intersection(state)
        if tuple(state[key].shape) != tuple(model_state[key].shape)
    )
    if missing_local or source_only or shape_mismatches:
        raise RuntimeError(
            "V1 frozen-B0 initialization failed: missing_local={}, "
            "source_only={}, shape_mismatches={}".format(
                missing_local, source_only, shape_mismatches))

    merged = dict(model_state)
    for key in local_keys:
        merged[key] = state[key]
    model.load_state_dict(merged, strict=True)
    report = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "loaded_local_key_count": len(local_keys),
        "fresh_adapter_key_count": len([
            key for key in model_state
            if key.startswith("plain_collaboration.")]),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "strict_full_load": True,
    }
    model.initialization_audit = report
    return report
