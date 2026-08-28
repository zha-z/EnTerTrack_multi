"""Strict B0-core initialization for the fresh E3 adapter."""

import torch


def _state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        checkpoint = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def load_target_prompt_collaboration_initialization(model, checkpoint_path):
    incoming = _state_dict(checkpoint_path)
    current = model.state_dict()
    inherited_adapter = sorted(
        key for key in incoming
        if key.startswith("target_prompt_collaboration."))
    if inherited_adapter:
        raise RuntimeError(
            "E3 B0 initialization refuses inherited adapter tensors: {}".format(
                inherited_adapter[:20]))
    prefix = "target_prompt_collaboration."
    extractor_prefix = "target_prompt_extractor."
    adapter_keys = {
        key for key in current
        if key.startswith(prefix) or key.startswith(extractor_prefix)}
    core_keys = set(current) - adapter_keys
    missing_core = sorted(core_keys - set(incoming))
    unexpected = sorted(
        key for key in incoming
        if key not in core_keys and not key.startswith((prefix, extractor_prefix)))
    shape_mismatch = sorted(
        key for key in core_keys & set(incoming)
        if tuple(current[key].shape) != tuple(incoming[key].shape))
    if missing_core or unexpected or shape_mismatch:
        raise RuntimeError(
            "E3 B0 initialization mismatch: missing={}, unexpected={}, shape={}".format(
                missing_core[:20], unexpected[:20], shape_mismatch[:20]))
    merged = dict(current)
    for key in core_keys:
        merged[key] = incoming[key]
    model.load_state_dict(merged, strict=True)
    report = {
        "strict_full_load": True,
        "fresh_adapter_key_count": len(adapter_keys),
        "fresh_adapter_keys": sorted(adapter_keys),
        "loaded_core_key_count": len(core_keys),
        "checkpoint_path": checkpoint_path,
    }
    model.initialization_audit = report
    return report
