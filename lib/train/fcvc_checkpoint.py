"""Rank-safe FCVC checkpoint and exact-resume contract."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


REQUIRED_KEYS = {
    "current_epoch", "within_epoch_case_offset", "global_optimizer_step",
    "student", "teacher", "optimizer", "scheduler", "sampler_state",
    "epoch_manifest", "manifest_sha256", "rng_state_by_rank",
    "training_config_sha256", "world_size", "accumulation_steps",
    "sample_per_epoch", "validation_metadata",
}


def config_digest(config):
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _module(model):
    module = model.module if hasattr(model, "module") else model
    if getattr(module, "is_fcvc_training_graph", False):
        module = module.fcvc
    return module


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])


def empty_resume_state():
    return {
        "current_epoch": 1,
        "within_epoch_case_offset": 0,
        "global_optimizer_step": 0,
        "checkpoint_manifest": None,
    }


def save_checkpoint(path, model, optimizer, config, sampler, epoch, offset,
                    global_step, rank=0, world_size=1, dist_module=None,
                    validation_metadata=None):
    local_rng = capture_rng_state()
    if dist_module is not None and world_size > 1:
        rng_states = [None] * world_size
        dist_module.all_gather_object(rng_states, local_rng)
    else:
        rng_states = [local_rng]
    if rank != 0:
        return None

    module = _module(model)
    state = module.state_dict()
    manifest = dict(sampler.current_contract)
    payload = {
        "current_epoch": int(epoch),
        "within_epoch_case_offset": int(offset),
        "global_optimizer_step": int(global_step),
        "student": {
            key: value.detach().cpu() for key, value in state.items()
            if not key.startswith("teacher.")
        },
        "teacher": {
            key: value.detach().cpu() for key, value in state.items()
            if key.startswith("teacher.")
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": {
            "type": config["scheduler"]["type"], "last_step": int(global_step),
            "warmup_steps": 556, "total_steps": 16680,
        },
        "sampler_state": sampler.state_dict(offset),
        "epoch_manifest": manifest,
        "manifest_sha256": manifest["manifest_sha256"],
        "rng_state_by_rank": rng_states,
        "training_config_sha256": config_digest(config),
        "world_size": int(world_size),
        "accumulation_steps": int(config["gradient_accumulation_steps"]),
        "sample_per_epoch": int(config["sample_per_epoch"]),
        "validation_metadata": validation_metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, optimizer, config, rank=0, world_size=1):
    if path is None:
        return empty_resume_state()
    checkpoint = torch.load(Path(path), map_location="cpu")
    missing = REQUIRED_KEYS - set(checkpoint)
    if missing:
        raise RuntimeError("resume checkpoint missing keys: {}".format(
            ",".join(sorted(missing))))
    expected = {
        "world_size": 6,
        "accumulation_steps": 3,
        "sample_per_epoch": 10008,
        "training_config_sha256": config_digest(config),
    }
    for key, value in expected.items():
        if checkpoint[key] != value:
            raise RuntimeError("resume {} mismatch: {} != {}".format(
                key, checkpoint[key], value))
    if int(world_size) != checkpoint["world_size"]:
        raise RuntimeError("resume requires the original world_size=6")
    module = _module(model)
    module.load_state_dict(
        {**checkpoint["student"], **checkpoint["teacher"]}, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    states = checkpoint["rng_state_by_rank"]
    if len(states) != world_size:
        raise RuntimeError("checkpoint does not contain one RNG state per rank")
    restore_rng_state(states[int(rank)])
    return {
        "current_epoch": int(checkpoint["current_epoch"]),
        "within_epoch_case_offset": int(checkpoint["within_epoch_case_offset"]),
        "global_optimizer_step": int(checkpoint["global_optimizer_step"]),
        "checkpoint_manifest": checkpoint["epoch_manifest"],
        "validation_metadata": checkpoint.get("validation_metadata"),
    }


def export_student(path, model):
    module = _module(model)
    student = {
        key: value.detach().cpu() for key, value in module.state_dict().items()
        if not key.startswith("teacher.")
    }
    payload = {
        "student": student,
        "student_parameter_count": sum(value.numel() for value in student.values()),
        "teacher_key_count": 0,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
