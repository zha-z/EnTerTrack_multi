#!/usr/bin/env python3
"""Deterministic, dataset-free FCVC behavior identity audit."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.models.entertrack.fcvc import (
    FCVCConfig,
    FCVCModel,
    align_loss,
    cycle_loss,
    fcvc_total_loss,
    reconstruction_loss,
    safe_loss,
)
from tests.test_fcvc import local_record, sender
from tests.test_fcvc_runtime_integration import candidate, make_tracker


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def named_tensor_digest(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_digest(tensor).encode("ascii"))
    return digest.hexdigest()


def run_identity():
    torch.manual_seed(42)
    model = FCVCModel(FCVCConfig(enabled=True)).eval()
    local = local_record(batch=1)
    bundles = (sender(1, batch=1), sender(2, batch=1))
    output = model(local, bundles)

    gt_roi = torch.ones(1, 1, 16, 16)
    teacher = model.teacher(
        [local["mid_search"]] * 3,
        [local["high_search"]] * 3,
        gt_roi,
    )
    attention = output["global_match"]["attention_weights"]
    target = torch.full_like(attention, 1.0 / attention.shape[-1])
    losses = {
        "L_track_student": output["queries"].square().mean(),
        "L_align": align_loss(attention, target),
        "L_recon": reconstruction_loss(output["queries"], teacher),
        "L_safe": safe_loss(
            output["queries"].square().mean(),
            output["queries"].detach().square().mean() * 0.95,
        ),
        "L_cycle": cycle_loss(output["high_block"]["queries"], output["queries"]),
        "L_teacher_track": model.teacher.tracking_residual(
            local["high_search"].detach(), teacher
        ).square().mean(),
    }
    losses["L_total"] = fcvc_total_loss(
        losses["L_track_student"],
        losses["L_align"],
        losses["L_recon"],
        losses["L_safe"],
        losses["L_cycle"],
        losses["L_teacher_track"],
    )
    model.zero_grad(set_to_none=True)
    losses["L_total"].backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    tracker = make_tracker()
    state_digests = []
    for frame in range(5):
        tracker.frame_id += 1
        result = tracker.fcvc_predict_frame(
            local_candidate=candidate(frame),
            collaborative_candidate=candidate(
                frame, bbox=[1000.0 + frame, -1000.0, 5.0, 5.0]
            ),
            debug_assertions=True,
        )
        tracker.fcvc_commit_frame_result(result, debug_assertions=True)
        state_digests.append(tracker.fcvc_persistent_state_digest())

    student_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("teacher.")
    )
    teacher_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("teacher.")
    )
    output_tensors = {
        "queries": output["queries"],
        "reported_search_tokens": output["reported_output"]["fcvc_search_tokens"],
        "matcher_attention": attention,
        "mid_residual": output["mid_writer"]["residual"],
        "high_residual": output["high_writer"]["residual"],
    }
    loss_values = {
        name: float(value.detach().cpu().item()) for name, value in losses.items()
    }
    return {
        "seed": 42,
        "student_parameter_count": student_count,
        "teacher_parameter_count": teacher_count,
        "output_digest": named_tensor_digest(output_tensors),
        "output_tensor_digests": {
            name: tensor_digest(value) for name, value in sorted(output_tensors.items())
        },
        "loss_digest": hashlib.sha256(
            json.dumps(loss_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "loss_values": loss_values,
        "gradient_digest": named_tensor_digest(gradients),
        "gradient_tensor_count": len(gradients),
        "state_digest": state_digests[-1],
        "state_digests": state_digests,
        "dataset_accessed": False,
        "optimizer_step_called": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_identity()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
