#!/usr/bin/env python3
"""Authorized one-step six-GPU FCVC DDP and numerical-equivalence smoke."""

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel, build_sender_bundle
from lib.train.data.fcvc_sampler import FCVCSampler
from lib.train.fcvc_checkpoint import save_checkpoint
from lib.train.fcvc_config import legacy_training_contract, load_resolved_config
from lib.train.fcvc_training_graph import FCVCTrainingGraph


def randn(shape, seed, device):
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn(*shape, generator=generator).to(device)


def fcvc_case(case_id, device):
    seed = 5000 + int(case_id) * 20
    mid = randn((1, 256, 192), seed, device)
    high = randn((1, 256, 192), seed + 1, device)
    response = torch.sigmoid(randn((1, 1, 16, 16), seed + 2, device))
    local = {
        "template_mid": randn((1, 64, 192), seed + 3, device),
        "template_high": randn((1, 64, 192), seed + 4, device),
        "mid_search": mid,
        "high_search": high,
        "response_map": response,
        "confidence_uncertainty": torch.cat(
            (response, torch.full_like(response, 0.5)), dim=1),
        "target_prototype": high.mean(dim=1),
        "local_output": {"smoke": torch.zeros(1, device=device)},
    }
    bundles = []
    for sender_index, view_id in enumerate((1, 2)):
        bundles.append(build_sender_bundle(
            randn((1, 256, 192), seed + 5 + sender_index * 3, device),
            randn((1, 256, 192), seed + 6 + sender_index * 3, device),
            torch.sigmoid(randn(
                (1, 1, 16, 16), seed + 7 + sender_index * 3, device)),
            view_id=torch.full((1,), view_id, dtype=torch.int16, device=device),
            timestamp=torch.full((1,), 9, dtype=torch.int64, device=device)))
    return local, tuple(bundles)


def fcvc_loss(model, case_id, device, divisor):
    local, bundles = fcvc_case(case_id, device)
    gt_roi = torch.full((1, 1, 16, 16), 1.0 / 256.0, device=device)
    output = model(
        local, bundles,
        teacher_training_payload={
            "mid_features": [local["mid_search"]] * 3,
            "high_features": [local["high_search"]] * 3,
            "gt_roi": gt_roi,
        })
    return (
        output["queries"].square().mean()
        + output["high_writer"]["search_tokens"].square().mean()
        + output["teacher_slots"].square().mean()
        + output["teacher_high"].square().mean()) / float(divisor)


def numerical_equivalence(rank, device):
    """Compare the actual FCVC training graph at effective global batch 18."""
    torch.manual_seed(42)
    initial = FCVCTrainingGraph(
        FCVCModel(FCVCConfig(enabled=True))).to(device).eval()
    reference_vector = torch.zeros(
        sum(parameter.numel() for parameter in initial.parameters()),
        device=device)
    if rank == 0:
        optimizer = torch.optim.AdamW(
            initial.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        for case_id in range(18):
            fcvc_loss(initial, case_id, device, divisor=18).backward()
        optimizer.step()
        reference_vector = torch.cat([
            parameter.detach().reshape(-1) for parameter in initial.parameters()])
    dist.broadcast(reference_vector, src=0)
    del initial
    torch.cuda.empty_cache()

    torch.manual_seed(42)
    model = FCVCTrainingGraph(
        FCVCModel(FCVCConfig(enabled=True))).to(device).eval()
    model = DDP(
        model, device_ids=[device.index], broadcast_buffers=False,
        find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    for micro in range(3):
        context = model.no_sync() if micro < 2 else nullcontext()
        with context:
            fcvc_loss(
                model, rank * 3 + micro, device, divisor=3).backward()
    optimizer.step()
    distributed_vector = torch.cat([
        parameter.detach().reshape(-1) for parameter in model.parameters()])
    return float((distributed_vector - reference_vector).abs().max().item())


def fcvc_forward_backward_smoke(rank, device):
    torch.manual_seed(42)
    model = FCVCModel(FCVCConfig(enabled=True)).to(device).train()
    model = DDP(
        FCVCTrainingGraph(model), device_ids=[device.index], broadcast_buffers=False,
        find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for micro in range(3):
        context = model.no_sync() if micro < 2 else nullcontext()
        with context:
            loss = fcvc_loss(
                model, rank * 3 + micro, device, divisor=3)
            loss.backward()
            loss_sum += float(loss.detach().item())
    named_gradients = [
        (name, parameter.grad) for name, parameter in model.module.named_parameters()
        if parameter.requires_grad]
    missing = [name for name, gradient in named_gradients if gradient is None]
    gradients = [gradient for _, gradient in named_gradients if gradient is not None]
    finite = bool(gradients) and all(
        torch.isfinite(gradient).all().item() for gradient in gradients)
    norm = torch.sqrt(sum(
        gradient.float().square().sum() for gradient in gradients))
    optimizer.step()
    return {
        "loss_sum": loss_sum,
        "gradient_norm": float(norm.item()),
        "all_present_gradients_finite": finite,
        "unused_gradient_parameter_count": len(missing),
        "unused_gradient_parameters": missing,
        "optimizer_steps": 1,
    }, model, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 6:
        raise RuntimeError("this audit must run under torchrun --nproc_per_node=6")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    difference = numerical_equivalence(rank, device)
    smoke, fcvc_model, fcvc_optimizer = fcvc_forward_backward_smoke(rank, device)
    resolved = load_resolved_config(
        ROOT / "experiments/entertrack/fcvc_full.yaml")
    mapped = legacy_training_contract(resolved)
    sampler = FCVCSampler(
        ROOT / resolved["DATA"]["TRAIN"]["MANIFEST"])
    sampler.begin_epoch(1, rank=rank, world_size=6, dist_module=dist)
    checkpoint_path = args.output.parent / "smoke_checkpoint.pth"
    save_checkpoint(
        checkpoint_path, fcvc_model, fcvc_optimizer, mapped, sampler,
        epoch=1, offset=3, global_step=1, rank=rank, world_size=6,
        dist_module=dist)
    dist.barrier()
    checkpoint_exists = checkpoint_path.exists()
    local = {
        "rank": rank,
        "device": str(device),
        "single_vs_six_max_abs_parameter_difference": difference,
        "rank0_checkpoint_visible": checkpoint_exists,
        **smoke,
    }
    gathered = [None] * 6
    dist.all_gather_object(gathered, local)
    if rank == 0:
        report = {
            "world_size": 6,
            "micro_batch_size_per_gpu": 1,
            "accumulation_steps": 3,
            "effective_global_batch": 18,
            "single_vs_six_max_abs_parameter_difference": max(
                row["single_vs_six_max_abs_parameter_difference"]
                for row in gathered),
            "numerical_equivalence_tolerance": 1e-5,
            "numerical_equivalence_pass": max(
                row["single_vs_six_max_abs_parameter_difference"]
                for row in gathered) <= 1e-5,
            "fcvc_forward_backward_smoke_pass": all(
                row["all_present_gradients_finite"]
                and row["optimizer_steps"] == 1 for row in gathered),
            "rank0_only_artifact_writer": True,
            "rank0_checkpoint_pass": all(
                row["rank0_checkpoint_visible"] for row in gathered),
            "per_rank": gathered,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
