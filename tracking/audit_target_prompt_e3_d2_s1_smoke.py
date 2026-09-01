#!/usr/bin/env python3
"""Bounded loader/GPU/DDP smoke for preregistered E3 D2-S1 P50."""

import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.models.entertrack import build_entertrack  # noqa: E402
from lib.train.actors.entertrack_threemdot import (  # noqa: E402
    EnTeRTrackActorThreeMDOT,
)
from lib.train.admin.settings import Settings  # noqa: E402
from lib.train.base_functions import (  # noqa: E402
    build_dataloaders_threemdot,
    get_optimizer_scheduler,
    update_settings,
)
from lib.train.data.sampler_threemdot import (  # noqa: E402
    TrackingSamplerThreeMDOT,
)
from lib.train.target_prompt_d2_s1_source_degradation import (  # noqa: E402
    apply_e3_d2_s1_source_degradation,
    d2_p1_clean_sample_id,
)
from lib.train.train_script import use_grouped_multiview_loader  # noqa: E402
from lib.utils.box_ops import giou_loss  # noqa: E402
from lib.utils.focal_loss import FocalLoss  # noqa: E402
from tracking.target_prompt_d2_p2_partial_degradation import (  # noqa: E402
    apply_partial_occlusion as apply_frozen_d2_p2,
)


SEED = 20260901
FROZEN_B0_SHA256 = (
    "363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3")


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_smoke_config(world_size=1):
    resolved = copy.deepcopy(cfg)
    update_config_from_file(os.path.join(
        ROOT, "experiments", "entertrack",
        "target_prompt_collaboration_e3_d2_s1.yaml"), base_cfg=resolved)
    # Smoke-only bounds; the formal YAML remains unchanged.
    resolved.TRAIN.NUM_WORKER = 0
    resolved.TRAIN.BATCH_SIZE = 2
    resolved.DATA.TRAIN.SAMPLE_PER_EPOCH = 2 * int(world_size)
    resolved.DATA.VAL.SAMPLE_PER_EPOCH = 2 * int(world_size)
    return resolved


def build_loaders(resolved, local_rank=-1):
    if not use_grouped_multiview_loader(resolved):
        raise RuntimeError("D2-S1 must resolve to grouped multiview loader")
    settings = Settings()
    settings.local_rank = int(local_rank)
    settings.use_lmdb = False
    update_settings(settings, resolved)
    train_loader, val_loader = build_dataloaders_threemdot(resolved, settings)
    if not isinstance(train_loader.dataset, TrackingSamplerThreeMDOT):
        raise RuntimeError("train loader is not TrackingSamplerThreeMDOT")
    if not isinstance(val_loader.dataset, TrackingSamplerThreeMDOT):
        raise RuntimeError("validation loader is not TrackingSamplerThreeMDOT")
    return settings, train_loader, val_loader


def validate_real_batch(batch):
    if tuple(batch["search_images"].shape[:2]) != (3, 2):
        raise RuntimeError("expected real search batch [V=3,B=2,...]")
    actual = tuple(tuple(str(value).upper() for value in row)
                   for row in batch["view_ids"])
    if actual != (("A", "A"), ("B", "B"), ("C", "C")):
        raise RuntimeError("non-canonical real ABC batch: {}".format(actual))
    if not bool(torch.as_tensor(batch["template_view_valid"]).all().item()):
        raise RuntimeError("template violates common-visible contract")
    if not bool(torch.as_tensor(batch["search_view_valid"]).all().item()):
        raise RuntimeError("search violates common-visible contract")
    if len(batch["search_frame_ids"]) != 1:
        raise RuntimeError("expected one synchronized search frame slot")


def nested_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [nested_list(item) for item in value]
    return value


def loader_smoke():
    resolved = load_smoke_config()
    _, train_loader, val_loader = build_loaders(resolved)
    reset_seed(SEED)
    train_batch = next(iter(train_loader))
    reset_seed(SEED + 1)
    val_batch = next(iter(val_loader))
    validate_real_batch(train_batch)
    validate_real_batch(val_batch)

    source = train_batch["search_images"].clone()
    template = train_batch["template_images"]
    annotation = train_batch["search_anno"]
    effective, audit = apply_e3_d2_s1_source_degradation(
        train_batch, resolved, training=True,
        generator=torch.Generator().manual_seed(SEED + 2))
    changed = (effective["search_images"] != source).flatten(2).any(2)
    if int(changed.sum().item()) != 1 or int(changed.any(0).sum().item()) != 1:
        raise RuntimeError("real batch did not change exactly one weak view")
    view_index, batch_index = [
        int(value.item()) for value in changed.nonzero()[0]]
    sample_id = d2_p1_clean_sample_id(
        train_batch, view_index, batch_index)
    frozen, frozen_audit = apply_frozen_d2_p2(
        source[view_index, batch_index],
        train_batch["search_anno"][view_index, batch_index],
        "P50", sample_id)
    if not torch.equal(
            frozen, effective["search_images"][view_index, batch_index]):
        raise RuntimeError("real P50 pixels differ from frozen D2-P2")
    if effective["template_images"] is not template:
        raise RuntimeError("D2-S1 replaced the template tensor")
    if effective["search_anno"] is not annotation:
        raise RuntimeError("D2-S1 replaced search supervision")
    validation, val_audit = apply_e3_d2_s1_source_degradation(
        val_batch, resolved, training=False)
    if validation is not val_batch or (
            validation["search_images"] is not val_batch["search_images"]):
        raise RuntimeError("validation is not exact bypass")

    return {
        "status": "PASS",
        "mode": "real_threemdot_loader",
        "seed": SEED,
        "formal_yaml_mutated": False,
        "official_test_accessed": False,
        "sampler_modified": False,
        "loader_class": type(train_loader.dataset).__name__,
        "grouped_multiview": True,
        "common_visible": True,
        "canonical_abc": True,
        "independent_view_sampling": False,
        "search_shape": list(train_batch["search_images"].shape),
        "template_shape": list(train_batch["template_images"].shape),
        "target_ids": [str(value) for value in train_batch["target_id"]],
        "search_frame_ids": nested_list(train_batch["search_frame_ids"]),
        "selected_triplets": audit["selected_triplets"],
        "weak_view_counts": [audit["weak_view_A"], audit["weak_view_B"],
                             audit["weak_view_C"]],
        "changed_view_triplet_indices": changed.nonzero().tolist(),
        "sample_id": sample_id,
        "orientation": frozen_audit["orientation"],
        "requested_coverage": audit["requested_coverage"],
        "realized_bbox_coverage": audit[
            "realized_bbox_coverage_mean"],
        "pixel_identity_with_frozen_d2_p2": True,
        "template_identity": effective["template_images"] is template,
        "annotation_identity": effective["search_anno"] is annotation,
        "validation_exact_bypass": validation is val_batch,
        "validation_applied": val_audit["applied"],
    }


def actor_for(net, resolved, settings):
    return EnTeRTrackActorThreeMDOT(
        net=net,
        objective={
            "giou": giou_loss,
            "l1": torch.nn.functional.l1_loss,
            "focal": FocalLoss(),
        },
        loss_weight={
            "giou": resolved.TRAIN.GIOU_WEIGHT,
            "l1": resolved.TRAIN.L1_WEIGHT,
            "focal": resolved.TRAIN.FOCAL_WEIGHT,
        },
        settings=settings,
        cfg=resolved,
    )


def gpu_or_ddp_smoke(ddp):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if ddp:
        if world_size < 2:
            raise RuntimeError("DDP smoke requires torchrun world_size >= 2")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    elif world_size != 1:
        raise RuntimeError("single-GPU smoke must not run under torchrun")
    device = torch.device("cuda:{}".format(local_rank))
    resolved = load_smoke_config(world_size=world_size)
    settings, train_loader, _ = build_loaders(
        resolved, local_rank=local_rank if ddp else -1)
    settings.device = device
    settings.batchsize = 2
    reset_seed(SEED + rank)
    batch = next(iter(train_loader))
    validate_real_batch(batch)
    batch = batch.to(device)
    batch["epoch"] = 1

    checkpoint_path = resolved.B0_CHECKPOINT
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT, checkpoint_path)
    if sha256_file(checkpoint_path) != FROZEN_B0_SHA256:
        raise RuntimeError("frozen B0 checkpoint SHA256 mismatch")
    model = build_entertrack(resolved, training=True).to(device)
    initialization = model.initialization_audit
    if not initialization["strict_full_load"]:
        raise RuntimeError("B0 strict full load failed")
    if initialization["checkpoint_path"] != checkpoint_path:
        raise RuntimeError("initialization did not use frozen B0 checkpoint")
    net = DDP(
        model, device_ids=[local_rank], find_unused_parameters=True,
        broadcast_buffers=False) if ddp else model
    optimizer, _ = get_optimizer_scheduler(net, resolved)
    trainable = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if sum(value.numel() for value in trainable.values()) != 148993:
        raise RuntimeError("unexpected trainable parameter count")
    if not all(name.startswith("target_prompt_collaboration.")
               for name in trainable):
        raise RuntimeError("non-adapter parameter remains trainable")

    actor = actor_for(net, resolved, settings)
    model.train()
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)
    loss, status = actor(batch)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("non-finite smoke loss")
    loss.backward()
    adapter_gradients = [
        parameter.grad for parameter in trainable.values()
        if parameter.grad is not None]
    if not adapter_gradients:
        raise RuntimeError("adapter received no gradients")
    if not all(bool(torch.isfinite(value).all().item())
               for value in adapter_gradients):
        raise RuntimeError("adapter gradient is non-finite")
    frozen_gradient_count = sum(
        int(parameter.grad is not None) for parameter in model.parameters()
        if not parameter.requires_grad)
    if frozen_gradient_count != 0:
        raise RuntimeError("frozen core received gradients")
    gradient_checksum = float(sum(
        value.detach().float().sum().item() for value in adapter_gradients))
    torch.cuda.synchronize(device)
    report = {
        "rank": rank,
        "loss": float(loss.detach().cpu().item()),
        "gradient_checksum": gradient_checksum,
        "finite_adapter_gradients": True,
        "frozen_core_gradient_count": frozen_gradient_count,
        "trainable_parameter_count": 148993,
        "trainable_only_e3_adapter": True,
        "strict_b0_core_load": True,
        "fresh_adapter_key_count": initialization["fresh_adapter_key_count"],
        "b0_checkpoint_sha256": FROZEN_B0_SHA256,
        "d2_s1_applied": status["D2S1/applied"],
        "selected_triplets": status["D2S1/selected_triplets"],
        "selected_ratio": status["D2S1/selected_ratio"],
        "realized_bbox_coverage": status[
            "D2S1/realized_bbox_coverage_mean"],
        "prompt_k": status["E3/prompt_k"],
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "official_test_accessed": False,
    }
    if ddp:
        reports = [None for _ in range(world_size)]
        dist.all_gather_object(reports, report)
        result = {
            "status": "PASS",
            "mode": "ddp_forward_backward",
            "world_size": world_size,
            "backend": dist.get_backend(),
            "ranks": reports,
            "backward_allreduce_completed": True,
            "optimizer_step_executed": False,
            "formal_training_started": False,
        }
        dist.barrier()
        dist.destroy_process_group()
        return result if rank == 0 else None
    return {
        "status": "PASS",
        "mode": "gpu_forward_backward",
        "world_size": 1,
        "rank": report,
        "formal_training_started": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("loader", "gpu", "ddp"), default="loader")
    args = parser.parse_args()
    if args.mode == "loader":
        result = loader_smoke()
    else:
        result = gpu_or_ddp_smoke(ddp=args.mode == "ddp")
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
