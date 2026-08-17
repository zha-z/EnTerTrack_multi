#!/usr/bin/env python3
"""Six-GPU dry-run for FCVC pair/online validation and state isolation."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.config.entertrack.config import cfg as tracker_cfg, update_config_from_file
from lib.models.entertrack.entertrack import build_entertrack
from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel
from lib.train.admin import env_settings
from lib.train.data.fcvc_processing import FCVCProcessing
from lib.train.data.fcvc_sampler import FCVCSampler
from lib.train.dataset.threemdot import ThreeMDOT
from lib.train.fcvc_config import legacy_training_contract, load_resolved_config
from lib.train.fcvc_online_validation import OnlineValidator
from lib.train.fcvc_pair_validation import PairValidator
from lib.train.fcvc_training_graph import FCVCTrainingGraph
from lib.train.fcvc_validation_sampler import FixedPairValidationSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 6:
        raise RuntimeError("validation smoke requires torchrun world_size=6")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    resolved = load_resolved_config(
        ROOT / "experiments/entertrack/fcvc_full.yaml")
    mapped = legacy_training_contract(resolved)
    split_path = (
        ROOT / "output/train/entertrack/fcvc_full/splits/target_split.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_sha = (
        ROOT / "output/train/entertrack/fcvc_full/splits/target_split_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    pair_path = (
        ROOT / "output/train/entertrack/fcvc_full/validation/val_pair_manifest.csv")

    update_config_from_file(str(
        ROOT / "experiments/entertrack/entertrack_threemdot_lasot_ft_cons.yaml"))
    tracker = build_entertrack(tracker_cfg, training=True).to(device).eval()
    for parameter in tracker.parameters():
        parameter.requires_grad_(False)
    fcvc_cfg = FCVCConfig(**{
        key.lower(): value for key, value in resolved["MODEL"]["FCVC"].items()})
    fcvc = FCVCModel(fcvc_cfg).to(device).train()
    optimizer = torch.optim.AdamW(fcvc.parameters(), lr=mapped["student_lr"])
    model = DDP(
        FCVCTrainingGraph(fcvc), device_ids=[local_rank],
        output_device=local_rank, broadcast_buffers=False,
        find_unused_parameters=False)
    dataset = ThreeMDOT(split="train")
    processing = FCVCProcessing(dataset, device, tracker, model)
    source_manifest = (ROOT / resolved["DATA"]["TRAIN"]["MANIFEST"]).resolve()
    training_sampler = FCVCSampler(
        source_manifest, allowed_targets=split["train_targets"])
    training_sampler.begin_epoch(
        1, rank=rank, world_size=6, dist_module=dist)
    pair_sampler = FixedPairValidationSampler(
        pair_path, split["validation_targets"])
    pair = PairValidator(
        model, tracker, processing, pair_sampler, optimizer,
        training_sampler, args.output.parent, device, rank=rank,
        world_size=6, dist_module=dist, epsilon=1e-6)
    pair_metrics, pair_isolation = pair.run(
        epoch=1, max_local_batches=1, write_outputs=False)

    online = OnlineValidator(
        model, tracker, env_settings().threemdot_dir,
        split["validation_targets"], device, rank=rank, world_size=6,
        dist_module=dist, epsilon=1e-6)
    online_metrics, online_isolation = online.run(epoch=5, max_frames=3)
    if rank == 0:
        partitions = []
        for part_rank in range(6):
            audit_sampler = FixedPairValidationSampler(
                pair_path, split["validation_targets"])
            rows = audit_sampler.partition(part_rank)
            partitions.append({
                "rank": part_rank, "cases": len(rows),
                "first_case": int(rows[0]["case_index"]),
                "last_case": int(rows[-1]["case_index"]),
            })
        report = {
            "formal_training_executed": False,
            "formal_test_executed": False,
            "threemdot_test_accessed": False,
            "world_size": 6,
            "split_sha256": split_sha,
            "pair_manifest_sha256": pair_sampler.sha256,
            "pair_full_contract": pair_sampler.audit(),
            "pair_smoke_metrics": pair_metrics,
            "pair_state_isolation": pair_isolation,
            "pair_partitions": partitions,
            "online_smoke_metrics": online_metrics,
            "online_state_isolation": online_isolation,
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
