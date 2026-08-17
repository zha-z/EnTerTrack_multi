#!/usr/bin/env python3
"""Reduced CPU fallback smoke; not a substitute for the required six-GPU smoke."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lib.config.entertrack.config import cfg as tracker_cfg, update_config_from_file
from lib.models.entertrack.entertrack import build_entertrack
from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel
from lib.train.admin import env_settings
from lib.train.data.fcvc_processing import FCVCProcessing
from lib.train.dataset.threemdot import ThreeMDOT
from lib.train.fcvc_checkpoint import capture_rng_state
from lib.train.fcvc_config import load_resolved_config
from lib.train.fcvc_online_validation import OnlineValidator
from lib.train.fcvc_pair_validation import _case_values, state_digest
from lib.train.fcvc_training_graph import FCVCTrainingGraph
from lib.train.fcvc_validation_sampler import FixedPairValidationSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    resolved = load_resolved_config(
        ROOT / "experiments/entertrack/fcvc_full.yaml")
    split = json.loads((
        ROOT / "output/train/entertrack/fcvc_full/splits/target_split.json"
    ).read_text(encoding="utf-8"))
    update_config_from_file(str(
        ROOT / "experiments/entertrack/entertrack_threemdot_lasot_ft_cons.yaml"))
    tracker = build_entertrack(tracker_cfg, training=True).to(device).eval()
    for parameter in tracker.parameters():
        parameter.requires_grad_(False)
    fcvc = FCVCModel(FCVCConfig(**{
        key.lower(): value for key, value in resolved["MODEL"]["FCVC"].items()
    })).to(device).train()
    model = FCVCTrainingGraph(fcvc)
    dataset = ThreeMDOT(split="train")
    processing = FCVCProcessing(dataset, device, tracker, model)
    pair_sampler = FixedPairValidationSampler(
        ROOT / "output/train/entertrack/fcvc_full/validation/val_pair_manifest.csv",
        split["validation_targets"])
    rows = pair_sampler.full_rows[:3]
    params_before = state_digest(fcvc.state_dict())
    rng_before = state_digest(capture_rng_state())
    model.eval()
    with torch.inference_mode():
        pair_values = [
            _case_values(fcvc, tracker, case, epsilon=1e-6)
            for case in processing(rows)]
    model.train()
    pair_isolation = {
        "parameters_unchanged": params_before == state_digest(fcvc.state_dict()),
        "rng_unchanged": rng_before == state_digest(capture_rng_state()),
        "teacher_called": False,
        "gt_roi_used": False,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
    }
    online = OnlineValidator(
        model, tracker, env_settings().threemdot_dir,
        split["validation_targets"][:1], device, rank=0, world_size=1,
        dist_module=None, epsilon=1e-6)
    online_metrics, online_isolation = online.run(epoch=5, max_frames=2)
    report = {
        "scope": "reduced_cpu_fallback_not_six_gpu",
        "pair_cases": 3,
        "pair_values": pair_values,
        "pair_isolation": pair_isolation,
        "online_target": split["validation_targets"][0],
        "online_frames_per_view": 2,
        "online_metrics": online_metrics,
        "online_isolation": online_isolation,
        "threemdot_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
