#!/usr/bin/env python3
"""Run one deterministic B1 inference and export token-path diagnostics."""

import argparse
import json
from pathlib import Path

import torch

from lib.models.entertrack import build_entertrack
from tracking.audit_b0_pcum_config import load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "output/diagnostics/b1_abc_arp/formal_20260819_seed42_4gpu_r001"
    / "checkpoints/train/entertrack/b1_abc_arp_4gpu/EnTeRTrack_ep0025.pth.tar"
)
DEFAULT_OUTPUT = (
    ROOT / "docs/results/b1_abc_arp_controlled_20260819"
    / "inference_path_smoke.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.manual_seed(42)
    config = load_config("b1_abc_arp_4gpu")
    model = build_entertrack(config, training=False)
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload["net"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    model.eval()

    template = torch.randn(1, 3, 128, 128)
    search = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(template, search, training=False, return_atp=True)

    removed = output["removed_indexes_s"]
    removed_counts = [int(item.numel()) for item in removed]
    keep_masks = output["atp_keep_masks"]
    keep_counts = [int(mask.to(torch.int64).sum().item()) for mask in keep_masks]
    compensation_masks = output["compensation_masks"]
    compensation_counts = [
        int(mask.to(torch.int64).sum().item()) for mask in compensation_masks
    ]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "training_flag": False,
        "input_template_shape": list(template.shape),
        "input_search_shape": list(search.shape),
        "initial_search_tokens": int(output["arp_initial_search_tokens"]),
        "physically_removed_tokens_per_layer": removed_counts,
        "kept_tokens_per_layer": keep_counts,
        "compensated_tokens_per_layer": compensation_counts,
        "restored_search_tokens": int(output["arp_output_search_tokens"]),
        "backbone_feature_shape": list(output["backbone_feat"].shape),
        "score_map_shape": list(output["score_map"].shape),
        "bbox_shape": list(output["pred_boxes"].shape),
        "pcum_present": model.pcum is not None,
        "c3r_present": model.c3r is not None,
        "remote_input_used": False,
        "finite_score_map": bool(torch.isfinite(output["score_map"]).all()),
        "finite_bbox": bool(torch.isfinite(output["pred_boxes"]).all()),
        "pass": bool(
            sum(removed_counts) > 0
            and int(output["arp_output_search_tokens"]) == 256
            and output["score_map"].shape[-2:] == (16, 16)
            and model.pcum is None
            and model.c3r is None
            and torch.isfinite(output["score_map"]).all()
            and torch.isfinite(output["pred_boxes"]).all()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
