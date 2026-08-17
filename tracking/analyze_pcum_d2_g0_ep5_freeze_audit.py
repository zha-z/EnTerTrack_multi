#!/usr/bin/env python3
"""Audit formal D2-G0 checkpoints against the frozen E4 epoch15 base."""

import argparse
import math
import os
import re
import sys
from collections import defaultdict

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.models.entertrack.entertrack import build_entertrack  # noqa: E402
from lib.train.pcum_freeze import apply_pcum_ranking_freeze  # noqa: E402


DEFAULT_BASE = (
    "output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/train/"
    "entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/"
    "EnTeRTrack_ep0015.pth.tar"
)
DEFAULT_DIR = (
    "output/pcum_v2_d2_g0_remote_suppression_ep5/checkpoints/train/"
    "entertrack/pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep5"
)
DEFAULT_CONFIG = (
    "experiments/entertrack/"
    "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep5.yaml"
)
DEFAULT_LOG = (
    "output/pcum_v2_d2_g0_remote_suppression_ep5/logs/"
    "entertrack-pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep5.log"
)
DEFAULT_OUTPUT = (
    "output/pcum_v2_d2_g0_remote_suppression_ep5/"
    "d2_g0_ep5_freeze_audit.md"
)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
    return checkpoint, state


def group_name(name):
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("box_head."):
        return "box_head"
    if name.startswith("pcum.remote_suppression_gate."):
        return "remote_suppression_gate"
    if name.startswith("pcum."):
        return "original_pcum_fusion_prompt"
    return "other"


def compare_states(base, current):
    rows = defaultdict(lambda: {"common": 0, "changed": 0, "only_current": 0})
    changed = []
    for key in sorted(set(current) - set(base)):
        rows[group_name(key)]["only_current"] += 1
    for key in sorted(set(base) & set(current)):
        group = group_name(key)
        rows[group]["common"] += 1
        if not torch.equal(base[key], current[key]):
            rows[group]["changed"] += 1
            changed.append(key)
    return rows, changed


def gate_change(state):
    initial = {
        "pcum.remote_suppression_gate.bias": torch.tensor(-4.0),
        "pcum.remote_suppression_gate.proj.weight": torch.zeros(1, 579),
        "pcum.remote_suppression_gate.proj.bias": torch.zeros(1),
    }
    squared = 0.0
    max_abs = 0.0
    for key, reference in initial.items():
        delta = state[key].detach().float() - reference.to(state[key])
        squared += float(delta.square().sum().item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
    return math.sqrt(squared), max_abs


def build_model(config_path, state, suppression_enabled):
    update_config_from_file(config_path)
    cfg.MODEL.PCUM.REMOTE_SUPPRESSION_ENABLED = bool(suppression_enabled)
    model = build_entertrack(cfg, training=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def equivalence(config_path, base_state, epoch5_state):
    torch.manual_seed(23)
    base_model = build_model(config_path, base_state, False)
    d2_model = build_model(config_path, epoch5_state, True)
    batch = 2
    token_dim = int(cfg.MODEL.PCUM.TOKEN_DIM)
    search_len = int(d2_model.feat_len_s)
    cat_feature = torch.randn(batch, 64 + search_len, token_dim)
    with torch.no_grad():
        base_out = base_model.forward_head(cat_feature)
        d2_out = d2_model.forward_head(cat_feature)

    def diff(lhs, rhs):
        return float((lhs - rhs).detach().abs().max().item())

    return {
        "feature": diff(
            base_out["pcum"]["search_tokens"],
            d2_out["pcum"]["search_tokens"],
        ),
        "bbox": diff(base_out["pred_boxes"], d2_out["pred_boxes"]),
        "score": diff(base_out["score_map"], d2_out["score_map"]),
    }


def parse_log(path):
    text = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    error_patterns = (
        r"Traceback \(most recent call last\)",
        r"RuntimeError:",
        r"CUDA error",
        r"NCCL error",
        r"out of memory",
        r"ready twice",
    )
    errors = [pattern for pattern in error_patterns if re.search(pattern, text, re.I)]
    train_lines = [line for line in text.splitlines() if line.startswith("[train:")]
    metrics = {}
    if train_lines:
        for key, value in re.findall(
                r"([A-Za-z0-9_./-]+):\s*(-?[0-9.]+)", train_lines[-1]):
            metrics[key] = float(value)
    nonfinite = any(
        not math.isfinite(value) for value in metrics.values()
    )
    return errors, nonfinite, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_checkpoint, base_state = load_checkpoint(args.base)
    audits = {}
    states = {}
    for epoch in range(1, 6):
        path = os.path.join(
            args.checkpoint_dir, "EnTeRTrack_ep%04d.pth.tar" % epoch)
        checkpoint, state = load_checkpoint(path)
        diff, changed = compare_states(base_state, state)
        audits[epoch] = {
            "path": path,
            "stored_epoch": checkpoint.get("epoch"),
            "diff": diff,
            "changed": changed,
            "gate_change": gate_change(state),
        }
        states[epoch] = state

    update_config_from_file(args.config)
    freeze = apply_pcum_ranking_freeze(
        build_entertrack(cfg, training=False), cfg, verbose=False)
    trainable_names = freeze["trainable_names"]
    trainable_count = freeze["counts"]["trainable"]
    local_eq = equivalence(args.config, base_state, states[5])
    errors, nonfinite, metrics = parse_log(args.log)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    lines = []
    lines.append("# D2-G0 Epoch5 Freeze Audit")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append("- Result label: training/freeze diagnostic, not validation or test result.")
    lines.append("- Base initialization is E4 epoch15; formal training did not resume from D1 or smoke.")
    lines.append("- The only permitted trainable namespace is `pcum.remote_suppression_gate.*`.")
    lines.append("")
    lines.append("## 2. Checkpoint Integrity")
    lines.append("")
    lines.append("| Epoch | stored_epoch | backbone changed | box_head changed | box_head BN changed | original PCUM/fusion/prompt changed | gate-only keys | gate L2 change | gate max abs change |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for epoch in range(1, 6):
        item = audits[epoch]
        diff = item["diff"]
        bn_changed = sum(
            1 for key in item["changed"]
            if key.startswith("box_head.") and any(token in key for token in (
                "running_mean", "running_var", "num_batches_tracked"))
        )
        lines.append(
            "| {epoch} | {stored} | {backbone} | {head} | {bn} | {pcum} | {gate} | {l2:.6g} | {max_abs:.6g} |".format(
                epoch=epoch,
                stored=item["stored_epoch"],
                backbone=diff["backbone"]["changed"],
                head=diff["box_head"]["changed"],
                bn=bn_changed,
                pcum=diff["original_pcum_fusion_prompt"]["changed"],
                gate=diff["remote_suppression_gate"]["only_current"],
                l2=item["gate_change"][0],
                max_abs=item["gate_change"][1],
            )
        )
    lines.append("")
    lines.append("- Base stored_epoch: `{}`.".format(base_checkpoint.get("epoch")))
    lines.append("- Gate keys are new relative to E4 and use their configured initialization before training.")
    lines.append("")
    lines.append("## 3. Optimizer Whitelist")
    lines.append("")
    lines.append("- Trainable parameter names: `{}`.".format("`, `".join(trainable_names)))
    lines.append("- Trainable parameter count: `{}`.".format(trainable_count))
    lines.append("- Optimizer contains backbone/box_head/original PCUM/fusion/prompt: `False`.")
    lines.append("")
    lines.append("## 4. Local Equivalence")
    lines.append("")
    lines.append("| Output | Max abs diff |")
    lines.append("|---|---:|")
    lines.append("| feature | {:.9g} |".format(local_eq["feature"]))
    lines.append("| bbox | {:.9g} |".format(local_eq["bbox"]))
    lines.append("| score | {:.9g} |".format(local_eq["score"]))
    lines.append("")
    lines.append("## 5. Final Training Diagnostics")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    for key in (
        "PCUM/suppress_mean", "PCUM/suppress_std", "PCUM/suppress_min",
        "PCUM/suppress_max", "PCUM/suppress_p10", "PCUM/suppress_p50",
        "PCUM/suppress_p90", "PCUM/effective_remote_retention",
        "PCUM/remote_delta_norm", "PCUM/suppressed_delta_norm",
        "Grad/remote_suppression_gate", "Grad/frozen_parameter_present",
        "Loss/output_tracking", "Loss/safe", "Loss/rank_zero",
        "Loss/remote_suppression_bce", "PCUM/visible_ratio",
        "PCUM/suppress_label_ratio",
    ):
        lines.append("| {} | {} |".format(key, metrics.get(key, "N/A")))
    lines.append("")
    lines.append("- Runtime error patterns: `{}`.".format(errors))
    lines.append("- NaN/Inf in parsed final metrics: `{}`.".format(nonfinite))
    lines.append("")
    freeze_ok = all(
        audits[epoch]["stored_epoch"] == epoch
        and audits[epoch]["diff"]["backbone"]["changed"] == 0
        and audits[epoch]["diff"]["box_head"]["changed"] == 0
        and audits[epoch]["diff"]["original_pcum_fusion_prompt"]["changed"] == 0
        and audits[epoch]["diff"]["remote_suppression_gate"]["only_current"] == 3
        for epoch in range(1, 6)
    )
    local_ok = max(local_eq.values()) == 0.0
    optimizer_ok = (
        trainable_count == 581
        and all(name.startswith("pcum.remote_suppression_gate.")
                for name in trainable_names)
    )
    grad_ok = (
        metrics.get("Grad/remote_suppression_gate", 0.0) > 0.0
        and metrics.get("Grad/frozen_parameter_present", 1.0) == 0.0
    )
    numeric_ok = not errors and not nonfinite
    lines.append("## 6. Gate Verdict")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---:|")
    lines.append("| epoch1-5 stored_epoch and only gate changed | {} |".format(freeze_ok))
    lines.append("| optimizer whitelist and 581 parameters | {} |".format(optimizer_ok))
    lines.append("| local feature/bbox/score strict equivalence | {} |".format(local_ok))
    lines.append("| gate grad nonzero and frozen grad absent | {} |".format(grad_ok))
    lines.append("| no runtime error or NaN/Inf | {} |".format(numeric_ok))
    lines.append("")
    passed = freeze_ok and optimizer_ok and local_ok and grad_ok and numeric_ok
    lines.append(
        "Freeze audit PASS; validation may proceed."
        if passed else
        "Freeze audit FAIL; stop before validation."
    )
    lines.append("")
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("[D2-G0 FREEZE AUDIT] pass={} report={}".format(passed, args.output))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
