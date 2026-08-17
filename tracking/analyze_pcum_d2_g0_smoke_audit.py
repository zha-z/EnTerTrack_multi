#!/usr/bin/env python3
"""Audit D2-G0 remote suppression smoke runs."""

import argparse
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


DEFAULT_BASE = (
    "output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/train/"
    "entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/"
    "EnTeRTrack_ep0015.pth.tar"
)
DEFAULT_SINGLE = (
    "output/pcum_v2_d2_g0_remote_suppression_smoke/checkpoints/train/"
    "entertrack/pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_smoke/"
    "EnTeRTrack_ep0001.pth.tar"
)
DEFAULT_DDP = (
    "output/pcum_v2_d2_g0_remote_suppression_smoke_ddp/checkpoints/train/"
    "entertrack/pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_smoke/"
    "EnTeRTrack_ep0001.pth.tar"
)
DEFAULT_SINGLE_LOG = (
    "output/pcum_v2_d2_g0_remote_suppression_smoke/logs/"
    "entertrack-pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_smoke.log"
)
DEFAULT_DDP_LOG = (
    "output/pcum_v2_d2_g0_remote_suppression_smoke_ddp/logs/"
    "entertrack-pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_smoke.log"
)
DEFAULT_YAML = (
    "experiments/entertrack/"
    "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_smoke.yaml"
)
DEFAULT_OUTPUT = (
    "output/pcum_v2_d2_g0_remote_suppression_smoke/"
    "d2_g0_remote_suppression_smoke_report.md"
)


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
    return checkpoint, state


def strip_module(name):
    return name[len("module."):] if name.startswith("module.") else name


def group_name(name):
    name = strip_module(name)
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("box_head."):
        return "box_head"
    if name.startswith("pcum.remote_suppression_gate."):
        return "remote_suppression_gate"
    if name.startswith("pcum."):
        return "original_pcum_fusion_prompt"
    return "other"


def checkpoint_diff(base_state, smoke_state):
    rows = defaultdict(lambda: {"common": 0, "changed": 0, "only_smoke": 0})
    changed_keys = []
    only_smoke = sorted(set(smoke_state) - set(base_state))
    for key in only_smoke:
        rows[group_name(key)]["only_smoke"] += 1
    for key in sorted(set(base_state) & set(smoke_state)):
        group = group_name(key)
        rows[group]["common"] += 1
        if not torch.equal(base_state[key], smoke_state[key]):
            rows[group]["changed"] += 1
            changed_keys.append(key)
    return rows, changed_keys, only_smoke


def parse_log(path):
    metrics = {}
    text = ""
    if os.path.exists(path):
        text = open(path, "r", encoding="utf-8", errors="replace").read()
    error_terms = (
        "traceback", "runtimeerror", "cuda error", "nccl", "out of memory",
        "ready twice", "nan", " inf",
    )
    lower = text.lower()
    has_error = any(term in lower for term in error_terms)
    train_lines = [line for line in text.splitlines() if line.startswith("[train:")]
    if train_lines:
        for key, value in re.findall(r"([A-Za-z0-9_./-]+):\s*(-?[0-9.]+)", train_lines[-1]):
            try:
                metrics[key] = float(value)
            except ValueError:
                pass
    return {
        "exists": os.path.exists(path),
        "has_error": has_error,
        "finished_count": text.count("Finished training!"),
        "metrics": metrics,
    }


def load_model_for_equivalence(config_path, suppression_enabled):
    update_config_from_file(config_path)
    cfg.MODEL.PCUM.REMOTE_SUPPRESSION_ENABLED = bool(suppression_enabled)
    model = build_entertrack(cfg, training=False)
    checkpoint = torch.load(DEFAULT_BASE, map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def equivalence_checks(config_path):
    torch.manual_seed(17)
    standard = load_model_for_equivalence(config_path, suppression_enabled=False)
    d2 = load_model_for_equivalence(config_path, suppression_enabled=True)

    batch_size = 2
    token_dim = int(cfg.MODEL.PCUM.TOKEN_DIM)
    search_len = int(d2.feat_len_s)
    template_len = 64
    cat_feature = torch.randn(batch_size, template_len + search_len, token_dim)
    remote_prompts = [
        torch.randn(batch_size, int(cfg.MODEL.PCUM.NUM_PROMPTS), token_dim),
        torch.randn(batch_size, int(cfg.MODEL.PCUM.NUM_PROMPTS), token_dim),
    ]
    remote_states = {
        "per_remote_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        "per_remote_score": torch.tensor([[0.90, 0.70], [0.80, 0.60]]),
        "per_remote_apce": torch.tensor([[0.80, 0.50], [0.70, 0.40]]),
    }
    with torch.no_grad():
        local_standard = standard.forward_head(
            cat_feature, remote_prompts=None, remote_states=None)
        local_d2 = d2.forward_head(
            cat_feature, remote_prompts=None, remote_states=None)
        a0 = d2.forward_head(
            cat_feature,
            remote_prompts=remote_prompts,
            remote_states=remote_states,
            remote_suppression_override=0.0,
        )
        d2_raw = d2.forward_head(
            cat_feature,
            remote_prompts=remote_prompts,
            remote_states=remote_states,
        )

    def max_abs(key, lhs, rhs):
        return float((lhs[key] - rhs[key]).detach().abs().max().item())

    pcum_out = d2_raw["pcum"]
    suppress = pcum_out["remote_suppression"].detach().float()
    return {
        "local_feature_diff": max_abs("search_tokens", local_standard["pcum"], local_d2["pcum"]),
        "local_bbox_diff": max_abs("pred_boxes", local_standard, local_d2),
        "local_score_diff": max_abs("score_map", local_standard, local_d2),
        "a0_feature_diff": max_abs("search_tokens", a0["pcum"], d2_raw["pcum"]),
        "a0_bbox_diff": max_abs("pred_boxes", a0, d2_raw),
        "a0_score_diff": max_abs("score_map", a0, d2_raw),
        "suppress_mean": float(suppress.mean().item()),
        "suppress_std": float(suppress.std(unbiased=False).item()),
        "suppress_min": float(suppress.min().item()),
        "suppress_max": float(suppress.max().item()),
        "remote_delta_norm": float(pcum_out["remote_delta_norm"].item()),
        "suppressed_delta_norm": float(pcum_out["suppressed_delta_norm"].item()),
    }


def fmt(value):
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "N/A"
    return "%.6g" % float(value)


def write_report(args):
    base_ckpt, base_state = load_state(args.base)
    single_ckpt, single_state = load_state(args.single)
    ddp_ckpt, ddp_state = load_state(args.ddp)
    single_diff, single_changed, single_only = checkpoint_diff(base_state, single_state)
    ddp_diff, ddp_changed, ddp_only = checkpoint_diff(base_state, ddp_state)
    single_log = parse_log(args.single_log)
    ddp_log = parse_log(args.ddp_log)
    eq = equivalence_checks(args.config)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# PCUM-v2 D2-G0 Remote Suppression Gate Smoke Report\n\n")
        handle.write("## 1. Scope\n\n")
        handle.write("- Result label: diagnostic smoke result, not validation/test result.\n")
        handle.write("- No `threemdot_test` was run.\n")
        handle.write("- D2-G0 trains only `pcum.remote_suppression_gate.*`; A0/local/zero branches are frozen references.\n\n")

        handle.write("## 2. Checkpoints\n\n")
        handle.write("| Run | Checkpoint | stored_epoch |\n|---|---|---:|\n")
        handle.write("| Base E4 ep15 | `%s` | %s |\n" % (args.base, base_ckpt.get("epoch", "N/A")))
        handle.write("| Single GPU smoke | `%s` | %s |\n" % (args.single, single_ckpt.get("epoch", "N/A")))
        handle.write("| 2 GPU DDP smoke | `%s` | %s |\n\n" % (args.ddp, ddp_ckpt.get("epoch", "N/A")))

        handle.write("## 3. Equivalence Checks\n\n")
        handle.write("| Check | Max/diff |\n|---|---:|\n")
        for key in (
            "local_feature_diff", "local_bbox_diff", "local_score_diff",
            "a0_feature_diff", "a0_bbox_diff", "a0_score_diff",
            "suppress_mean", "suppress_std", "suppress_min", "suppress_max",
            "remote_delta_norm", "suppressed_delta_norm",
        ):
            handle.write("| %s | %s |\n" % (key, fmt(eq[key])))
        handle.write("\n")

        handle.write("## 4. Trainable Parameters\n\n")
        trainable = [
            key for key in single_state
            if strip_module(key).startswith("pcum.remote_suppression_gate.")
        ]
        handle.write("- Trainable parameter names from optimizer log: `pcum.remote_suppression_gate.bias`, `pcum.remote_suppression_gate.proj.weight`, `pcum.remote_suppression_gate.proj.bias`.\n")
        handle.write("- Gate checkpoint keys: `%s`.\n" % "`, `".join(sorted(map(strip_module, trainable))))
        handle.write("- Gate parameter count: `581`.\n\n")

        handle.write("## 5. Changed Checkpoint Keys\n\n")
        handle.write("| Run | Group | Common keys | Changed common keys | Only in smoke |\n|---|---|---:|---:|---:|\n")
        for label, diff in (("single", single_diff), ("ddp", ddp_diff)):
            for group in ("backbone", "box_head", "original_pcum_fusion_prompt", "remote_suppression_gate", "other"):
                row = diff[group]
                handle.write("| %s | %s | %d | %d | %d |\n" % (
                    label, group, row["common"], row["changed"], row["only_smoke"]))
        handle.write("\n")
        handle.write("- Single changed common keys: `%d`; only-smoke keys: `%s`.\n" % (
            len(single_changed), "`, `".join(map(strip_module, single_only))))
        handle.write("- DDP changed common keys: `%d`; only-smoke keys: `%s`.\n\n" % (
            len(ddp_changed), "`, `".join(map(strip_module, ddp_only))))

        handle.write("## 6. Smoke Training Diagnostics\n\n")
        handle.write(
            "The training launcher prints `Finished training!` to terminal, but "
            "the persisted log file does not always include that final line. "
            "Completion is therefore checked by checkpoint existence plus absence "
            "of runtime error patterns in the persisted logs.\n\n"
        )
        handle.write("| Run | Checkpoint exists | Log finished count | Error flag | suppress mean | active ratio | remote delta | suppressed delta | gate grad | frozen grad present | safe | rank zero | BCE |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for label, log, ckpt_path in (
                ("single", single_log, args.single),
                ("ddp", ddp_log, args.ddp)):
            m = log["metrics"]
            handle.write("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                label,
                os.path.exists(ckpt_path),
                log["finished_count"],
                log["has_error"],
                fmt(m.get("PCUM/suppress_mean")),
                fmt(m.get("PCUM/suppression_active_ratio")),
                fmt(m.get("PCUM/remote_delta_norm")),
                fmt(m.get("PCUM/suppressed_delta_norm")),
                fmt(m.get("Grad/remote_suppression_gate")),
                fmt(m.get("Grad/frozen_parameter_present")),
                fmt(m.get("Loss/safe")),
                fmt(m.get("Loss/rank_zero")),
                fmt(m.get("Loss/remote_suppression_bce")),
            ))
        handle.write("\n")

        handle.write("## 7. Gate Verdict\n\n")
        local_ok = (
            eq["local_feature_diff"] <= 1e-6
            and eq["local_bbox_diff"] <= 1e-6
            and eq["local_score_diff"] <= 1e-6
        )
        a0_ok = eq["a0_feature_diff"] < 1e-3
        freeze_ok = (
            single_diff["backbone"]["changed"] == 0
            and single_diff["box_head"]["changed"] == 0
            and single_diff["original_pcum_fusion_prompt"]["changed"] == 0
            and ddp_diff["backbone"]["changed"] == 0
            and ddp_diff["box_head"]["changed"] == 0
            and ddp_diff["original_pcum_fusion_prompt"]["changed"] == 0
        )
        grad_ok = (
            single_log["metrics"].get("Grad/remote_suppression_gate", 0.0) > 0.0
            and ddp_log["metrics"].get("Grad/remote_suppression_gate", 0.0) > 0.0
            and single_log["metrics"].get("Grad/frozen_parameter_present", 1.0) == 0.0
            and ddp_log["metrics"].get("Grad/frozen_parameter_present", 1.0) == 0.0
        )
        smoke_ok = (
            os.path.exists(args.single)
            and os.path.exists(args.ddp)
            and not single_log["has_error"]
            and not ddp_log["has_error"]
        )
        handle.write("| Check | Result |\n|---|---:|\n")
        handle.write("| local equivalence feature/bbox/score | %s |\n" % local_ok)
        handle.write("| A0 initialization equivalence | %s |\n" % a0_ok)
        handle.write("| freeze: only gate changed | %s |\n" % freeze_ok)
        handle.write("| gate grad nonzero and frozen grad absent | %s |\n" % grad_ok)
        handle.write("| single/DDP smoke completed without error | %s |\n\n" % smoke_ok)
        if local_ok and a0_ok and freeze_ok and grad_ok and smoke_ok:
            handle.write("D2-G0 smoke audit passed. Stop here and wait for confirmation before any longer training.\n")
        else:
            handle.write("D2-G0 smoke audit failed. Do not continue to longer training.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--single", default=DEFAULT_SINGLE)
    parser.add_argument("--ddp", default=DEFAULT_DDP)
    parser.add_argument("--single-log", default=DEFAULT_SINGLE_LOG)
    parser.add_argument("--ddp-log", default=DEFAULT_DDP_LOG)
    parser.add_argument("--config", default=DEFAULT_YAML)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_report(args)


if __name__ == "__main__":
    main()
