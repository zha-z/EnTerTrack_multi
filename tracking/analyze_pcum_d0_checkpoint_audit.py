#!/usr/bin/env python3
"""Audit D0 ranking fine-tuning checkpoint/path attribution."""

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


DEFAULT_BASE = (
    "output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/train/"
    "entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/"
    "EnTeRTrack_ep0015.pth.tar"
)
DEFAULT_D0 = (
    "output/pcum_v2_d_ranking/checkpoints/train/entertrack/"
    "pcum_v2_d0_ranking_softmax_t010_ep5/EnTeRTrack_ep0005.pth.tar"
)
DEFAULT_TRAIN_YAML = (
    "experiments/entertrack/pcum_v2_d0_ranking_softmax_t010_ep5.yaml"
)
DEFAULT_T1_YAML = (
    "experiments/entertrack/pcum_v2_d0_ranking_softmax_t010_ep5_t1_val.yaml"
)
DEFAULT_LOG = (
    "output/pcum_v2_d_ranking/logs/"
    "entertrack-pcum_v2_d0_ranking_softmax_t010_ep5.log"
)
DEFAULT_OUT = "output/pcum_v2_d_ranking/d0_checkpoint_and_path_audit.md"


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict is not a dict: %s" % path)
    return checkpoint, state


def strip_module(name):
    return name[len("module."):] if name.startswith("module.") else name


def module_group(name):
    n = strip_module(name)
    if n.startswith("backbone."):
        return "backbone"
    if "transformer" in n:
        return "transformer"
    if n.startswith(("neck.", "input_proj.", "featurefusion_network.")):
        return "neck"
    if n.startswith("box_head."):
        return "box_head"
    if n.startswith(("cls.", "head.", "search_prompt_gate.")) or ".cls" in n:
        return "cls/head"
    if n.startswith("pcum."):
        if "fusion" in n:
            return "fusion"
        if "prompt" in n or "encoder" in n or "selector" in n:
            return "prompt"
        if "residual_scale" in n:
            return "residual_scale"
        return "PCUM"
    if "fusion" in n:
        return "fusion"
    if "prompt" in n:
        return "prompt"
    if "residual_scale" in n:
        return "residual_scale"
    return "other"


def train_group(name):
    n = strip_module(name)
    if n.startswith("backbone."):
        return "backbone"
    if n.startswith("pcum."):
        return "PCUM"
    if n.startswith("box_head."):
        return "box_head"
    return "head_and_other"


def is_buffer_key(name):
    canonical = strip_module(name)
    return canonical.endswith((
        ".running_mean",
        ".running_var",
        ".num_batches_tracked",
    ))


def tensor_stats(base_tensor, d0_tensor):
    if base_tensor.shape != d0_tensor.shape:
        return {
            "shape_mismatch": True,
            "numel": 0,
            "same": False,
            "l2": math.nan,
            "max_abs": math.nan,
            "rel": math.nan,
        }
    base = base_tensor.detach().float()
    d0 = d0_tensor.detach().float()
    diff = d0 - base
    l2 = float(torch.linalg.vector_norm(diff).item())
    base_l2 = float(torch.linalg.vector_norm(base).item())
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    same = bool(torch.equal(base_tensor, d0_tensor))
    rel = l2 / (base_l2 + 1e-12)
    return {
        "shape_mismatch": False,
        "numel": int(diff.numel()),
        "same": same,
        "l2": l2,
        "max_abs": max_abs,
        "rel": rel,
    }


def parse_simple_yaml_flags(path):
    flags = {}
    if not os.path.exists(path):
        return flags
    text = open(path, "r", encoding="utf-8").read()
    patterns = {
        "FREEZE_BACKBONE": r"^\s*FREEZE_BACKBONE:\s*(\S+)",
        "FREEZE_HEAD": r"^\s*FREEZE_HEAD:\s*(\S+)",
        "MODEL_PCUM_ENABLED": r"^\s*ENABLED:\s*(\S+)",
        "TRAIN_RANKING_ENABLED": r"^\s*RANKING_ENABLED:\s*(\S+)",
        "DELAY_BRANCH_MODE": r"^\s*DELAY_BRANCH_MODE:\s*(\S+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            flags[key] = match.group(1)
    return flags


def parse_test_pcum_block(path):
    values = {}
    if not os.path.exists(path):
        return values
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    in_test = False
    in_pcum = False
    for line in lines:
        if re.match(r"^TEST:\s*$", line):
            in_test = True
            in_pcum = False
            continue
        if in_test and re.match(r"^[A-Z][A-Z0-9_]*:\s*$", line):
            in_test = False
            in_pcum = False
        if not in_test:
            continue
        if re.match(r"^\s{2}PCUM:\s*$", line):
            in_pcum = True
            continue
        if in_pcum:
            match = re.match(r"^\s{4}([A-Z0-9_]+):\s*(.+?)\s*$", line)
            if match:
                values[match.group(1)] = match.group(2)
            elif re.match(r"^\s{2}[A-Z0-9_]+:", line):
                in_pcum = False
    return values


def read_log_summary(path):
    summary = {
        "exists": os.path.exists(path),
        "freeze_backbone_line": False,
        "freeze_head_line": False,
        "optimizer_group_lines": [],
        "nan_inf_errors": False,
        "residual_scale_last": None,
        "effective_residual_scale_last": None,
        "rank_zero_last": None,
        "rank_delay_last": None,
        "rank_local_last": None,
        "grad_pcum_encoder_last": None,
        "grad_pcum_aligner_last": None,
        "grad_pcum_fusion_film_last": None,
        "grad_residual_scale_last": None,
    }
    if not summary["exists"]:
        return summary
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "FREEZE_BACKBONE enabled" in line:
                summary["freeze_backbone_line"] = True
            if "FREEZE_HEAD enabled" in line:
                summary["freeze_head_line"] = True
            if "Optimizer parameter groups" in line or re.search(
                    r"\s(head_and_other|backbone|pcum): tensors=", line):
                summary["optimizer_group_lines"].append(line.strip())
            lower = line.lower()
            if any(token in lower for token in (
                "traceback", "runtimeerror", "cuda error", "out of memory",
                " nan", " inf",
            )):
                summary["nan_inf_errors"] = True
            for key, report_key in (
                ("PCUM/raw_residual_scale:", "residual_scale_last"),
                ("PCUM/effective_residual_scale:", "effective_residual_scale_last"),
                ("Loss/rank_zero:", "rank_zero_last"),
                ("Loss/rank_delay:", "rank_delay_last"),
                ("Loss/rank_local:", "rank_local_last"),
                ("Grad/pcum_encoder:", "grad_pcum_encoder_last"),
                ("Grad/pcum_aligner:", "grad_pcum_aligner_last"),
                ("Grad/pcum_fusion_film:", "grad_pcum_fusion_film_last"),
                ("Grad/residual_scale:", "grad_residual_scale_last"),
            ):
                if key in line:
                    match = re.search(re.escape(key) + r"\s*([-+0-9.eE]+)", line)
                    if match:
                        summary[report_key] = float(match.group(1))
    return summary


def infer_trainable(base_state, freeze_backbone, freeze_head):
    rows = []
    counts = defaultdict(lambda: {"keys": 0, "params": 0})
    frozen_counts = defaultdict(lambda: {"keys": 0, "params": 0})
    buffer_counts = defaultdict(lambda: {"keys": 0, "params": 0})
    for name, tensor in base_state.items():
        canonical = strip_module(name)
        group = train_group(canonical)
        numel = int(tensor.numel())
        if is_buffer_key(canonical):
            buffer_counts[group]["keys"] += 1
            buffer_counts[group]["params"] += numel
            continue
        requires_grad = True
        if freeze_backbone and canonical.startswith("backbone."):
            requires_grad = False
        if freeze_head and not canonical.startswith(("backbone.", "pcum.")):
            requires_grad = False
        if requires_grad:
            counts[group]["keys"] += 1
            counts[group]["params"] += numel
            rows.append((canonical, group, numel))
        else:
            frozen_counts[group]["keys"] += 1
            frozen_counts[group]["params"] += numel
    return rows, counts, frozen_counts, buffer_counts


def fmt_float(value):
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return "%.6g" % float(value)


def write_report(args):
    base_ckpt, base_state = load_checkpoint(args.base)
    d0_ckpt, d0_state = load_checkpoint(args.d0)

    base_keys = set(base_state.keys())
    d0_keys = set(d0_state.keys())
    common = sorted(base_keys & d0_keys)
    only_base = sorted(base_keys - d0_keys)
    only_d0 = sorted(d0_keys - base_keys)

    per_key = []
    same_count = 0
    changed_count = 0
    mismatch_count = 0
    groups = defaultdict(lambda: {
        "keys": 0,
        "changed_keys": 0,
        "params": 0,
        "l2_sq": 0.0,
        "base_l2_sq": 0.0,
        "max_abs": 0.0,
    })
    for key in common:
        stats = tensor_stats(base_state[key], d0_state[key])
        group = module_group(key)
        base_l2 = float(torch.linalg.vector_norm(
            base_state[key].detach().float()).item()) if not stats["shape_mismatch"] else 0.0
        groups[group]["keys"] += 1
        groups[group]["params"] += int(base_state[key].numel())
        if stats["shape_mismatch"]:
            mismatch_count += 1
            groups[group]["changed_keys"] += 1
        elif stats["same"]:
            same_count += 1
        else:
            changed_count += 1
            groups[group]["changed_keys"] += 1
            groups[group]["l2_sq"] += stats["l2"] ** 2
            groups[group]["base_l2_sq"] += base_l2 ** 2
            groups[group]["max_abs"] = max(groups[group]["max_abs"], stats["max_abs"])
        per_key.append((key, group, stats))

    train_flags = parse_simple_yaml_flags(args.train_yaml)
    test_flags = parse_test_pcum_block(args.t1_yaml)
    freeze_backbone = str(train_flags.get("FREEZE_BACKBONE", "false")).lower() == "true"
    freeze_head = str(train_flags.get("FREEZE_HEAD", "false")).lower() == "true"
    trainable_rows, trainable_counts, frozen_counts, buffer_counts = infer_trainable(
        d0_state, freeze_backbone=freeze_backbone, freeze_head=freeze_head)

    log_summary = read_log_summary(args.train_log)

    top_changed = sorted(
        [row for row in per_key if not row[2]["shape_mismatch"] and not row[2]["same"]],
        key=lambda row: (row[2]["l2"], row[2]["max_abs"]),
        reverse=True,
    )[:50]

    changed_groups = {group for _, group, stats in per_key
                      if stats["shape_mismatch"] or not stats["same"]}
    changed_backbone = "backbone" in changed_groups
    changed_head = any(group in changed_groups for group in (
        "box_head", "cls/head", "neck", "transformer", "other"))
    only_pcum_like = changed_groups.issubset({"PCUM", "fusion", "prompt", "residual_scale"})

    box_head_trainable = [
        name for name, group, _ in trainable_rows if group == "box_head"
    ]
    backbone_trainable = [
        name for name, group, _ in trainable_rows if group == "backbone"
    ]
    transformer_trainable = [
        name for name, _, _ in trainable_rows if "transformer" in name
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# D0 Checkpoint and Path Attribution Audit\n\n")
        f.write("## 1. Checkpoint Metadata\n\n")
        f.write("| Item | Base E4 ep15 | D0 ep5 |\n")
        f.write("|---|---:|---:|\n")
        f.write("| stored_epoch | %s | %s |\n" % (
            base_ckpt.get("epoch", "N/A"), d0_ckpt.get("epoch", "N/A")))
        f.write("| state_dict keys | %d | %d |\n" % (len(base_state), len(d0_state)))
        f.write("| common keys | %d | %d |\n" % (len(common), len(common)))
        f.write("| only in checkpoint | %d | %d |\n\n" % (len(only_base), len(only_d0)))
        f.write("- 完全相同参数 tensor 数：`%d`\n" % same_count)
        f.write("- 发生数值变化 tensor 数：`%d`\n" % changed_count)
        f.write("- shape mismatch tensor 数：`%d`\n\n" % mismatch_count)

        f.write("## 2. Module-wise Parameter Diff\n\n")
        f.write("| Group | Keys | Changed keys | Params | L2 diff | Max abs diff | Relative diff |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        order = [
            "backbone", "transformer", "neck", "box_head", "cls/head",
            "PCUM", "fusion", "prompt", "residual_scale", "other"
        ]
        for group in order:
            item = groups[group]
            l2 = math.sqrt(item["l2_sq"])
            base_l2 = math.sqrt(item["base_l2_sq"])
            rel = l2 / (base_l2 + 1e-12)
            f.write("| %s | %d | %d | %d | %s | %s | %s |\n" % (
                group,
                item["keys"],
                item["changed_keys"],
                item["params"],
                fmt_float(l2),
                fmt_float(item["max_abs"]),
                fmt_float(rel),
            ))
        f.write("\n")

        f.write("## 3. Top 50 Changed Parameters\n\n")
        f.write("| Rank | Parameter | Group | Numel | L2 diff | Max abs diff | Relative diff |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|\n")
        for idx, (key, group, stats) in enumerate(top_changed, 1):
            f.write("| %d | `%s` | %s | %d | %s | %s | %s |\n" % (
                idx, key, group, stats["numel"], fmt_float(stats["l2"]),
                fmt_float(stats["max_abs"]), fmt_float(stats["rel"])))
        f.write("\n")

        f.write("## 4. Freeze and Optimizer Audit\n\n")
        f.write("| Item | Value |\n|---|---|\n")
        f.write("| TRAIN.FREEZE_BACKBONE | `%s` |\n" % train_flags.get("FREEZE_BACKBONE", "N/A"))
        f.write("| TRAIN.FREEZE_HEAD | `%s` |\n" % train_flags.get("FREEZE_HEAD", "N/A"))
        f.write("| TRAIN.PCUM.RANKING_ENABLED | `%s` |\n" % train_flags.get("TRAIN_RANKING_ENABLED", "N/A"))
        f.write("| TRAIN.PCUM.DELAY_BRANCH_MODE | `%s` |\n" % train_flags.get("DELAY_BRANCH_MODE", "N/A"))
        f.write("| training log has FREEZE_BACKBONE line | `%s` |\n" % log_summary["freeze_backbone_line"])
        f.write("| training log has FREEZE_HEAD line | `%s` |\n" % log_summary["freeze_head_line"])
        f.write("| training log NaN/Inf/Error flag | `%s` |\n" % log_summary["nan_inf_errors"])
        f.write("| last raw residual scale | `%s` |\n" % fmt_float(log_summary["residual_scale_last"]))
        f.write("| last effective residual scale | `%s` |\n" % fmt_float(log_summary["effective_residual_scale_last"]))
        f.write("\n")
        f.write(
            "Reconstructed optimizer parameters from D0 YAML freeze rules. "
            "BatchNorm running statistics and `num_batches_tracked` are listed "
            "as buffers, not optimizer parameters.\n\n"
        )
        f.write("| Optimizer group | Trainable parameter keys | Trainable params | Frozen parameter keys | Frozen params | Buffer keys |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for group in ("backbone", "box_head", "head_and_other", "PCUM"):
            f.write("| %s | %d | %d | %d | %d | %d |\n" % (
                group,
                trainable_counts[group]["keys"],
                trainable_counts[group]["params"],
                frozen_counts[group]["keys"],
                frozen_counts[group]["params"],
                buffer_counts[group]["keys"],
            ))
        f.write("\n")
        f.write("- box_head 参数进入 optimizer：`%s`，数量 `%d`\n" % (
            bool(box_head_trainable), len(box_head_trainable)))
        f.write("- backbone 参数进入 optimizer：`%s`，数量 `%d`\n" % (
            bool(backbone_trainable), len(backbone_trainable)))
        f.write("- transformer 命名参数进入 optimizer：`%s`，数量 `%d`\n\n" % (
            bool(transformer_trainable), len(transformer_trainable)))
        if box_head_trainable:
            f.write("box_head trainable key examples:\n\n")
            for name in box_head_trainable[:30]:
                f.write("- `%s`\n" % name)
            f.write("\n")
        if backbone_trainable:
            f.write("backbone trainable key examples:\n\n")
            for name in backbone_trainable[:30]:
                f.write("- `%s`\n" % name)
            f.write("\n")

        f.write("## 5. Smoke Training Diagnostics\n\n")
        f.write("| Item | Value |\n|---|---:|\n")
        f.write("| Loss/rank_zero last | `%s` |\n" % fmt_float(log_summary["rank_zero_last"]))
        f.write("| Loss/rank_delay last | `%s` |\n" % fmt_float(log_summary["rank_delay_last"]))
        f.write("| Loss/rank_local last | `%s` |\n" % fmt_float(log_summary["rank_local_last"]))
        f.write("| Grad/pcum_encoder last | `%s` |\n" % fmt_float(log_summary["grad_pcum_encoder_last"]))
        f.write("| Grad/pcum_aligner last | `%s` |\n" % fmt_float(log_summary["grad_pcum_aligner_last"]))
        f.write("| Grad/pcum_fusion_film last | `%s` |\n" % fmt_float(log_summary["grad_pcum_fusion_film_last"]))
        f.write("| Grad/residual_scale last | `%s` |\n" % fmt_float(log_summary["grad_residual_scale_last"]))
        f.write("| NaN/Inf/Error found | `%s` |\n\n" % log_summary["nan_inf_errors"])
        ranking_nonzero = any(
            value is not None and abs(value) > 0.0
            for value in (
                log_summary["rank_zero_last"],
                log_summary["rank_delay_last"],
                log_summary["rank_local_last"],
            )
        )
        grad_nonzero = any(
            value is not None and abs(value) > 0.0
            for value in (
                log_summary["grad_pcum_encoder_last"],
                log_summary["grad_pcum_aligner_last"],
                log_summary["grad_pcum_fusion_film_last"],
                log_summary["grad_residual_scale_last"],
            )
        )
        f.write("- ranking loss 非全 0：`%s`\n" % ranking_nonzero)
        f.write("- PCUM/fusion/residual gradient 非全 0：`%s`\n\n" % grad_nonzero)

        f.write("## 6. Local-only Path Audit\n\n")
        f.write("| T1 validation config item | Value |\n|---|---|\n")
        for key in (
            "USE_REMOTE", "USE_REMOTE_VISIBLE_MASK", "REMOTE_STATE_SOURCE",
            "REMOTE_ABLATION", "REMOTE_ABLATION_OFFSET",
        ):
            f.write("| TEST.PCUM.%s | `%s` |\n" % (key, test_flags.get(key, "N/A")))
        f.write("\n")
        f.write(
            "Code path: `lib/models/entertrack/entertrack.py::forward_head` calls "
            "`self.pcum(...)` whenever `MODEL.PCUM.ENABLED=True`, regardless of "
            "whether `remote_prompts` is `None`. Therefore D0 T1 with "
            "`TEST.PCUM.USE_REMOTE=false` is not a pure non-PCUM model; it is a "
            "local PCUM prompt/fusion path without remote prompts. The observed "
            "0/15 remote weight diagnostics are expected because no remote prompt "
            "aggregation is executed.\n\n"
        )
        f.write(
            "Attribution: D0 T1 can improve over E4 T1 because D0 changed "
            "parameters on the local PCUM/fusion path, and the T1 path still "
            "uses those modules. It does not require any remote prompt to benefit.\n\n"
        )

        f.write("## 7. Delay Branch Consistency\n\n")
        f.write(
            "- Training delay branch: `TRAIN.PCUM.DELAY_BRANCH_MODE=batch_roll`, "
            "implemented as batch-roll prompt negative inside the actor.\n"
            "- Test delay branch: existing `REMOTE_ABLATION=temporal_shuffle`, "
            "a causal temporal delay queue across frames.\n"
            "- These are not strictly identical. D0 raw > delay is useful as a "
            "negative-prompt sanity check, but it is weaker evidence for temporal "
            "synchronization than a training branch that uses the same causal "
            "temporal delay as test.\n"
            "- A D0-fixed follow-up should align train delay with causal temporal "
            "delay if the delay conclusion is central.\n\n"
        )

        f.write("## 8. Attribution Verdict\n\n")
        f.write("- Backbone parameters changed: `%s`\n" % changed_backbone)
        f.write("- Head / non-PCUM parameters changed: `%s`\n" % changed_head)
        f.write("- Only PCUM/fusion/prompt/residual changed: `%s`\n\n" % only_pcum_like)
        if changed_head or changed_backbone:
            f.write(
                "**Verdict A:** D0 is not a strict PCUM-only fine-tune. Current "
                "D0 validation should not be used as a clean PCUM ranking-loss "
                "causal conclusion. The immediate fix is to rerun D0-fixed with "
                "`FREEZE_BACKBONE=true` and `FREEZE_HEAD=true`, then re-check that "
                "only PCUM/fusion/prompt/residual parameters changed before "
                "validation.\n\n"
            )
        elif only_pcum_like:
            f.write(
                "**Verdict B/C:** Only PCUM-like parameters changed. Because T1 "
                "still uses PCUM/fusion when `MODEL.PCUM.ENABLED=true`, local-only "
                "is not completely bypassing those parameters; either redefine T1 "
                "or add a strict bypass path if pure local attribution is required.\n\n"
            )
        else:
            f.write(
                "Changed parameters do not fit a clean PCUM-only grouping; inspect "
                "the Top 50 table and optimizer grouping before further runs.\n\n"
            )

        if only_base or only_d0:
            f.write("## 9. Non-overlapping Keys\n\n")
            f.write("- Only base keys: `%d`\n" % len(only_base))
            f.write("- Only D0 keys: `%d`\n" % len(only_d0))
            for title, keys in (("Only base", only_base[:20]), ("Only D0", only_d0[:20])):
                if keys:
                    f.write("\n%s examples:\n\n" % title)
                    for key in keys:
                        f.write("- `%s`\n" % key)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--d0", default=DEFAULT_D0)
    parser.add_argument("--train-yaml", default=DEFAULT_TRAIN_YAML)
    parser.add_argument("--t1-yaml", default=DEFAULT_T1_YAML)
    parser.add_argument("--train-log", default=DEFAULT_LOG)
    parser.add_argument("--output", default=DEFAULT_OUT)
    args = parser.parse_args()
    write_report(args)
    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
