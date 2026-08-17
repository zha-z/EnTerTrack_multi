#!/usr/bin/env python3
"""Audit J0/J1 paired partial-adaptation configs without real sequence runs."""

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.models.entertrack.entertrack import build_entertrack  # noqa: E402
from lib.train.actors.entertrack_threemdot import EnTeRTrackActorThreeMDOT  # noqa: E402
from lib.train.optimizer_groups import build_optimizer_param_groups  # noqa: E402
from lib.train.pcum_freeze import apply_partial_adaptation_freeze  # noqa: E402
from lib.utils.box_ops import giou_loss  # noqa: E402
from lib.utils.focal_loss import FocalLoss  # noqa: E402


J0_CONFIG = "ostrack_deit_tiny_b0_j0_partial_adapt_ep15"
J1_CONFIG = "ostrack_deit_tiny_b0_j1_pcum_partial_adapt_ep15"
B0_SHA256 = "88706aa3087d245c22c152d3feb5417e20bd12f06942283cc0c513c53d2c6128"
OUT_DIR = ROOT / "output/controlled_baselines/pcum_joint_adaptation"


class Settings:
    batchsize = 1
    grad_clip_norm = 0.1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(name):
    resolved = copy.deepcopy(cfg)
    update_config_from_file(str(ROOT / "experiments/entertrack" / (name + ".yaml")),
                            base_cfg=resolved)
    return resolved


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("model", checkpoint))
    if state and all(name.startswith("module.") for name in state):
        state = {name[len("module."):]: value for name, value in state.items()}
    return checkpoint, state


def parameter_category(name):
    if name.startswith("pcum."):
        return "pcum"
    if name.startswith("backbone.blocks.4.") or name.startswith("backbone.blocks.5.") \
            or name.startswith("backbone.norm."):
        return "last_backbone"
    if name.startswith("backbone."):
        return "frozen_backbone"
    return "head"


def summarize_trainable(model):
    rows = []
    totals = defaultdict(int)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = parameter_category(name)
        count = int(parameter.numel())
        totals[category] += count
        rows.append({
            "name": name,
            "category": category,
            "shape": "x".join(str(v) for v in parameter.shape),
            "numel": count,
        })
    totals["total"] = sum(row["numel"] for row in rows)
    return rows, dict(totals)


def optimizer_summary(groups, model):
    id_to_name = {id(parameter): name for name, parameter in model.named_parameters()}
    rows = []
    for index, group in enumerate(groups):
        names = [id_to_name[id(parameter)] for parameter in group["params"]]
        rows.append({
            "group_index": index,
            "group_name": group.get("group_name", ""),
            "lr": float(group["lr"]),
            "tensor_count": len(names),
            "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
            "examples": names[:8],
            "names": names,
        })
    return rows


def build_for_audit(config_name):
    resolved = load_config(config_name)
    torch.manual_seed(42)
    model = build_entertrack(resolved, training=True)
    apply_partial_adaptation_freeze(model, resolved, verbose=False)
    groups = build_optimizer_param_groups(model, resolved, verbose=False)
    return resolved, model, groups


def synthetic_data(cfg):
    torch.manual_seed(20260715)
    batch = 1
    template = torch.randn(
        3, batch, 3, int(cfg.DATA.TEMPLATE.SIZE), int(cfg.DATA.TEMPLATE.SIZE))
    search = torch.randn(
        3, batch, 3, int(cfg.DATA.SEARCH.SIZE), int(cfg.DATA.SEARCH.SIZE))
    template_anno = torch.tensor(
        [[[0.35, 0.35, 0.22, 0.20]],
         [[0.38, 0.34, 0.20, 0.22]],
         [[0.33, 0.37, 0.24, 0.19]]],
        dtype=torch.float32,
    )
    search_anno = torch.tensor(
        [[[0.42, 0.44, 0.20, 0.18]],
         [[0.40, 0.43, 0.21, 0.19]],
         [[0.45, 0.41, 0.19, 0.20]]],
        dtype=torch.float32,
    )
    return {
        "template_images": template,
        "search_images": search,
        "template_anno": template_anno,
        "search_anno": search_anno,
        "template_view_valid": torch.ones(3, batch, dtype=torch.bool),
        "search_view_valid": torch.ones(3, batch, dtype=torch.bool),
        "epoch": 1,
    }


def clone_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def changed_names(before, after):
    return sorted(name for name in before if not torch.equal(before[name], after[name]))


def finite_grad_summary(model):
    grad_names = []
    bad = []
    total_norm_sq = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            continue
        grad_names.append(name)
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all().item():
            bad.append(name)
        total_norm_sq += float(grad.float().norm().item()) ** 2
    return {
        "grad_tensor_count": len(grad_names),
        "bad_grad_names": bad,
        "grad_norm": total_norm_sq ** 0.5,
    }


def run_one_step(label, cfg, model, groups):
    model.train()
    objective = {
        "giou": giou_loss,
        "l1": F.l1_loss,
        "focal": FocalLoss(),
    }
    loss_weight = {
        "giou": float(cfg.TRAIN.GIOU_WEIGHT),
        "l1": float(cfg.TRAIN.L1_WEIGHT),
        "focal": float(cfg.TRAIN.FOCAL_WEIGHT),
    }
    actor = EnTeRTrackActorThreeMDOT(model, objective, loss_weight, Settings(), cfg=cfg)
    optimizer = torch.optim.AdamW(groups, lr=float(cfg.TRAIN.LR),
                                  weight_decay=float(cfg.TRAIN.WEIGHT_DECAY))
    data = synthetic_data(cfg)
    before = clone_state(model)
    optimizer.zero_grad()
    if bool(cfg.TRAIN.PCUM.PAIRED_SUPERVISION):
        actor.begin_paired_iteration(data, diagnostics_active=False)
        local_loss, cache = actor.paired_local_stage(data)
        local_loss.backward()
        collaborative_loss, stats = actor.paired_collaborative_stage(data, cache)
        collaborative_loss.backward()
        loss_value = float((local_loss.detach() + collaborative_loss.detach()).item())
    else:
        loss, stats = actor(data)
        loss.backward()
        loss_value = float(loss.detach().item())
    grad_summary = finite_grad_summary(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.TRAIN.GRAD_CLIP_NORM))
    optimizer.step()
    after = clone_state(model)
    changed = changed_names(before, after)
    unexpected_backbone_changed = [
        name for name in changed
        if name.startswith("backbone.")
        and not (
            name.startswith("backbone.blocks.4.")
            or name.startswith("backbone.blocks.5.")
            or name.startswith("backbone.norm.")
        )
    ]
    return {
        "label": label,
        "loss": loss_value,
        "loss_finite": bool(torch.isfinite(torch.tensor(loss_value)).item()),
        "status": {key: float(value) for key, value in stats.items()
                   if isinstance(value, (int, float))},
        "grad_summary": grad_summary,
        "changed_names": changed,
        "changed_count": len(changed),
        "last_backbone_changed": any(parameter_category(name) == "last_backbone"
                                     for name in changed),
        "head_changed": any(parameter_category(name) == "head" for name in changed),
        "pcum_changed": any(parameter_category(name) == "pcum" for name in changed),
        "unexpected_backbone_changed": unexpected_backbone_changed,
    }


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key in sorted(value):
            child = "%s.%s" % (prefix, key) if prefix else str(key)
            out.update(flatten(value[key], child))
        return out
    return {prefix: value}


def config_diff(j0, j1):
    left = flatten(j0)
    right = flatten(j1)
    rows = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) == right.get(key):
            continue
        allowed = (
            key == "MODEL_ROLE"
            or key.startswith("MODEL.PCUM.")
            or key.startswith("TRAIN.PCUM.")
            or key.startswith("TEST.PCUM.")
            or key in {"TEST.SAVE_DIR", "TEST.CHECKPOINT_NAME"}
        )
        rows.append({
            "key": key,
            "j0": left.get(key, "<missing>"),
            "j1": right.get(key, "<missing>"),
            "allowed": allowed,
        })
    return rows


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-reports", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    j0_cfg, j0_model, j0_groups = build_for_audit(J0_CONFIG)
    j1_cfg, j1_model, j1_groups = build_for_audit(J1_CONFIG)

    block_names = sorted(set(
        ".".join(name.split(".")[:3])
        for name, _ in j0_model.named_parameters()
        if name.startswith("backbone.blocks.")
    ))
    final_norm_names = sorted(
        name for name, _ in j0_model.named_parameters()
        if name.startswith("backbone.norm.")
    )

    j0_trainable, j0_totals = summarize_trainable(j0_model)
    j1_trainable, j1_totals = summarize_trainable(j1_model)
    j0_group_summary = optimizer_summary(j0_groups, j0_model)
    j1_group_summary = optimizer_summary(j1_groups, j1_model)
    j0_smoke = run_one_step("J0", j0_cfg, j0_model, j0_groups)
    j1_smoke = run_one_step("J1", j1_cfg, j1_model, j1_groups)

    b0_path = Path(str(j0_cfg.B0_CHECKPOINT))
    b0_sha = sha256_file(b0_path)
    checkpoint, _ = checkpoint_state(b0_path)
    diff_rows = config_diff(j0_cfg, j1_cfg)

    method_diff_ok = all(row["allowed"] for row in diff_rows)
    optimizer_ok = (
        {row["group_name"] for row in j0_group_summary} == {"last_backbone", "head"}
        and {row["group_name"] for row in j1_group_summary} == {"last_backbone", "head", "pcum"}
    )
    smoke_ok = (
        j0_smoke["loss_finite"]
        and j1_smoke["loss_finite"]
        and not j0_smoke["grad_summary"]["bad_grad_names"]
        and not j1_smoke["grad_summary"]["bad_grad_names"]
        and j0_smoke["last_backbone_changed"]
        and j0_smoke["head_changed"]
        and not j0_smoke["pcum_changed"]
        and j1_smoke["last_backbone_changed"]
        and j1_smoke["head_changed"]
        and j1_smoke["pcum_changed"]
        and not j0_smoke["unexpected_backbone_changed"]
        and not j1_smoke["unexpected_backbone_changed"]
    )
    status = "PASS" if (
        b0_sha == B0_SHA256 and method_diff_ok and optimizer_ok and smoke_ok
    ) else "BLOCKED"

    summary = {
        "status": status,
        "j0_config": J0_CONFIG,
        "j1_config": J1_CONFIG,
        "b0_checkpoint": str(b0_path),
        "b0_sha256": b0_sha,
        "b0_epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
        "block_names": block_names,
        "final_norm_names": final_norm_names,
        "unfrozen_layers": ["backbone.blocks.4", "backbone.blocks.5", "backbone.norm", "box_head"],
        "j0_trainable_totals": j0_totals,
        "j1_trainable_totals": j1_totals,
        "j0_optimizer_groups": j0_group_summary,
        "j1_optimizer_groups": j1_group_summary,
        "j0_smoke": j0_smoke,
        "j1_smoke": j1_smoke,
        "config_diff": diff_rows,
        "method_diff_ok": method_diff_ok,
        "optimizer_ok": optimizer_ok,
        "smoke_ok": smoke_ok,
    }

    if args.write_reports:
        write_csv(
            OUT_DIR / "j0_trainable_parameters.csv",
            j0_trainable,
            ["name", "category", "shape", "numel"],
        )
        write_csv(
            OUT_DIR / "j1_trainable_parameters.csv",
            j1_trainable,
            ["name", "category", "shape", "numel"],
        )
        diff_table = "\n".join(
            "| {key} | `{j0}` | `{j1}` | `{allowed}` |".format(**row)
            for row in diff_rows
        )
        (OUT_DIR / "j0_j1_config_diff.md").write_text(
            "# J0/J1 config diff\n\n"
            "| Key | J0 | J1 | Allowed |\n"
            "|---|---|---|---|\n%s\n\n"
            "Allowed method differences are PCUM enablement, PCUM parameters, "
            "and PCUM supervision/safe loss. MODEL_ROLE and output checkpoint "
            "names are bookkeeping differences.\n" % diff_table
        )
        smoke_lines = [
            "# J0/J1 smoke report",
            "",
            "| Field | J0 | J1 |",
            "|---|---:|---:|",
            "| loss | %.6f | %.6f |" % (j0_smoke["loss"], j1_smoke["loss"]),
            "| grad_norm | %.6f | %.6f |" % (
                j0_smoke["grad_summary"]["grad_norm"],
                j1_smoke["grad_summary"]["grad_norm"],
            ),
            "| changed tensors | %d | %d |" % (
                j0_smoke["changed_count"], j1_smoke["changed_count"]),
            "| last_backbone_changed | `%s` | `%s` |" % (
                j0_smoke["last_backbone_changed"], j1_smoke["last_backbone_changed"]),
            "| head_changed | `%s` | `%s` |" % (
                j0_smoke["head_changed"], j1_smoke["head_changed"]),
            "| pcum_changed | `%s` | `%s` |" % (
                j0_smoke["pcum_changed"], j1_smoke["pcum_changed"]),
            "| unexpected_backbone_changed | `%s` | `%s` |" % (
                bool(j0_smoke["unexpected_backbone_changed"]),
                bool(j1_smoke["unexpected_backbone_changed"]),
            ),
            "",
            "Overall smoke status: `%s`." % ("PASS" if smoke_ok else "BLOCKED"),
        ]
        (OUT_DIR / "j0_j1_smoke_report.md").write_text("\n".join(smoke_lines) + "\n")
        protocol = """# J0/J1 paired partial-adaptation protocol

Experiment role: `post-hoc exploratory paired adaptation experiment`.

## Definitions

- J0: initialize from frozen B0 epoch25, train only `backbone.blocks.4`,
  `backbone.blocks.5`, `backbone.norm`, and `box_head`; PCUM disabled.
- J1: initialize from the same B0 epoch25, train the exact same backbone/head
  range plus fresh PCUM; PCUM supervision/safe loss enabled.

Both use seed 42, 15 epochs, AdamW, weight decay 0.0001, StepLR drop epoch 28,
template/search 128/256, no ARP/ATP/pruning/compensation/MCR, tracker remote
state, visible mask false at inference.

## Learning rates

- last_backbone: 2.4e-6
- head: 8e-6
- pcum: 8e-5, J1 only

## B0 initialization

- checkpoint: `%s`
- SHA256: `%s`
- stored epoch: `%s`
""" % (b0_path, b0_sha, summary["b0_epoch"])
        (OUT_DIR / "j0_j1_protocol.md").write_text(protocol)
        cv_plan = """# Target-group development protocol

- Do not use Three-MDOT test for J0/J1 selection.
- Primary development should use Three-MDOT train 23 targets with target-group
  cross-validation.
- A target group is `md####`; its three views must stay in the same fold.
- Suggested minimal protocol: 5 folds over target groups, fixed seed 42 for fold
  generation, train J0/J1 on the same folds and compare J1-J0 within each held
  out target group.
- Report target-cluster mean, bootstrap CI, and per-view deltas.
- Two-MDOT val may be used only as a two-view generalization diagnostic, not as
  a Three-MDOT three-view model-selection source.
- No new Three-MDOT test run unless a new formal frozen protocol or independent
  held-out data is approved.
"""
        (OUT_DIR / "target_group_cv_plan.md").write_text(cv_plan)
        commands = """# Manual run commands for J0/J1

These commands are intentionally not executed by this audit.

## Re-run lightweight audit only

```bash
cd /data/zjy/EnTeR-Track-main
/home/user/.conda/envs/zjy/bin/python tracking/audit_j0_j1_partial_adaptation.py --write-reports
```

## Future full training, only after approval

Run J0 and J1 as a paired experiment with the same seed and hardware budget.
Do not run validation/test as part of this configuration audit.
"""
        (OUT_DIR / "manual_run_commands.md").write_text(commands)
        registry = """# J0/J1 experiment registry

| Field | Value |
|---|---|
| status | `%s` |
| experiment_role | `posthoc_exploratory_paired_adaptation` |
| J0 config | `%s` |
| J1 config | `%s` |
| initialization | `B0 epoch25` |
| seed | `42` |
| total_epoch | `15` |
| test_usage | `forbidden_without_new_freeze_or_heldout_protocol` |
| no_temperature_tuning | `true` |
| no_mcr_or_threshold_tuning | `true` |
""" % (status, J0_CONFIG, J1_CONFIG)
        (OUT_DIR / "experiment_registry.md").write_text(registry)

    (OUT_DIR / "j0_j1_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
