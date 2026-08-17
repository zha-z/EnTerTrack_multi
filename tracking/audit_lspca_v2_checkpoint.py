#!/usr/bin/env python3
"""Read-only integrity and update-boundary audit for LSPCA-v2 checkpoints."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config.entertrack.config import get_default_config, update_config_from_file  # noqa: E402
from lib.models.entertrack.entertrack import build_entertrack  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(name):
    resolved = get_default_config()
    update_config_from_file(
        str(ROOT / "experiments/entertrack" / (name + ".yaml")),
        base_cfg=resolved,
    )
    return resolved


def strip_module_prefix(state):
    if state and all(name.startswith("module.") for name in state):
        return {name[len("module."):]: value for name, value in state.items()}
    return state


def category(name):
    if name.startswith("pcum."):
        return "pcum"
    if name.startswith((
        "backbone.blocks.4.",
        "backbone.blocks.5.",
        "backbone.norm.",
    )):
        return "last_backbone"
    if name.startswith("backbone."):
        return "frozen_backbone"
    return "head"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("j0", "j1"))
    parser.add_argument("--expected-epoch", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(int(cfg.TRAIN.SEED))
    initialized = build_entertrack(cfg, training=True)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in initialized.state_dict().items()
    }

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    final_state = strip_module_prefix(checkpoint.get("net", checkpoint))
    strict_error = None
    try:
        initialized.load_state_dict(final_state, strict=True)
    except RuntimeError as error:
        strict_error = str(error)

    initial_names = set(initial_state)
    final_names = set(final_state)
    comparable_names = sorted(initial_names & final_names)
    changed = [
        name for name in comparable_names
        if not torch.equal(initial_state[name], final_state[name].detach().cpu())
    ]
    changed_categories = Counter(category(name) for name in changed)
    unexpected = [
        name for name in changed
        if category(name) == "frozen_backbone"
        or (args.role == "j0" and category(name) == "pcum")
    ]
    nonfinite = [
        name for name, value in final_state.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all().item()
    ]
    optimizer_groups = checkpoint.get("optimizer", {}).get("param_groups", [])
    optimizer_group_lrs = [float(group["lr"]) for group in optimizer_groups]
    expected_group_count = 2 if args.role == "j0" else 3
    initialization_audit = getattr(initialized, "initialization_audit", {})

    checks = {
        "checkpoint_exists_nonempty": args.checkpoint.is_file()
        and args.checkpoint.stat().st_size > 0,
        "epoch_matches": checkpoint.get("epoch") == args.expected_epoch,
        "strict_final_load": strict_error is None,
        "state_key_sets_match": initial_names == final_names,
        "all_tensors_finite": not nonfinite,
        "no_disallowed_updates": not unexpected,
        "last_backbone_updated": changed_categories["last_backbone"] > 0,
        "head_updated": changed_categories["head"] > 0,
        "pcum_boundary_matches_role": (
            changed_categories["pcum"] == 0
            if args.role == "j0"
            else changed_categories["pcum"] > 0
        ),
        "optimizer_group_count_matches": len(optimizer_groups) == expected_group_count,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "role": args.role,
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "epoch": checkpoint.get("epoch"),
        "state_key_count": len(final_state),
        "optimizer_group_count": len(optimizer_groups),
        "optimizer_group_lrs": optimizer_group_lrs,
        "initialization_audit": initialization_audit,
        "changed_key_count": len(changed),
        "changed_category_counts": dict(sorted(changed_categories.items())),
        "changed_key_examples": changed[:40],
        "unexpected_changed_keys": unexpected,
        "nonfinite_keys": nonfinite,
        "missing_from_final": sorted(initial_names - final_names),
        "unexpected_in_final": sorted(final_names - initial_names),
        "strict_load_error": strict_error,
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
