#!/usr/bin/env python3
"""Static validation audit that never constructs a tracker or test dataset."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.train.data.fcvc_sampler import FCVCSampler
from lib.train.fcvc_config import load_resolved_config
from lib.train.fcvc_validation_reporting import is_better_online
from lib.train.fcvc_validation_sampler import FixedPairValidationSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = load_resolved_config(
        ROOT / "experiments/entertrack/fcvc_full.yaml")
    split_dir = ROOT / "output/train/entertrack/fcvc_full/splits"
    validation_dir = ROOT / "output/train/entertrack/fcvc_full/validation"
    split = json.loads((split_dir / "target_split.json").read_text(encoding="utf-8"))
    split_sha = (split_dir / "target_split_sha256.txt").read_text(
        encoding="utf-8").strip()
    pair_sampler = FixedPairValidationSampler(
        validation_dir / "val_pair_manifest.csv",
        split["validation_targets"])
    partitions = []
    combined = []
    for rank in range(6):
        local = FixedPairValidationSampler(
            validation_dir / "val_pair_manifest.csv",
            split["validation_targets"])
        rows = local.partition(rank)
        combined.extend(int(row["case_index"]) for row in rows)
        partitions.append({
            "rank": rank, "cases": len(rows), "groups": len(rows) // 3,
            "first_case": int(rows[0]["case_index"]),
            "last_case": int(rows[-1]["case_index"]),
        })
    source = (ROOT / resolved["DATA"]["TRAIN"]["MANIFEST"]).resolve()
    train_sampler = FCVCSampler(
        source, allowed_targets=split["train_targets"])
    train_rows, train_contract = train_sampler.generate_epoch(1)
    best_simulation = {
        "higher_auc_wins": is_better_online(
            {"epoch": 10, "auc_collab": .61, "auc_delta": 0,
             "harmful_rate": .3},
            {"epoch": 5, "auc_collab": .60, "auc_delta": .1,
             "harmful_rate": .1}),
        "higher_delta_breaks_tie": is_better_online(
            {"epoch": 10, "auc_collab": .60, "auc_delta": .02,
             "harmful_rate": .3},
            {"epoch": 5, "auc_collab": .60, "auc_delta": .01,
             "harmful_rate": .1}),
        "lower_harmful_breaks_tie": is_better_online(
            {"epoch": 10, "auc_collab": .60, "auc_delta": .01,
             "harmful_rate": .1},
            {"epoch": 5, "auc_collab": .60, "auc_delta": .01,
             "harmful_rate": .2}),
        "earlier_epoch_breaks_full_tie": not is_better_online(
            {"epoch": 10, "auc_collab": .60, "auc_delta": .01,
             "harmful_rate": .2},
            {"epoch": 5, "auc_collab": .60, "auc_delta": .01,
             "harmful_rate": .2}),
    }
    report = {
        "split_sha256": split_sha,
        "train_targets": split["train_targets"],
        "validation_targets": split["validation_targets"],
        "target_intersection": sorted(
            set(split["train_targets"]) & set(split["validation_targets"])),
        "split_statistics": {
            "train": split["train"], "validation": split["validation"]},
        "training_epoch_contract": train_contract,
        "training_target_leakage_cases": sum(
            row["target"] in set(split["validation_targets"])
            for row in train_rows),
        "pair_contract": pair_sampler.audit(),
        "pair_partitions": partitions,
        "pair_exact_unique_coverage": combined == list(range(1512)),
        "best_selection_simulation": best_simulation,
        "online_epochs": [5, 10, 15, 20, 25, 30],
        "threemdot_test_accessed": False,
        "formal_training_executed": False,
        "formal_test_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
