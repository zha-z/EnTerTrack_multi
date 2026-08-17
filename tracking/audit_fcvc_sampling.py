#!/usr/bin/env python3
"""Audit-only FCVC synchronized sampling and six-rank coverage report."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.train.data.fcvc_sampler import FCVCSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "output/multi_agent_collaboration_clean/fcvc_manual_run/full_train_receiver_manifest.csv")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sampler = FCVCSampler(args.manifest)
    rows, contract = sampler.generate_epoch(args.epoch)
    repeated_rows, repeated_contract = sampler.generate_epoch(args.epoch)
    groups = defaultdict(list)
    for row in rows:
        groups[row["sync_group_id"]].append(row)
    partitions = []
    local_counts = []
    for rank in range(6):
        local = FCVCSampler(args.manifest)
        local_contract = local.begin_epoch(args.epoch, rank=rank, world_size=6)
        partitions.extend(local.rows)
        local_counts.append({
            "rank": rank,
            "cases": len(local.rows),
            "groups": local_contract["local_group_count"],
            "receivers": dict(sorted(Counter(
                row["receiver"] for row in local.rows).items())),
        })
    report = {
        **contract,
        "deterministic_replay": (
            rows == repeated_rows and contract == repeated_contract),
        "abc_sync_valid": all(
            len(group) == 3
            and {row["receiver"] for row in group} == {"A", "B", "C"}
            and len({row["template_frame"] for row in group}) == 1
            and len({row["search_frame"] for row in group}) == 1
            for group in groups.values()),
        "max_observed_interval": max(
            abs(row["search_frame"] - row["template_frame"]) for row in rows),
        "target_balance_range": (
            max(contract["target_group_counts"].values())
            - min(contract["target_group_counts"].values())),
        "student_input_gt_rows": sum(
            bool(row["uses_gt_in_student_input"]) for row in rows),
        "partition_exact_order_coverage": partitions == rows,
        "partition_unique_case_count": len({
            row["case_index"] for row in partitions}),
        "per_rank": local_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
