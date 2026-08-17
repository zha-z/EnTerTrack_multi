#!/usr/bin/env python3
"""Target-level paired analysis for one completed LSPCA-v2 CV fold."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.analyze_j0_j1_group_cv import METRICS, bootstrap, evaluate_target, target_id


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--j0-results", required=True, type=Path)
    parser.add_argument("--j1-results", required=True, type=Path)
    parser.add_argument("--j0-runid", required=True, type=int)
    parser.add_argument("--j1-runid", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    sequences = [
        line.strip() for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    grouped = {}
    for sequence in sequences:
        grouped.setdefault(target_id(sequence), []).append(sequence)
    malformed = {
        target: names for target, names in grouped.items()
        if sorted(name.rsplit("-", 1)[-1] for name in names) != ["1", "2", "3"]
    }
    if malformed:
        raise RuntimeError("Malformed target/view groups: {}".format(malformed))

    target_rows = []
    sequence_rows = []
    for target, names in grouped.items():
        names = sorted(names, key=lambda name: int(name.rsplit("-", 1)[-1]))
        j0 = evaluate_target(args.j0_results, target, names)
        j1 = evaluate_target(args.j1_results, target, names)
        row = {"fold_id": args.fold, "target": target}
        for metric in METRICS:
            row["j0_" + metric] = j0[metric]
            row["j1_" + metric] = j1[metric]
            row["delta_" + metric] = j1[metric] - j0[metric]
        target_rows.append(row)
        for left, right in zip(j0["sequence_rows"], j1["sequence_rows"]):
            sequence_row = {
                "fold_id": args.fold,
                "target": target,
                "sequence": left["sequence"],
                "view": left["view"],
            }
            for metric in METRICS:
                sequence_row["j0_" + metric] = left[metric]
                sequence_row["j1_" + metric] = right[metric]
                sequence_row["delta_" + metric] = right[metric] - left[metric]
            sequence_rows.append(sequence_row)

    summary = {
        "status": "DESCRIPTIVE_FOLD_ONLY",
        "fold_id": args.fold,
        "target_count": len(target_rows),
        "sequence_count": len(sequence_rows),
        "j0_runid": args.j0_runid,
        "j1_runid": args.j1_runid,
        "metrics": {},
        "view_deltas": {},
    }
    for metric in METRICS:
        deltas = np.asarray(
            [row["delta_" + metric] for row in target_rows], dtype=np.float64)
        stats = bootstrap(deltas, seed=20260715, samples=100000)
        positive = deltas[deltas > 0]
        stats["positive_gain_concentration_max"] = (
            float(positive.max() / positive.sum()) if positive.size and positive.sum() > 0
            else None
        )
        stats["j0_mean"] = float(np.mean([row["j0_" + metric] for row in target_rows]))
        stats["j1_mean"] = float(np.mean([row["j1_" + metric] for row in target_rows]))
        summary["metrics"][metric] = stats
        summary["view_deltas"][metric] = {
            view: float(np.mean([
                row["delta_" + metric] for row in sequence_rows if row["view"] == view
            ]))
            for view in ("1", "2", "3")
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "target_metrics.csv", target_rows)
    write_csv(args.output_dir / "sequence_view_metrics.csv", sequence_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
