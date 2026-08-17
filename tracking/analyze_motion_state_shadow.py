#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import yaml

from lib.test.tracker.motion_state import MotionState, summarize_motion_records


SIGNALS = (
    "max_score",
    "apce",
    "response_entropy",
    "response_top1_top2_gap",
    "response_peak_sharpness",
    "normalized_motion_residual",
    "bbox_border_proximity",
    "search_region_border_proximity",
    "remote_quality",
    "remote_weight_entropy",
    "remote_max_weight",
    "valid_remote_count",
)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze prediction-only M0 shadow logs without running a tracker."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="output/motion_state_shadow_analysis",
    )
    return parser.parse_args()


def find_logs(input_dir):
    root = os.path.abspath(input_dir)
    candidates = []
    for current_root, _, files in os.walk(root):
        if os.path.basename(current_root) != "motion_state_diagnostics":
            continue
        candidates.extend(
            os.path.join(current_root, name)
            for name in files
            if name.endswith(".jsonl")
        )
    if os.path.basename(root) == "motion_state_diagnostics":
        candidates.extend(
            os.path.join(root, name)
            for name in os.listdir(root)
            if name.endswith(".jsonl")
        )
    return sorted(set(candidates))


def load_records(path):
    records = []
    with open(path) as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "state" not in record or "frame_id" not in record:
                raise ValueError("invalid M0 record at {}:{}".format(path, line_number))
            records.append(record)
    return records


def finite_values(records, field):
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def state_run_lengths(records):
    runs = []
    last = None
    length = 0
    for record in records:
        state = record.get("state", "UNKNOWN")
        if state == last:
            length += 1
        else:
            if last is not None:
                runs.append((last, length))
            last = state
            length = 1
    if last is not None:
        runs.append((last, length))
    return runs


def write_statistics(path, all_records):
    rows = []
    for signal in SIGNALS:
        values = finite_values(all_records, signal)
        row = {
            "signal": signal,
            "available_count": int(values.size),
            "missing_count": len(all_records) - int(values.size),
            "available_ratio": float(values.size / max(len(all_records), 1)),
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "nearly_constant": bool(values.size and values.std() < 1e-8),
        }
        for quantile in QUANTILES:
            name = "q{:02d}".format(int(round(quantile * 100)))
            row[name] = float(np.quantile(values, quantile)) if values.size else None
        rows.append(row)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_sequence_summary(path, sequence_records):
    rows = []
    for sequence, records in sorted(sequence_records.items()):
        summary = summarize_motion_records(records)
        runs = state_run_lengths(records)
        transitions = summary["state_transition_count"]
        rows.append({
            "sequence": sequence,
            "frame_count": summary["frame_count"],
            "normal_frames": summary["state_counts"][MotionState.NORMAL.value],
            "uncertain_frames": summary["state_counts"][MotionState.UNCERTAIN.value],
            "lost_frames": summary["state_counts"][MotionState.LOST.value],
            "recover_frames": summary["state_counts"][MotionState.RECOVER.value],
            "state_transitions": transitions,
            "oscillation_rate": transitions / max(len(records) - 1, 1),
            "longest_lost_duration": summary["longest_lost_duration"],
            "mean_state_run_length": float(np.mean([length for _, length in runs]))
            if runs else 0.0,
            "mean_score": summary["mean_score"],
            "mean_apce": summary["mean_apce"],
            "mean_motion_residual": summary["mean_normalized_motion_residual"],
            "bbox_border_events": summary["bbox_border_event_count"],
            "search_border_events": summary["search_region_border_event_count"],
        })
    with open(path, "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def proposed_thresholds(all_records):
    def q(field, quantile, fallback=0.0):
        values = finite_values(all_records, field)
        return float(np.quantile(values, quantile)) if values.size else fallback

    return {
        "provenance": "validation-derived prediction-only M0 shadow diagnostics",
        "warning": "provisional; select on threemdot_val only and never on test",
        "TEST": {
            "MOTION_STATE": {
                "SCORE_LOW": q("max_score", 0.10),
                "SCORE_RECOVER": q("max_score", 0.50),
                "APCE_LOW": q("apce", 0.10),
                "APCE_RECOVER": q("apce", 0.50),
                "MOTION_RESIDUAL_HIGH": q("normalized_motion_residual", 0.90),
            }
        },
    }


def main():
    args = parse_args()
    logs = find_logs(args.input_dir)
    if not logs:
        raise FileNotFoundError("no motion_state_diagnostics JSONL files found")
    sequence_records = {
        os.path.splitext(os.path.basename(path))[0]: load_records(path)
        for path in logs
    }
    all_records = [
        record for records in sequence_records.values() for record in records
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    statistics = write_statistics(
        os.path.join(args.output_dir, "motion_signal_statistics.csv"),
        all_records,
    )
    sequence_rows = write_sequence_summary(
        os.path.join(args.output_dir, "sequence_state_summary.csv"),
        sequence_records,
    )
    thresholds = proposed_thresholds(all_records)
    with open(os.path.join(args.output_dir, "proposed_val_thresholds.yaml"), "w") as fh:
        yaml.safe_dump(thresholds, fh, sort_keys=False)

    trigger_counts = Counter()
    transition_counts = Counter()
    for record in all_records:
        trigger_counts.update(record.get("low_quality_reasons", []))
        reason = str(record.get("transition_reason", ""))
        if "_to_" in reason:
            transition_counts[reason.split(":", 1)[0]] += 1
    unavailable = [
        row["signal"] for row in statistics if row["available_count"] == 0
    ]
    constant = [
        row["signal"] for row in statistics if row["nearly_constant"]
    ]
    max_oscillation = max(row["oscillation_rate"] for row in sequence_rows)
    summary_path = os.path.join(args.output_dir, "motion_signal_summary.md")
    with open(summary_path, "w") as fh:
        fh.write("# M0 Motion State Shadow Signal Summary\n\n")
        fh.write("- Result label: validation-derived diagnostic only.\n")
        fh.write("- GT read by this script: no.\n")
        fh.write("- Sequences: `{}`; frames: `{}`.\n".format(
            len(sequence_records), len(all_records)))
        fh.write("- Maximum sequence oscillation rate: `{:.4f}`.\n".format(
            max_oscillation))
        fh.write("- Unavailable signals: `{}`.\n".format(
            ", ".join(unavailable) if unavailable else "none"))
        fh.write("- Nearly constant signals: `{}`.\n\n".format(
            ", ".join(constant) if constant else "none"))
        fh.write("## Trigger Frequencies\n\n")
        for name, count in trigger_counts.most_common():
            fh.write("- `{}`: {}\n".format(name, count))
        fh.write("\n## State Transitions\n\n")
        for name, count in transition_counts.most_common():
            fh.write("- `{}`: {}\n".format(name, count))
        fh.write("\n## Threshold Boundary\n\n")
        fh.write(
            "`proposed_val_thresholds.yaml` is derived only from these logs. "
            "It is provisional and may be selected only on `threemdot_val`; it "
            "must not be derived from or tuned on `threemdot_test`.\n"
        )
    print(summary_path)


if __name__ == "__main__":
    main()
