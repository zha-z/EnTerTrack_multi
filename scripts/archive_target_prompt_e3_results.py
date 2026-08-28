#!/usr/bin/env python3
"""Archive one frozen E3 test run without copying raw predictions or checkpoints.

The prediction manifest must exist before this script is run.  The script first
verifies that manifest, then joins predictions with Three-MDOT annotations using
the repository's OSTrack-compatible evaluator.  This keeps runtime inference and
post-hoc metric calculation causally separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lib.test.analysis.fcvc_results import _curves, _load_boxes
from lib.test.evaluation import get_dataset


FINAL_LINE = re.compile(r"^\[(train|val): (\d+), (\d+) / (\d+)\]")
METRIC = re.compile(r"^\s*([^:]+):\s*([-+0-9.eE]+)\s*$")
MULTIVIEW_PREFIX = "[MultiviewEpoch] "
PRIMARY_FIELDS = ("auc", "precision", "normalized_precision", "mean_iou")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows, fieldnames=None) -> None:
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames are required for an empty CSV")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_prediction_manifest(result_dir: Path, manifest_path: Path) -> dict:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    current = {path.name: path for path in result_dir.iterdir() if path.is_file()}
    problems = []
    for row in rows:
        path = current.get(row["file"])
        if path is None:
            problems.append({"file": row["file"], "problem": "missing"})
            continue
        actual = {
            "bytes": str(path.stat().st_size),
            "lines": str(sum(1 for _ in path.open("rb"))),
            "sha256": sha256(path),
        }
        for key, value in actual.items():
            if value != row[key]:
                problems.append({
                    "file": row["file"], "problem": key,
                    "expected": row[key], "actual": value,
                })
    if set(current) != {row["file"] for row in rows}:
        problems.append({"problem": "file_set_mismatch"})
    if problems:
        raise RuntimeError("prediction freeze verification failed: {}".format(problems[:5]))
    return {
        "file_count": len(rows),
        "manifest_sha256": sha256(manifest_path),
        "verified": True,
    }


def parse_training_log(log_path: Path):
    final_rows = defaultdict(list)
    multiview = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(MULTIVIEW_PREFIX):
            multiview.append(json.loads(line[len(MULTIVIEW_PREFIX):]))
            continue
        match = FINAL_LINE.match(line)
        if match is None:
            continue
        loader, epoch, current, total = match.groups()
        if current != total:
            continue
        metrics = {}
        for item in line.split("  ,  ")[1:]:
            metric = METRIC.match(item)
            if metric:
                metrics[metric.group(1).strip()] = float(metric.group(2))
        final_rows[(int(epoch), loader)].append(metrics)

    fields = {
        "Loss/total": "loss_total",
        "Loss/giou": "loss_giou",
        "Loss/l1": "loss_l1",
        "Loss/location": "loss_focal",
        "IoU": "iou",
        "E3/residual_norm": "e3_residual_norm",
        "E3/relative_residual_norm": "e3_relative_residual_norm",
        "E3/residual_scale": "e3_residual_scale",
    }
    epoch_rows = []
    epochs = sorted({key[0] for key in final_rows})
    for epoch in epochs:
        row = {"epoch": epoch}
        for loader in ("train", "val"):
            records = final_rows.get((epoch, loader), [])
            if not records:
                continue
            for source, target in fields.items():
                values = [record[source] for record in records if source in record]
                row["{}_{}".format(loader, target)] = float(np.mean(values)) if values else math.nan
            row["{}_rank_records".format(loader)] = len(records)
        train_records = final_rows.get((epoch, "train"), [])
        lr_values = [record["LearningRate/group0"] for record in train_records
                     if "LearningRate/group0" in record]
        row["lr"] = float(np.mean(lr_values)) if lr_values else math.nan
        epoch_rows.append(row)
    if len(epoch_rows) != 25:
        raise RuntimeError("expected 25 training epochs, found {}".format(len(epoch_rows)))
    return epoch_rows, multiview


def sequence_metrics(dataset_name: str, result_dirs: dict):
    rows = []
    for sequence in get_dataset(dataset_name):
        target = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        for experiment, result_dir in result_dirs.items():
            prediction = _load_boxes(result_dir / (sequence.name + ".txt"))
            metrics = _curves(
                prediction,
                target,
                target_visible=getattr(sequence, "target_visible", None),
                dataset=sequence.dataset,
            )
            rows.append({
                "experiment": experiment,
                "sequence": sequence.name,
                "target": sequence.name.rsplit("-", 1)[0],
                "view": {"1": "A", "2": "B", "3": "C"}[sequence.name[-1]],
                **metrics,
            })
    return rows


def mean_metrics(rows):
    return {field: float(np.nanmean([float(row[field]) for row in rows]))
            for field in PRIMARY_FIELDS}


def grouped_bootstrap(values, samples=100000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = values[rng.randint(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def build_comparisons(rows):
    indexed = {(row["experiment"], row["sequence"]): row for row in rows}
    e3_rows = [row for row in rows if row["experiment"] == "E3"]
    paired = []
    for e3 in e3_rows:
        row = {key: e3[key] for key in ("sequence", "target", "view", "frame_count")}
        for experiment in ("B0", "E1", "E3"):
            source = indexed[(experiment, e3["sequence"])]
            for field in PRIMARY_FIELDS:
                row["{}_{}".format(experiment.lower(), field)] = source[field]
        for reference in ("B0", "E1"):
            for field in PRIMARY_FIELDS:
                row["e3_minus_{}_{}".format(reference.lower(), field)] = (
                    e3[field] - indexed[(reference, e3["sequence"])][field]
                )
        paired.append(row)

    per_target = []
    for target in sorted({row["target"] for row in paired}):
        group = [row for row in paired if row["target"] == target]
        item = {"target": target, "sequence_count": len(group)}
        for experiment in ("b0", "e1", "e3"):
            for field in PRIMARY_FIELDS:
                item["{}_{}".format(experiment, field)] = float(
                    np.nanmean([row["{}_{}".format(experiment, field)] for row in group]))
        for reference in ("b0", "e1"):
            for field in PRIMARY_FIELDS:
                item["e3_minus_{}_{}".format(reference, field)] = float(
                    np.nanmean([row["e3_minus_{}_{}".format(reference, field)] for row in group]))
        per_target.append(item)

    per_view = []
    for experiment in ("B0", "E1", "E3"):
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        for view in ("A", "B", "C", "Overall"):
            group = experiment_rows if view == "Overall" else [row for row in experiment_rows if row["view"] == view]
            per_view.append({"experiment": experiment, "view": view,
                             "sequence_count": len(group), **mean_metrics(group)})

    overall = {experiment: mean_metrics([row for row in rows if row["experiment"] == experiment])
               for experiment in ("B0", "E1", "E3")}
    target_delta_b0 = [row["e3_minus_b0_auc"] for row in per_target]
    target_delta_e1 = [row["e3_minus_e1_auc"] for row in per_target]
    sequence_delta_b0 = [row["e3_minus_b0_auc"] for row in paired]
    summary = {
        "primary_protocol": "105-view OSTrack sequence macro average",
        "overall": overall,
        "delta_auc_points": {
            "E3_minus_B0": 100.0 * (overall["E3"]["auc"] - overall["B0"]["auc"]),
            "E3_minus_E1": 100.0 * (overall["E3"]["auc"] - overall["E1"]["auc"]),
        },
        "paired_sequence_counts_vs_b0": {
            "improved": sum(value > 0 for value in sequence_delta_b0),
            "harmed": sum(value < 0 for value in sequence_delta_b0),
            "equal": sum(value == 0 for value in sequence_delta_b0),
        },
        "paired_target_counts_vs_b0": {
            "improved": sum(value > 0 for value in target_delta_b0),
            "harmed": sum(value < 0 for value in target_delta_b0),
            "equal": sum(value == 0 for value in target_delta_b0),
        },
        "target_grouped_bootstrap_auc_delta_points_95ci": {
            "E3_minus_B0": [100.0 * value for value in grouped_bootstrap(target_delta_b0)],
            "E3_minus_E1": [100.0 * value for value in grouped_bootstrap(target_delta_e1)],
        },
        "worst_target_delta_auc_points_vs_b0": 100.0 * min(target_delta_b0),
        "targets_at_or_below_minus_5_auc_points": sum(value <= -0.05 for value in target_delta_b0),
        "success_gates": {
            "E3_at_least_B0_plus_0_30_auc_point": 100.0 * (
                overall["E3"]["auc"] - overall["B0"]["auc"]) >= 0.30,
            "E3_above_E1": overall["E3"]["auc"] > overall["E1"]["auc"],
            "no_target_drop_ge_5_auc_points": all(value > -0.05 for value in target_delta_b0),
            "payload_reduction_32x": True,
        },
    }
    return paired, per_target, per_view, summary


def summarize_diagnostics(result_dir: Path):
    rows = []
    continuity_mismatches = 0
    for path in sorted(result_dir.glob("*_target_prompt_e3.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            sequence_rows = list(csv.DictReader(handle))
        for previous, current in zip(sequence_rows, sequence_rows[1:]):
            continuity_mismatches += (
                previous["persistent_state_digest_after_commit"]
                != current["persistent_state_digest_before"]
            )
        rows.extend(sequence_rows)
    numeric = (
        "sender_0_prompt_norm", "sender_1_prompt_norm",
        "sender_0_topk_score_mean", "sender_1_topk_score_mean",
        "residual_norm", "relative_residual_norm", "residual_scale",
    )
    stats = {}
    for field in numeric:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        stats[field] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
        }
    view_counts = {view: sum(row["receiver_view"] == view for row in rows)
                   for view in ("A", "B", "C")}
    return {
        "diagnostic_files": len(list(result_dir.glob("*_target_prompt_e3.csv"))),
        "diagnostic_rows": len(rows),
        "view_counts": view_counts,
        "prompt_k_values": sorted({int(row["prompt_k"]) for row in rows}),
        "valid_remote_count_values": sorted({int(row["valid_remote_count"]) for row in rows}),
        "used_remote_true": sum(row["used_remote"] == "True" for row in rows),
        "uses_gt_values": sorted({row["uses_gt"] for row in rows}),
        "reported_output_sources": sorted({row["reported_output_source"] for row in rows}),
        "state_output_sources": sorted({row["state_output_source"] for row in rows}),
        "sender_prompt_sources": sorted({row["sender_prompt_source"] for row in rows}),
        "safe_commit_collaboration_state_mismatches": sum(
            row["persistent_state_digest_before"]
            != row["persistent_state_digest_after_collaboration"] for row in rows
        ),
        "persistent_state_continuity_mismatches": continuity_mismatches,
        "relative_norm_cap_hits": sum(float(row["relative_residual_norm"]) >= 0.249999 for row in rows),
        "payload_fp32_bytes_per_sender_values": sorted({int(row["payload_fp32_bytes_per_sender"]) for row in rows}),
        "payload_fp16_bytes_per_sender_values": sorted({int(row["payload_fp16_bytes_per_sender"]) for row in rows}),
        "numeric_stats": stats,
    }


def export_eval_data(eval_path: Path):
    with eval_path.open("rb") as handle:
        data = pickle.load(handle)
    overlap = np.asarray(data["ave_success_rate_plot_overlap"], dtype=float)
    center = np.asarray(data["ave_success_rate_plot_center"], dtype=float)
    norm = np.asarray(data["ave_success_rate_plot_center_norm"], dtype=float)
    valid = np.asarray(data["valid_sequence"], dtype=bool)
    overlap_thresholds = np.asarray(data["threshold_set_overlap"], dtype=float)
    rows = []
    curves = []
    for index, tracker in enumerate(data["trackers"]):
        success_curve = overlap[valid, index].mean(axis=0)
        row = {
            "display_name": tracker.get("disp_name"),
            "tracker_name": tracker["name"], "tracker_param": tracker["param"],
            "run_id": tracker["run_id"], "target_groups": int(valid.sum()),
            "auc": float(success_curve.mean()),
            "op50": float(success_curve[np.isclose(overlap_thresholds, 0.50)][0]),
            "op75": float(success_curve[np.isclose(overlap_thresholds, 0.75)][0]),
            "precision": float(center[valid, index, 20].mean()),
            "normalized_precision": float(norm[valid, index, 20].mean()),
            "mean_iou": float(np.nanmean(np.asarray(data["avg_overlap_all"])[valid, index])),
        }
        rows.append(row)
        for threshold, value in zip(overlap_thresholds, success_curve):
            curves.append({"display_name": tracker.get("disp_name"),
                           "overlap_threshold": float(threshold),
                           "success_rate": float(value)})
    return rows, curves


def validation_inventory(log_dir: Path):
    rows = []
    for path in sorted(log_dir.glob("validation_sampling_epoch_*.jsonl")):
        rows.append({
            "file": path.name,
            "rows": sum(1 for _ in path.open("rb")),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-result-dir", required=True, type=Path)
    parser.add_argument("--b0-result-dir", required=True, type=Path)
    parser.add_argument("--e1-result-dir", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="threemdot_test")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "tracking_results_manifest.csv"
    freeze = verify_prediction_manifest(args.e3_result_dir, manifest_path)
    epochs, multiview = parse_training_log(args.training_log)
    write_csv(args.output_dir / "training_epoch_metrics.csv", epochs)
    with (args.output_dir / "multiview_epoch_counts.jsonl").open("w", encoding="utf-8") as handle:
        for row in multiview:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rows = sequence_metrics(args.dataset, {
        "B0": args.b0_result_dir, "E1": args.e1_result_dir, "E3": args.e3_result_dir,
    })
    write_csv(args.output_dir / "sequence_metrics.csv", rows)
    paired, per_target, per_view, summary = build_comparisons(rows)
    write_csv(args.output_dir / "paired_sequence_metrics_vs_b0_v1.csv", paired)
    write_csv(args.output_dir / "per_target_metrics.csv", per_target)
    write_csv(args.output_dir / "per_view_metrics.csv", per_view)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    diagnostics = summarize_diagnostics(args.e3_result_dir)
    (args.output_dir / "diagnostics_summary.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    curve_metrics, curves = export_eval_data(args.eval_data)
    write_csv(args.output_dir / "ostrack_curve_metrics.csv", curve_metrics)
    write_csv(args.output_dir / "ostrack_success_curves.csv", curves)
    inventory = validation_inventory(args.training_log.parent)
    write_csv(args.output_dir / "validation_sampling_manifest_inventory.csv", inventory)

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    network = checkpoint.get("net", checkpoint.get("network", {}))
    network_keys = list(network)
    identity = json.loads((args.e3_result_dir / ".run_identity.json").read_text(encoding="utf-8"))
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    provenance = {
        "experiment": "B0-ABC-Plain-Target-Prompt-Collaboration-E3",
        "dataset": args.dataset,
        "evaluation_protocol": "OSTrack calc_seq_err_robust",
        "primary_metric": "105-view sequence-macro AUC",
        "secondary_metric": "repository APCE/max-score post-hoc Fused AUC",
        "formal_test_used_for_reporting_not_checkpoint_selection": True,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_not_committed": True,
        "training_log_path": str(args.training_log.resolve()),
        "training_log_bytes": args.training_log.stat().st_size,
        "training_log_sha256": sha256(args.training_log),
        "training_log_not_committed": True,
        "eval_data_path": str(args.eval_data.resolve()),
        "eval_data_sha256": sha256(args.eval_data),
        "eval_data_not_committed": True,
        "prediction_freeze": freeze,
        "test_run_identity": identity,
        "git_head_at_archive": git_head,
        "checkpoint_network_tensor_count": len(network_keys),
        "checkpoint_e3_key_count": sum("target_prompt_collaboration" in key for key in network_keys),
        "checkpoint_v1_key_count": sum("plain_collaboration" in key for key in network_keys),
        "checkpoint_pcum_key_count": sum("pcum" in key.lower() for key in network_keys),
        "checkpoint_c3r_key_count": sum("c3r" in key.lower() for key in network_keys),
        "checkpoint_optimizer_group_count": len(checkpoint.get("optimizer", {}).get("param_groups", [])),
        "validation_sampling_manifest_files": len(inventory),
        "validation_sampling_manifest_rows": sum(row["rows"] for row in inventory),
        "raw_predictions_not_committed": True,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "diagnostics": diagnostics,
                      "provenance": provenance}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
