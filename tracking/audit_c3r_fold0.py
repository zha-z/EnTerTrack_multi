#!/usr/bin/env python3
"""Audit and evaluate one formal C3R CV fold without running trackers."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.test.evaluation.run_id import read_run_identity, result_directory


REQUIRED_REGISTRY_FIELDS = (
    "experiment_id", "model_role", "fold_id", "config", "dataset",
    "split_manifest", "checkpoint", "checkpoint_sha256", "runid",
    "output_dir", "tracker_log", "evaluation_log", "expected_targets",
    "expected_sequences", "message_mode", "status",
)
METRICS = ("auc", "precision", "norm_precision")
def read_registry(path, fold_id=0):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_REGISTRY_FIELDS) - set(reader.fieldnames or ()))
        if missing:
            raise RuntimeError("evaluation registry missing fields: {}".format(missing))
        rows = [row for row in reader if row["fold_id"] == str(fold_id)]
    ids = [row["experiment_id"] for row in rows]
    expected = ["C0_F{}".format(fold_id), "C1_F{}".format(fold_id),
                "E0_F{}".format(fold_id)]
    if sorted(ids) != expected:
        raise RuntimeError(
            "registry must contain exactly {}".format(", ".join(expected)))
    return rows


def read_split(path, expected_targets, expected_sequences):
    with Path(path).open("r", encoding="utf-8") as handle:
        sequences = [line.strip() for line in handle if line.strip()]
    if len(sequences) != expected_sequences or len(set(sequences)) != expected_sequences:
        raise RuntimeError("holdout sequence count/uniqueness mismatch")
    groups = defaultdict(set)
    for sequence in sequences:
        target, view = sequence.rsplit("-", 1)
        groups[target].add(int(view))
    if len(groups) != expected_targets or any(
            views != {1, 2, 3} for views in groups.values()):
        raise RuntimeError("holdout target count or view triplets mismatch")
    return sequences


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path, columns=None):
    try:
        value = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    except ValueError:
        value = np.loadtxt(str(path), dtype=np.float64)
    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value.reshape(1, -1) if columns else value.reshape(-1, 1)
    if columns is not None:
        value = value[:, :columns]
    return value


def sequence_metrics(pred, gt):
    pred = np.asarray(pred, dtype=np.float64).copy()
    gt = np.asarray(gt, dtype=np.float64)
    pred[0] = gt[0]
    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    pred_center = pred[:, :2] + 0.5 * (pred[:, 2:] - 1.0)
    gt_center = gt[:, :2] + 0.5 * (gt[:, 2:] - 1.0)
    center_error = np.sqrt(((pred_center - gt_center) ** 2).sum(axis=1))
    normalized_error = np.sqrt(
        (((pred_center / gt[:, 2:]) - (gt_center / gt[:, 2:])) ** 2).sum(axis=1))
    top_left = np.maximum(pred[:, :2], gt[:, :2])
    bottom_right = np.minimum(
        pred[:, :2] + pred[:, 2:] - 1.0,
        gt[:, :2] + gt[:, 2:] - 1.0,
    )
    size = np.maximum(bottom_right - top_left + 1.0, 0.0)
    intersection = size[:, 0] * size[:, 1]
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - intersection
    overlap = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0)
    center_error[~valid] = np.inf
    normalized_error[~valid] = -1.0
    overlap[~valid] = -1.0
    length = len(gt)
    overlap_thresholds = np.arange(0.0, 1.0 + 0.05, 0.05)
    center_thresholds = np.arange(0, 51)
    normalized_thresholds = center_thresholds / 100.0
    return {
        "auc": float(((overlap[:, None] > overlap_thresholds).sum(0) / length).mean() * 100.0),
        "precision": float(((center_error[:, None] <= center_thresholds).sum(0) / length)[20] * 100.0),
        "norm_precision": float(((normalized_error[:, None] <= normalized_thresholds).sum(0) / length)[20] * 100.0),
    }


def _bool_text(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def validate_registry_entry(row, registry_path):
    if row["dataset"] != "threemdot_cv":
        raise RuntimeError("registry entry is outside formal C3R CV")
    split = (Path(registry_path).parent.parent.parent / row["split_manifest"]).resolve()
    if not split.is_file():
        split = Path(row["split_manifest"]).resolve()
    expected_targets = int(row["expected_targets"])
    expected_sequences = int(row["expected_sequences"])
    if expected_sequences != expected_targets * 3:
        raise RuntimeError("registry expected sequence count is not three views per target")
    sequences = read_split(split, expected_targets, expected_sequences)
    checkpoint = row["checkpoint"].strip()
    checkpoint_sha = row["checkpoint_sha256"].strip()
    if not checkpoint or not checkpoint_sha:
        raise RuntimeError(
            "{} checkpoint remains pending".format(row["experiment_id"]))
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("registry checkpoint SHA256 mismatch")
    result_path = Path(row["output_dir"])
    expected_path = result_directory(
        result_path.parents[1], "entertrack", row["config"], row["runid"])
    if result_path.resolve() != expected_path.resolve():
        raise RuntimeError("registry output_dir disagrees with runid contract")
    identity = read_run_identity(result_path)
    expected_identity = {
        "parameter_name": row["config"],
        "dataset_name": row["dataset"],
        "runid": row["runid"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "no_gt_inference": True,
    }
    for key, value in expected_identity.items():
        if identity.get(key) != value:
            raise RuntimeError("result identity mismatch for {}".format(key))
    return split, sequences, result_path


def inspect_experiment(row, registry_path, gt_root):
    split, sequences, result_path = validate_registry_entry(row, registry_path)
    gt_root = Path(gt_root)
    sequence_rows = []
    diagnostics_rows = []
    communication_rows = []
    for sequence in sequences:
        target = sequence.rsplit("-", 1)[0]
        gt_path = gt_root / target / sequence / "groundtruth.txt"
        bbox_path = result_path / (sequence + ".txt")
        score_path = result_path / (sequence + "_max_score.txt")
        apce_path = result_path / (sequence + "_APCE.txt")
        for required in (gt_path, bbox_path, score_path, apce_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        gt = load_matrix(gt_path, columns=4)
        pred = load_matrix(bbox_path, columns=4)
        score = load_matrix(score_path).reshape(-1)
        apce = load_matrix(apce_path).reshape(-1)
        if not (len(pred) == len(score) == len(apce) == len(gt)):
            raise RuntimeError("prediction/diagnostic length mismatch for {}".format(sequence))
        if not all(np.isfinite(value).all() for value in (pred, score, apce)):
            raise RuntimeError("non-finite tracker output for {}".format(sequence))
        row_metrics = {
            "experiment_id": row["experiment_id"],
            "target": target,
            "sequence": sequence,
            "view": int(sequence.rsplit("-", 1)[1]),
        }
        row_metrics.update(sequence_metrics(pred, gt))
        sequence_rows.append(row_metrics)

        diag_path = result_path / (sequence + "_c3r_diagnostics.csv")
        summary_path = result_path / (sequence + "_c3r_comm_summary.json")
        if row["message_mode"] == "none":
            if diag_path.exists() or summary_path.exists():
                raise RuntimeError("E0 unexpectedly produced C3R diagnostics")
        else:
            if not diag_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(diag_path if not diag_path.is_file() else summary_path)
            with diag_path.open(newline="", encoding="utf-8") as handle:
                diag = list(csv.DictReader(handle))
            if len(diag) != len(gt):
                raise RuntimeError("C3R diagnostic length mismatch for {}".format(sequence))
            for item in diag:
                if _bool_text(item["uses_gt"]):
                    raise RuntimeError("C3R diagnostics report GT use")
                if int(item["packet_bytes"]) != 320:
                    raise RuntimeError("C3R packet byte contract mismatch")
                gates = json.loads(item["gates"])
                if any(not math.isfinite(float(gate)) for gate in gates):
                    raise RuntimeError("non-finite C3R gate")
                diagnostics_rows.append(dict(item, sequence=sequence))
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if int(summary["serialized_packet_bytes"]) != 320:
                raise RuntimeError("communication summary byte mismatch")
            communication_rows.append(dict(summary, sequence=sequence))
    return {
        "entry": row,
        "split": str(split),
        "result_path": str(result_path),
        "sequence_rows": sequence_rows,
        "diagnostics_rows": diagnostics_rows,
        "communication_rows": communication_rows,
    }


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_targets(sequence_rows):
    grouped = defaultdict(list)
    for row in sequence_rows:
        grouped[row["target"]].append(row)
    output = []
    for target in sorted(grouped):
        rows = grouped[target]
        item = {"experiment_id": rows[0]["experiment_id"], "target": target}
        for metric in METRICS:
            item[metric] = float(np.mean([row[metric] for row in rows]))
        output.append(item)
    return output


def run_metrics(row, registry_path, gt_root, output_dir):
    inspected = inspect_experiment(row, registry_path, gt_root)
    output_dir = Path(output_dir)
    experiment_id = row["experiment_id"]
    sequence_rows = inspected["sequence_rows"]
    target_rows = aggregate_targets(sequence_rows)
    write_csv(output_dir / ("sequence_metrics_" + experiment_id + ".csv"), sequence_rows)
    write_csv(output_dir / ("target_metrics_" + experiment_id + ".csv"), target_rows)
    summary = {
        "experiment_id": experiment_id,
        "targets": len(target_rows),
        "sequences": len(sequence_rows),
        "metrics": {
            metric: float(np.mean([row[metric] for row in target_rows]))
            for metric in METRICS
        },
        "uses_gt_inference": False,
        "message_mode": row["message_mode"],
    }
    (output_dir / ("metrics_" + experiment_id + ".json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    role = experiment_id.split("_", 1)[0]
    audit_name = {"E0": "30_e0_evaluation_audit.md",
                  "C0": "31_c0_evaluation_audit.md",
                  "C1": "32_c1_evaluation_audit.md"}[role]
    (output_dir / audit_name).write_text(
        "# {0} evaluation audit\n\nStatus: PASS.\n\n"
        "- Targets: {1}/{1}\n- Sequences: {2}/{2}\n"
        "- Prediction/score/APCE lengths: complete\n"
        "- All tracker outputs finite\n- Uses GT during inference: false\n"
        "- Message mode: `{3}`\n".format(
            experiment_id, row["expected_targets"],
            row["expected_sequences"], row["message_mode"]),
        encoding="utf-8")
    return inspected, target_rows, summary


def run_completeness(rows, registry_path, gt_root, output_dir):
    records = []
    for row in rows:
        inspected = inspect_experiment(row, registry_path, gt_root)
        records.append((row["experiment_id"], len(inspected["sequence_rows"]),
                        int(row["expected_sequences"])))
    fold_id = rows[0]["fold_id"]
    lines = ["# Fold {} evaluation completeness".format(fold_id), "",
             "Status: **PASS**.", ""]
    lines.extend("- {}: {}/{} sequences complete".format(name, count, expected)
                 for name, count, expected in records)
    lines.append("- no-GT identity markers and diagnostics: PASS")
    lines.append("- checkpoint/config/runid identity: PASS")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "33_evaluation_completeness.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_pair(rows, output_dir):
    output_dir = Path(output_dir)
    fold_id = rows[0]["fold_id"]
    role_ids = {role: "{}_F{}".format(role, fold_id)
                for role in ("E0", "C0", "C1")}
    by_experiment = {
        row["experiment_id"]: read_csv_rows(
            output_dir / ("target_metrics_" + row["experiment_id"] + ".csv"))
        for row in rows
    }
    sequence_by_experiment = {
        row["experiment_id"]: read_csv_rows(
            output_dir / ("sequence_metrics_" + row["experiment_id"] + ".csv"))
        for row in rows
    }
    targets = sorted({row["target"] for row in by_experiment[role_ids["E0"]]})
    target_output = []
    for target in targets:
        item = {"target": target}
        role_rows = {}
        for experiment_id in (role_ids["E0"], role_ids["C0"], role_ids["C1"]):
            role_rows[experiment_id] = next(
                row for row in by_experiment[experiment_id]
                if row["target"] == target)
        for metric in METRICS:
            e0 = float(role_rows[role_ids["E0"]][metric])
            c0 = float(role_rows[role_ids["C0"]][metric])
            c1 = float(role_rows[role_ids["C1"]][metric])
            item.update({
                "e0_" + metric: e0,
                "c0_" + metric: c0,
                "c1_" + metric: c1,
                "c0_minus_e0_" + metric: c0 - e0,
                "c1_minus_e0_" + metric: c1 - e0,
                "c1_minus_c0_" + metric: c1 - c0,
            })
        target_output.append(item)
    write_csv(output_dir / ("34_fold{}_target_metrics.csv".format(fold_id)), target_output)

    sequence_output = []
    sequence_names = sorted({row["sequence"] for row in
                             sequence_by_experiment[role_ids["E0"]]})
    for sequence in sequence_names:
        item = {"sequence": sequence, "target": sequence.rsplit("-", 1)[0],
                "view": int(sequence.rsplit("-", 1)[1])}
        role_rows = {
            experiment_id: next(
                row for row in sequence_by_experiment[experiment_id]
                if row["sequence"] == sequence)
            for experiment_id in (role_ids["E0"], role_ids["C0"], role_ids["C1"])
        }
        for metric in METRICS:
            for experiment_id, prefix in (
                    (role_ids["E0"], "e0"), (role_ids["C0"], "c0"),
                    (role_ids["C1"], "c1")):
                item[prefix + "_" + metric] = float(role_rows[experiment_id][metric])
        sequence_output.append(item)
    write_csv(output_dir / ("35_fold{}_sequence_metrics.csv".format(fold_id)),
              sequence_output)

    lines = ["# Fold {} E0/C0/C1 pair report".format(fold_id), "",
             "Result label: **Fold {} execution and diagnostic gate**.".format(
                 fold_id), ""]
    for metric in METRICS:
        e0 = np.mean([row["e0_" + metric] for row in target_output])
        c0 = np.mean([row["c0_" + metric] for row in target_output])
        c1 = np.mean([row["c1_" + metric] for row in target_output])
        lines.append(
            "- {}: E0={:.6f}, C0={:.6f}, C1={:.6f}, C1-C0={:.6f}".format(
                metric, e0, c0, c1, c1 - c0))
    pair_name = "38_fold{}_pair_report.md".format(fold_id)
    (output_dir / pair_name).write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "label": "fold{} execution and diagnostic gate".format(fold_id),
        "target_count": len(target_output),
        "sequence_count": len(sequence_output),
        "files": [
            "34_fold{}_target_metrics.csv".format(fold_id),
            "35_fold{}_sequence_metrics.csv".format(fold_id), pair_name,
        ],
    }
    (output_dir / ("39_fold{}_result_manifest.md".format(fold_id))).write_text(
        "# Fold {} result manifest\n\n```json\n{}\n```\n".format(
            fold_id,
            json.dumps(manifest, indent=2, sort_keys=True)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--fold-id", type=int, default=0, choices=range(5))
    parser.add_argument("--mode", choices=("metrics", "completeness", "pair"), required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--gt-root", default="/data2/Three-MDOT")
    parser.add_argument(
        "--output-dir")
    args = parser.parse_args()
    rows = read_registry(args.registry, args.fold_id)
    if args.output_dir is None:
        args.output_dir = (
            "output/multi_agent_collaboration_clean/formal/fold_{}".format(
                args.fold_id))
    if args.mode == "metrics":
        if args.experiment is None:
            parser.error("--experiment is required for metrics mode")
        row = next(item for item in rows if item["experiment_id"] == args.experiment)
        run_metrics(row, args.registry, args.gt_root, args.output_dir)
    elif args.mode == "completeness":
        run_completeness(rows, args.registry, args.gt_root, args.output_dir)
    else:
        run_pair(rows, args.output_dir)


if __name__ == "__main__":
    main()
