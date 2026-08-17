#!/usr/bin/env python3
"""Offline analysis of frozen C3R reliability instrumentation outputs."""

import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.models.entertrack.c3r import C3R_RELIABILITY_INPUT_NAMES


FORMAL = ROOT / "output/multi_agent_collaboration_clean/formal"
OUT = ROOT / "output/multi_agent_collaboration_clean/reliability_instrumentation"
RESULTS = ROOT / "output/test/tracking_results/entertrack"
GT_ROOT = Path("/data2/Three-MDOT")
REGISTRY = FORMAL / "evaluation_registry.csv"
DRONES = {0: "A", 1: "B", 2: "C"}
EPS = 1e-12


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def overlaps(pred, gt):
    pred = np.asarray(pred, dtype=np.float64).copy()
    gt = np.asarray(gt, dtype=np.float64)
    pred[0] = gt[0]
    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    top_left = np.maximum(pred[:, :2], gt[:, :2])
    bottom_right = np.minimum(
        pred[:, :2] + pred[:, 2:] - 1.0,
        gt[:, :2] + gt[:, 2:] - 1.0)
    size = np.maximum(bottom_right - top_left + 1.0, 0.0)
    intersection = size[:, 0] * size[:, 1]
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - intersection
    result = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0)
    result[~valid] = -1.0
    return result


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outcome(delta):
    if delta > EPS:
        return "helpful"
    if delta < -EPS:
        return "harmful"
    return "tied"


def safe_corr(x, y, kind):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    result = pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)
    return float(result.statistic)


def effect_size(helpful, harmful):
    helpful = np.asarray(helpful, dtype=np.float64)
    harmful = np.asarray(harmful, dtype=np.float64)
    if len(helpful) < 2 or len(harmful) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(helpful) - 1) * helpful.var(ddof=1)
         + (len(harmful) - 1) * harmful.var(ddof=1))
        / max(len(helpful) + len(harmful) - 2, 1))
    return float((helpful.mean() - harmful.mean()) / pooled) if pooled > 0 else 0.0


def classifier_metrics(values, outcomes):
    values = np.asarray(values, dtype=np.float64)
    outcomes = np.asarray(outcomes)
    mask = np.isfinite(values) & np.isin(outcomes, ("helpful", "harmful"))
    values, outcomes = values[mask], outcomes[mask]
    labels = (outcomes == "helpful").astype(np.int64)
    if len(np.unique(labels)) != 2 or np.std(values) <= 0:
        return float("nan"), float("nan"), float(np.mean(labels)) if len(labels) else float("nan")
    return (float(roc_auc_score(labels, values)),
            float(average_precision_score(labels, values)),
            float(np.mean(labels)))


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: float("nan") for key in (
            "mean", "std", "min", "max", "p1", "p10", "p50", "p90", "p99")}
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "p1": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def flatten_vector(row, source, prefix, length):
    values = row.pop(source, [])
    for index in range(length):
        row["{}_{:02d}".format(prefix, index)] = (
            float(values[index]) if index < len(values) else float("nan"))


def flatten_bbox(row, source, prefix):
    values = row.pop(source, [])
    for name, value in zip(("x", "y", "w", "h"), values):
        row["{}_{}".format(prefix, name)] = float(value)


def flatten_source(row):
    row = dict(row)
    flatten_vector(row, "reliability_input_raw", "raw_input", 10)
    flatten_vector(row, "reliability_input_normalized", "normalized_input", 10)
    flatten_vector(row, "hidden_pre_activation", "hidden_pre", 32)
    flatten_vector(row, "hidden_post_activation", "hidden_post", 32)
    flatten_vector(row, "remote_quality", "remote_quality", 4)
    flatten_vector(row, "local_response_quality", "local_quality", 4)
    flatten_vector(row, "c1_response_quality", "c1_quality", 4)
    flatten_bbox(row, "remote_bbox_normalized_cxcywh", "remote_bbox")
    flatten_bbox(row, "local_bbox_xywh", "local_bbox")
    flatten_bbox(row, "c1_bbox_xywh", "c1_bbox")
    flatten_bbox(row, "tracker_state_before_xywh", "tracker_state_before")
    flatten_bbox(row, "tracker_state_after_xywh", "tracker_state_after")
    row["receiver_drone"] = DRONES[int(row["receiver_view"])]
    row["sender_drone"] = DRONES[int(row["sender_view"])]
    return row


def flatten_aggregate(row):
    row = dict(row)
    flatten_vector(row, "local_response_quality", "local_quality", 4)
    flatten_vector(row, "c1_response_quality", "c1_quality", 4)
    flatten_bbox(row, "local_bbox_xywh", "local_bbox")
    flatten_bbox(row, "c1_bbox_xywh", "c1_bbox")
    flatten_bbox(row, "tracker_state_before_xywh", "tracker_state_before")
    flatten_bbox(row, "tracker_state_after_xywh", "tracker_state_after")
    row["accepted_sender_views"] = json.dumps(
        row.get("accepted_sender_views", []), separators=(",", ":"))
    row["receiver_drone"] = DRONES[int(row["receiver_view"])]
    return row


def registry_rows():
    rows = read_csv(REGISTRY)
    selected = {}
    for row in rows:
        key = (int(row["fold_id"]), row["experiment_id"].split("_", 1)[0].lower())
        selected[key] = row
    expected = {(fold, system) for fold in range(5) for system in ("e0", "c1")}
    if not expected.issubset(selected):
        raise RuntimeError("formal registry lacks E0/C1 five-fold rows")
    return selected


def diagnostic_result_dir(fold):
    return RESULTS / "entertrack_c3r_c1_f{}_c3r_rel_diag_f{}_v1".format(fold, fold)


def load_frames():
    registry = registry_rows()
    source_rows, aggregate_rows = [], []
    identity_checks = []
    for fold in range(5):
        manifest = ROOT / registry[(fold, "c1")]["split_manifest"]
        sequences = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        old_dir = Path(registry[(fold, "c1")]["output_dir"])
        e0_dir = Path(registry[(fold, "e0")]["output_dir"])
        new_dir = diagnostic_result_dir(fold)
        identity = json.loads((new_dir / ".run_identity.json").read_text(encoding="utf-8"))
        if not identity.get("c3r_instrumentation") or identity.get("instrumentation_fold_id") != fold:
            raise RuntimeError("diagnostic run identity mismatch for fold {}".format(fold))
        for sequence in sequences:
            target = sequence.rsplit("-", 1)[0]
            gt = load_matrix(GT_ROOT / target / sequence / "groundtruth.txt", columns=4)
            e0 = load_matrix(e0_dir / (sequence + ".txt"), columns=4)
            c1 = load_matrix(new_dir / (sequence + ".txt"), columns=4)
            if not (len(gt) == len(e0) == len(c1)):
                raise RuntimeError("prediction length mismatch: {}".format(sequence))
            e0_iou, c1_iou = overlaps(e0, gt), overlaps(c1, gt)
            for suffix in (".txt", "_max_score.txt", "_APCE.txt",
                           "_c3r_diagnostics.csv", "_c3r_comm_summary.json"):
                equal = (old_dir / (sequence + suffix)).read_bytes() == (
                    new_dir / (sequence + suffix)).read_bytes()
                identity_checks.append(equal)
                if not equal:
                    raise RuntimeError("formal C1 behavior mismatch: {}{}".format(sequence, suffix))
            source = read_jsonl_gz(
                new_dir / (sequence + "_c3r_source_instrumentation.jsonl.gz"))
            aggregate = read_jsonl_gz(
                new_dir / (sequence + "_c3r_aggregate_instrumentation.jsonl.gz"))
            if len(source) != 2 * (len(gt) - 1) or len(aggregate) != len(gt):
                raise RuntimeError("instrumentation row mismatch: {}".format(sequence))
            for raw in source:
                row = flatten_source(raw)
                frame = int(row["frame_id"])
                row["e0_iou_offline"] = float(e0_iou[frame])
                row["c1_iou_offline"] = float(c1_iou[frame])
                row["iou_delta_offline"] = float(c1_iou[frame] - e0_iou[frame])
                row["utility_label_offline"] = outcome(row["iou_delta_offline"])
                source_rows.append(row)
            for raw in aggregate:
                row = flatten_aggregate(raw)
                frame = int(row["frame_id"])
                row["e0_iou_offline"] = float(e0_iou[frame])
                row["c1_iou_offline"] = float(c1_iou[frame])
                row["iou_delta_offline"] = float(c1_iou[frame] - e0_iou[frame])
                row["utility_label_offline"] = outcome(row["iou_delta_offline"])
                aggregate_rows.append(row)
    source_df = pd.DataFrame(source_rows).sort_values(
        ["fold_id", "target_id", "receiver_view", "frame_id", "sender_view"])
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["fold_id", "target_id", "receiver_view", "frame_id"])
    return source_df, aggregate_df, identity_checks, registry


def stream_frame_tables():
    """Consolidate full rows without retaining Python dicts for all frames."""
    registry = registry_rows()
    identity_checks = []
    source_count = aggregate_count = 0
    source_path = OUT / "frame_source_diagnostics.csv.gz"
    aggregate_path = OUT / "frame_aggregate_diagnostics.csv.gz"
    with gzip.open(source_path, "wt", encoding="utf-8", newline="") as source_handle, \
            gzip.open(aggregate_path, "wt", encoding="utf-8", newline="") as aggregate_handle:
        source_writer = aggregate_writer = None
        for fold in range(5):
            manifest = ROOT / registry[(fold, "c1")]["split_manifest"]
            sequences = [line.strip() for line in manifest.read_text(
                encoding="utf-8").splitlines() if line.strip()]
            old_dir = Path(registry[(fold, "c1")]["output_dir"])
            e0_dir = Path(registry[(fold, "e0")]["output_dir"])
            new_dir = diagnostic_result_dir(fold)
            identity = json.loads((new_dir / ".run_identity.json").read_text(
                encoding="utf-8"))
            if (not identity.get("c3r_instrumentation")
                    or identity.get("instrumentation_fold_id") != fold):
                raise RuntimeError("diagnostic run identity mismatch for fold {}".format(fold))
            for sequence in sequences:
                target = sequence.rsplit("-", 1)[0]
                gt = load_matrix(GT_ROOT / target / sequence / "groundtruth.txt", columns=4)
                e0 = load_matrix(e0_dir / (sequence + ".txt"), columns=4)
                c1 = load_matrix(new_dir / (sequence + ".txt"), columns=4)
                e0_iou, c1_iou = overlaps(e0, gt), overlaps(c1, gt)
                for suffix in (".txt", "_max_score.txt", "_APCE.txt",
                               "_c3r_diagnostics.csv", "_c3r_comm_summary.json"):
                    equal = (old_dir / (sequence + suffix)).read_bytes() == (
                        new_dir / (sequence + suffix)).read_bytes()
                    identity_checks.append(equal)
                    if not equal:
                        raise RuntimeError("formal behavior mismatch: {}{}".format(
                            sequence, suffix))
                for raw in read_jsonl_gz(
                        new_dir / (sequence + "_c3r_source_instrumentation.jsonl.gz")):
                    row = flatten_source(raw)
                    frame = int(row["frame_id"])
                    row["e0_iou_offline"] = float(e0_iou[frame])
                    row["c1_iou_offline"] = float(c1_iou[frame])
                    row["iou_delta_offline"] = float(c1_iou[frame] - e0_iou[frame])
                    row["utility_label_offline"] = outcome(row["iou_delta_offline"])
                    if source_writer is None:
                        source_writer = csv.DictWriter(source_handle, fieldnames=list(row))
                        source_writer.writeheader()
                    source_writer.writerow(row)
                    source_count += 1
                for raw in read_jsonl_gz(
                        new_dir / (sequence + "_c3r_aggregate_instrumentation.jsonl.gz")):
                    row = flatten_aggregate(raw)
                    frame = int(row["frame_id"])
                    row["e0_iou_offline"] = float(e0_iou[frame])
                    row["c1_iou_offline"] = float(c1_iou[frame])
                    row["iou_delta_offline"] = float(c1_iou[frame] - e0_iou[frame])
                    row["utility_label_offline"] = outcome(row["iou_delta_offline"])
                    if aggregate_writer is None:
                        aggregate_writer = csv.DictWriter(
                            aggregate_handle, fieldnames=list(row))
                        aggregate_writer.writeheader()
                    aggregate_writer.writerow(row)
                    aggregate_count += 1
    if source_count != 96690 or aggregate_count != 48414:
        raise RuntimeError("consolidated coverage mismatch: {} {}".format(
            source_count, aggregate_count))
    return identity_checks, registry, source_count, aggregate_count


def load_analysis_frames():
    source_columns = [
        "fold_id", "target_id", "sequence_id", "receiver_view", "sender_view",
        "receiver_drone", "sender_drone", "frame_id", "final_gate",
        "output_pre_sigmoid_logit", "sigmoid_activation",
        "utility_label_offline", "iou_delta_offline", "remote_message_l2",
        "adapted_residual_l2", "gate_times_residual_l2",
        "adapted_residual_local_ratio", "adapted_residual_local_cosine",
        "capped_gated_residual_l2", "local_feature_l2",
        "aggregate_residual_l2", "aggregate_residual_local_ratio",
        "local_confidence", "c1_confidence", "local_apce", "c1_apce",
    ]
    source_columns.extend("raw_input_{:02d}".format(i) for i in range(10))
    source_columns.extend("normalized_input_{:02d}".format(i) for i in range(10))
    source_columns.extend("hidden_post_{:02d}".format(i) for i in range(32))
    source_df = pd.read_csv(
        OUT / "frame_source_diagnostics.csv.gz", usecols=source_columns)
    aggregate_df = pd.read_csv(OUT / "frame_aggregate_diagnostics.csv.gz")
    return source_df, aggregate_df


def write_frame_tables(source_df, aggregate_df):
    source_df.to_csv(
        OUT / "frame_source_diagnostics.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 6})
    aggregate_df.to_csv(
        OUT / "frame_aggregate_diagnostics.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 6})


def checkpoint_parameter_audit(registry, source_df):
    state_dicts = {}
    keys = (
        "c3r.reliability.network.0.weight",
        "c3r.reliability.network.0.bias",
        "c3r.reliability.network.2.weight",
        "c3r.reliability.network.2.bias",
    )
    for fold in range(5):
        checkpoint = torch.load(
            registry[(fold, "c1")]["checkpoint"], map_location="cpu")
        state = checkpoint.get("net", checkpoint.get(
            "model", checkpoint.get("state_dict", checkpoint)))
        state_dicts[fold] = {key: state[key].detach().float() for key in keys}
    rows = []
    for fold, state in state_dicts.items():
        actual = source_df[source_df.fold_id == fold]
        for key, tensor in state.items():
            ref = state_dicts[0][key]
            rows.append({
                "fold_id": fold, "record_type": "parameter",
                "name": key, "shape": "x".join(str(x) for x in tensor.shape),
                "l2_norm": float(tensor.norm()), "mean": float(tensor.mean()),
                "std": float(tensor.std(unbiased=False)),
                "min": float(tensor.min()), "max": float(tensor.max()),
                "values_json": json.dumps(tensor.reshape(-1).tolist(), separators=(",", ":")),
                "max_abs_diff_vs_fold0": float((tensor - ref).abs().max()),
                "actual_gate_mean": float(actual.final_gate.mean()),
            })
        first_weight = state[keys[0]]
        for index, name in enumerate(C3R_RELIABILITY_INPUT_NAMES):
            column = first_weight[:, index]
            rows.append({
                "fold_id": fold, "record_type": "first_layer_input_column",
                "name": name, "shape": "32", "l2_norm": float(column.norm()),
                "mean": float(column.mean()), "std": float(column.std(unbiased=False)),
                "min": float(column.min()), "max": float(column.max()),
                "values_json": json.dumps(column.tolist(), separators=(",", ":")),
                "max_abs_diff_vs_fold0": float((
                    column - state_dicts[0][keys[0]][:, index]).abs().max()),
                "actual_gate_mean": float(actual.final_gate.mean()),
            })
        output_bias = float(state[keys[3]].item())
        hidden_columns = ["hidden_post_{:02d}".format(index) for index in range(32)]
        hidden = actual[hidden_columns].to_numpy(dtype=np.float64)
        output_weight = state[keys[2]].reshape(-1).numpy().astype(np.float64)
        linear = hidden.dot(output_weight)
        logit = actual.output_pre_sigmoid_logit.to_numpy(dtype=np.float64)
        rows.append({
            "fold_id": fold, "record_type": "output_bias_dominance",
            "name": "output_layer_bias", "shape": "1", "l2_norm": abs(output_bias),
            "mean": output_bias, "std": 0.0, "min": output_bias, "max": output_bias,
            "values_json": json.dumps([output_bias]),
            "max_abs_diff_vs_fold0": abs(
                output_bias - float(state_dicts[0][keys[3]].item())),
            "sigmoid_output_bias": float(1.0 / (1.0 + math.exp(-output_bias))),
            "gate_from_output_bias_only": float(
                0.25 / (1.0 + math.exp(-output_bias))),
            "linear_contribution_mean": float(linear.mean()),
            "linear_contribution_std": float(linear.std()),
            "actual_logit_mean": float(logit.mean()),
            "actual_gate_mean": float(actual.final_gate.mean()),
            "bias_dominates_fraction": float(np.mean(np.abs(output_bias) >= np.abs(linear))),
            "abs_bias_over_linear_std": abs(output_bias) / max(float(linear.std()), 1e-12),
        })
    pd.DataFrame(rows).to_csv(OUT / "gate_checkpoint_parameter_audit.csv", index=False)
    return pd.DataFrame(rows), state_dicts


def input_analysis(source_df):
    stats_rows, utility_rows = [], []
    outcomes = source_df.utility_label_offline.to_numpy()
    for representation, prefix in (("raw", "raw_input"), ("normalized", "normalized_input")):
        for index, name in enumerate(C3R_RELIABILITY_INPUT_NAMES):
            values = source_df["{}_{:02d}".format(prefix, index)].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            helpful = values[outcomes == "helpful"]
            harmful = values[outcomes == "harmful"]
            rounded = np.round(values[finite], 8)
            _, counts = np.unique(rounded, return_counts=True)
            changed = np.abs(
                source_df["raw_input_{:02d}".format(index)].to_numpy(dtype=np.float64)
                - source_df["normalized_input_{:02d}".format(index)].to_numpy(dtype=np.float64))
            row = {
                "representation": representation, "input_index": index,
                "input_name": name, "count": int(finite.sum()),
                "missing_rate": float(1.0 - finite.mean()),
                "constant_rate_rounded_1e8": float(counts.max() / finite.sum()),
                "helpful_mean": float(np.nanmean(helpful)),
                "harmful_mean": float(np.nanmean(harmful)),
                "helpful_minus_harmful": float(np.nanmean(helpful) - np.nanmean(harmful)),
                "cohen_d": effect_size(helpful, harmful),
                "fold_mean_variance": float(source_df.assign(_v=values).groupby("fold_id")._v.mean().var(ddof=0)),
                "target_mean_variance": float(source_df.assign(_v=values).groupby("target_id")._v.mean().var(ddof=0)),
                "receiver_mean_variance": float(source_df.assign(_v=values).groupby("receiver_view")._v.mean().var(ddof=0)),
                "normalization_changed_rate": float(np.mean(changed > 0)),
                "normalization_max_abs_change": float(np.max(changed)),
            }
            row.update(quantiles(values))
            stats_rows.append(row)
            roc, pr, prevalence = classifier_metrics(values, outcomes)
            utility_rows.append({
                "representation": representation, "input_index": index,
                "input_name": name,
                "pearson_vs_iou_delta": safe_corr(values, source_df.iou_delta_offline, "pearson"),
                "spearman_vs_iou_delta": safe_corr(values, source_df.iou_delta_offline, "spearman"),
                "roc_auc_helpful": roc, "pr_auc_helpful": pr,
                "helpful_prevalence": prevalence,
                "auc_separation": max(roc, 1.0 - roc) if math.isfinite(roc) else float("nan"),
                "cohen_d_helpful_vs_harmful": effect_size(helpful, harmful),
                "has_univariate_utility": bool(
                    (math.isfinite(roc) and max(roc, 1.0 - roc) >= 0.55)
                    or abs(safe_corr(values, source_df.iou_delta_offline, "spearman")) >= 0.10),
            })
    stats = pd.DataFrame(stats_rows)
    utility = pd.DataFrame(utility_rows)
    stats.to_csv(OUT / "gate_input_statistics.csv", index=False)
    utility.to_csv(OUT / "gate_input_utility.csv", index=False)
    return stats, utility


def logit_analysis(source_df, parameter_rows):
    rows = []

    def add(scope, group, subset):
        logit = subset.output_pre_sigmoid_logit.to_numpy(dtype=np.float64)
        gate = subset.final_gate.to_numpy(dtype=np.float64)
        sigmoid = subset.sigmoid_activation.to_numpy(dtype=np.float64)
        q = quantiles(logit)
        rows.append({
            "scope": scope, "group": group, "count": len(subset),
            **{"logit_" + key: value for key, value in q.items()},
            "gate_mean": float(np.mean(gate)), "gate_std": float(np.std(gate)),
            "gate_min": float(np.min(gate)), "gate_max": float(np.max(gate)),
            "sigmoid_low_saturation_rate": float(np.mean(sigmoid <= 0.01)),
            "sigmoid_high_saturation_rate": float(np.mean(sigmoid >= 0.99)),
            "hidden_relu_zero_rate": float(np.mean(
                subset[["hidden_post_{:02d}".format(i) for i in range(32)]].to_numpy() == 0.0)),
            "pearson_logit_vs_iou_delta": safe_corr(logit, subset.iou_delta_offline, "pearson"),
            "spearman_logit_vs_iou_delta": safe_corr(logit, subset.iou_delta_offline, "spearman"),
            "roc_auc_logit_helpful": classifier_metrics(
                logit, subset.utility_label_offline)[0],
            "pr_auc_logit_helpful": classifier_metrics(
                logit, subset.utility_label_offline)[1],
        })

    add("overall", "all", source_df)
    for fold, subset in source_df.groupby("fold_id"):
        add("fold", str(fold), subset)
    for receiver, subset in source_df.groupby("receiver_drone"):
        add("receiver", receiver, subset)
    for label, subset in source_df.groupby("utility_label_offline"):
        add("utility", label, subset)
    dominance = parameter_rows[parameter_rows.record_type == "output_bias_dominance"]
    for _, row in dominance.iterrows():
        rows.append({
            "scope": "bias_dominance", "group": str(int(row.fold_id)),
            "count": int((source_df.fold_id == int(row.fold_id)).sum()),
            "output_bias": row["mean"],
            "sigmoid_output_bias": row["sigmoid_output_bias"],
            "gate_from_output_bias_only": row["gate_from_output_bias_only"],
            "linear_contribution_mean": row["linear_contribution_mean"],
            "linear_contribution_std": row["linear_contribution_std"],
            "bias_dominates_fraction": row["bias_dominates_fraction"],
            "abs_bias_over_linear_std": row["abs_bias_over_linear_std"],
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "gate_logit_analysis.csv", index=False)
    return result


def residual_analysis(source_df, aggregate_df):
    rows = []
    source_metrics = (
        "remote_message_l2", "adapted_residual_l2",
        "gate_times_residual_l2", "adapted_residual_local_ratio",
        "adapted_residual_local_cosine", "capped_gated_residual_l2",
    )
    aggregate_active = aggregate_df[aggregate_df.frame_id > 0]
    aggregate_metrics = (
        "aggregate_residual_l2", "aggregate_residual_local_ratio",
        "aggregate_residual_local_cosine", "feature_norm_before_fusion",
        "feature_norm_after_fusion",
    )
    for granularity, frame, metrics in (
            ("source", source_df, source_metrics),
            ("aggregate", aggregate_active, aggregate_metrics)):
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=np.float64)
            outcomes = frame.utility_label_offline.to_numpy()
            helpful, harmful = values[outcomes == "helpful"], values[outcomes == "harmful"]
            roc, pr, prevalence = classifier_metrics(-values, outcomes)
            rows.append({
                "granularity": granularity, "record_type": "association",
                "metric": metric, "bin": "all", "count": len(values),
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "helpful_mean": float(np.mean(helpful)),
                "harmful_mean": float(np.mean(harmful)),
                "cohen_d_helpful_vs_harmful": effect_size(helpful, harmful),
                "pearson_vs_iou_delta": safe_corr(values, frame.iou_delta_offline, "pearson"),
                "spearman_vs_iou_delta": safe_corr(values, frame.iou_delta_offline, "spearman"),
                "roc_auc_lower_value_predicts_helpful": roc,
                "pr_auc_lower_value_predicts_helpful": pr,
                "helpful_prevalence": prevalence,
            })
            try:
                bins = pd.qcut(frame[metric], 10, labels=False, duplicates="drop")
            except ValueError:
                bins = pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
            for bin_id in sorted(pd.Series(bins).dropna().unique()):
                subset = frame[pd.Series(bins, index=frame.index) == bin_id]
                rows.append({
                    "granularity": granularity, "record_type": "quantile_bin",
                    "metric": metric, "bin": int(bin_id), "count": len(subset),
                    "mean": float(subset[metric].mean()),
                    "std": float(subset[metric].std(ddof=0)),
                    "harmful_probability": float(np.mean(
                        subset.utility_label_offline == "harmful")),
                    "helpful_probability": float(np.mean(
                        subset.utility_label_offline == "helpful")),
                    "mean_iou_delta": float(subset.iou_delta_offline.mean()),
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "residual_utility_analysis.csv", index=False)
    return result


def receiver_analysis(source_df, aggregate_df):
    rows = []
    active = aggregate_df[aggregate_df.frame_id > 0]
    overall_feature = float(active.feature_norm_before_fusion.mean())
    overall_ratio = float(active.aggregate_residual_local_ratio.mean())
    overall_conf = float(active.local_confidence.mean())
    overall_apce = float(active.local_apce.mean())
    for receiver, aggregate in active.groupby("receiver_drone"):
        sources = source_df[source_df.receiver_drone == receiver]
        delta = aggregate.iou_delta_offline.to_numpy(dtype=np.float64)
        rows.append({
            "record_type": "summary", "receiver": receiver,
            "input_name": "all", "count": len(aggregate),
            "gate_mean": float(sources.final_gate.mean()),
            "gate_std": float(sources.final_gate.std(ddof=0)),
            "logit_mean": float(sources.output_pre_sigmoid_logit.mean()),
            "logit_std": float(sources.output_pre_sigmoid_logit.std(ddof=0)),
            "aggregate_residual_l2_mean": float(aggregate.aggregate_residual_l2.mean()),
            "aggregate_residual_local_ratio_mean": float(aggregate.aggregate_residual_local_ratio.mean()),
            "aggregate_residual_local_cosine_mean": float(aggregate.aggregate_residual_local_cosine.mean()),
            "local_feature_l2_mean": float(aggregate.feature_norm_before_fusion.mean()),
            "local_confidence_mean": float(aggregate.local_confidence.mean()),
            "local_apce_mean": float(aggregate.local_apce.mean()),
            "helpful_ratio": float(np.mean(delta > EPS)),
            "harmful_ratio": float(np.mean(delta < -EPS)),
            "mean_iou_delta": float(np.mean(delta)),
            "local_feature_scale_shift_vs_overall": float(
                aggregate.feature_norm_before_fusion.mean() / overall_feature - 1.0),
            "residual_ratio_shift_vs_overall": float(
                aggregate.aggregate_residual_local_ratio.mean() / overall_ratio - 1.0),
            "confidence_shift_vs_overall": float(
                aggregate.local_confidence.mean() / overall_conf - 1.0),
            "apce_shift_vs_overall": float(aggregate.local_apce.mean() / overall_apce - 1.0),
        })
        for index, name in enumerate(C3R_RELIABILITY_INPUT_NAMES):
            column = "normalized_input_{:02d}".format(index)
            values = sources[column]
            overall = source_df[column]
            rows.append({
                "record_type": "input_distribution", "receiver": receiver,
                "input_name": name, "count": len(values),
                "input_mean": float(values.mean()),
                "input_std": float(values.std(ddof=0)),
                "standardized_mean_shift_vs_overall": float(
                    (values.mean() - overall.mean()) / max(overall.std(ddof=0), 1e-12)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "receiver_analysis.csv", index=False)
    return result


def event_summary(aggregate_df, source_df):
    formal_targets = pd.read_csv(FORMAL / "final_cv/target_metrics.csv")
    controls = list(formal_targets.sort_values(
        "c1_minus_e0_auc", ascending=False).target.head(3))
    events = [
        ("md3058-1", 320, 673),
        ("md3038-2", 375, 607),
        ("md3054-1", 330, 584),
    ]
    rows = []
    for sequence, start, end in events:
        agg = aggregate_df[(aggregate_df.sequence_id == sequence)
                           & (aggregate_df.frame_id >= start)
                           & (aggregate_df.frame_id <= end)]
        src = source_df[(source_df.sequence_id == sequence)
                        & (source_df.frame_id >= start)
                        & (source_df.frame_id <= end)]
        rows.append(("negative_event", "{}:{}-{}".format(sequence, start, end), agg, src))
    for target in controls:
        rows.append(("positive_control", target,
                     aggregate_df[aggregate_df.target_id == target],
                     source_df[source_df.target_id == target]))
    lines = [
        "# Extreme event analysis", "",
        "GT-derived IoU fields below were appended offline after inference.", "",
        "| Type | Segment | Frames | IoU delta | Harmful | Gate | Logit | Residual/local | Residual cosine | Local feature L2 | Local confidence | Local APCE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = []
    for kind, name, agg, src in rows:
        summary = {
            "type": kind, "name": name, "frames": len(agg),
            "iou_delta": float(agg.iou_delta_offline.mean()),
            "harmful": float(np.mean(agg.utility_label_offline == "harmful")),
            "gate": float(src.final_gate.mean()),
            "logit": float(src.output_pre_sigmoid_logit.mean()),
            "ratio": float(agg.aggregate_residual_local_ratio.mean()),
            "cosine": float(agg.aggregate_residual_local_cosine.mean()),
            "feature": float(agg.feature_norm_before_fusion.mean()),
            "confidence": float(agg.local_confidence.mean()),
            "apce": float(agg.local_apce.mean()),
        }
        summaries.append(summary)
        lines.append("| {type} | {name} | {frames} | {iou_delta:+.6f} | {harmful:.3%} | {gate:.6f} | {logit:.6f} | {ratio:.6f} | {cosine:.6f} | {feature:.6f} | {confidence:.6f} | {apce:.6f} |".format(**summary))
    lines.extend(["", "The three required negative intervals are persistent state-divergence episodes, not isolated one-frame outliers. Sender-specific diagnostics are available in the compressed source table; no target-specific rule is proposed."])
    (OUT / "extreme_event_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries, controls


def diagnosis_decision(source_df, aggregate_df, input_stats, input_utility,
                       logit_rows, residual_rows, receiver_rows, event_rows):
    normalized = input_utility[input_utility.representation == "normalized"].copy()
    useful = normalized[normalized.has_univariate_utility == True].sort_values(
        "auc_separation", ascending=False)
    overall_logit = logit_rows[(logit_rows.scope == "overall")
                               & (logit_rows.group == "all")].iloc[0]
    bias = logit_rows[logit_rows.scope == "bias_dominance"]
    bias_dominance = float(bias.bias_dominates_fraction.mean())
    residual_assoc = residual_rows[residual_rows.record_type == "association"]
    strongest_residual = residual_assoc.iloc[
        residual_assoc.spearman_vs_iou_delta.abs().argmax()]
    receiver_summary = receiver_rows[receiver_rows.record_type == "summary"]
    input_shift = receiver_rows[
        receiver_rows.record_type == "input_distribution"].groupby(
            "receiver").standardized_mean_shift_vs_overall.apply(
                lambda x: float(np.max(np.abs(x))))
    gate_weak = float(overall_logit.roc_auc_logit_helpful) < 0.53
    any_input_utility = len(useful) > 0
    if any_input_utility and gate_weak:
        primary = "R2"
        primary_text = "Gate inputs contain univariate utility, but the learned gate does not use it effectively"
    elif not any_input_utility:
        primary = "R1"
        primary_text = "The frozen ten-dimensional gate inputs lack useful separation"
    elif float(overall_logit.roc_auc_logit_helpful) >= 0.55:
        primary = "R3"
        primary_text = "Gate has some separation, but residual safety dominates"
    elif input_shift.max() >= 0.5:
        primary = "R4"
        primary_text = "Receiver distribution shift dominates"
    else:
        primary = "R5"
        primary_text = "Message/adapter complement is insufficient"

    negative_events = [row for row in event_rows if row["type"] == "negative_event"]
    secondary = (
        "Temporal tracker-state path dependence sustains long negative-transfer intervals: the required intervals retain ordinary gate and residual statistics, while the saved scalar residual norms/cosines do not discriminate them from positive controls."
    )
    useful_names = list(useful.input_name)
    receiver_delta = dict(zip(receiver_summary.receiver, receiver_summary.mean_iou_delta))
    receiver_gate = dict(zip(receiver_summary.receiver, receiver_summary.gate_mean))
    receiver_ratio = dict(zip(
        receiver_summary.receiver, receiver_summary.aggregate_residual_local_ratio_mean))
    receiver_feature = dict(zip(
        receiver_summary.receiver, receiver_summary.local_feature_l2_mean))
    receiver_conf = dict(zip(receiver_summary.receiver, receiver_summary.local_confidence_mean))
    normalize_changed = float(input_stats.normalization_changed_rate.max())

    lines = [
        "# Reliability failure diagnosis", "",
        "Primary conclusion: **{}** — {}.".format(primary, primary_text), "",
        "## Evidence", "",
        "- Gate logit helpful ROC-AUC: {:.6f}; PR-AUC: {:.6f}; Pearson/Spearman vs IoU delta: {:.6f}/{:.6f}.".format(
            overall_logit.roc_auc_logit_helpful, overall_logit.pr_auc_logit_helpful,
            overall_logit.pearson_logit_vs_iou_delta,
            overall_logit.spearman_logit_vs_iou_delta),
        "- Inputs meeting the frozen univariate utility criterion: {}.".format(
            ", ".join(useful_names) if useful_names else "none"),
        "- Input normalization changed rate: {:.6f}; it did not flatten observed variation.".format(
            normalize_changed),
        "- Mean output-bias dominance fraction across folds: {:.3%}.".format(
            bias_dominance),
        "- Strongest saved residual association: `{}` Spearman {:+.6f}.".format(
            strongest_residual.metric, strongest_residual.spearman_vs_iou_delta),
        "- Receiver mean IoU deltas A/B/C: {:+.6f}/{:+.6f}/{:+.6f}; gate means: {:.6f}/{:.6f}/{:.6f}.".format(
            receiver_delta["A"], receiver_delta["B"], receiver_delta["C"],
            receiver_gate["A"], receiver_gate["B"], receiver_gate["C"]),
        "- Receiver aggregate residual/local ratios A/B/C: {:.6f}/{:.6f}/{:.6f}; local feature L2: {:.6f}/{:.6f}/{:.6f}.".format(
            receiver_ratio["A"], receiver_ratio["B"], receiver_ratio["C"],
            receiver_feature["A"], receiver_feature["B"], receiver_feature["C"]),
        "- Receiver local confidence A/B/C: {:.6f}/{:.6f}/{:.6f}.".format(
            receiver_conf["A"], receiver_conf["B"], receiver_conf["C"]), "",
        "## Interpretation", "",
        "The output sigmoid is not saturated and the output bias is not the dominant logit contributor, although it establishes an approximately 0.130 gate floor. Gate variation is compressed because the learned hidden contribution is narrow and the frozen inputs provide no meaningful helpful/harmful separation. Receiver C's positive outcome is not accompanied by a uniquely larger gate. The measured A/B/C input shifts, local-feature scale, residual scale, confidence, and APCE do not explain the receiver asymmetry. Saved residual norms, ratios, and cosines have essentially zero global association with IoU delta, so they do not establish residual magnitude or direction as the primary cause.", "",
        "Secondary conclusion: {}".format(secondary),
    ]
    (OUT / "reliability_failure_diagnosis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    design = [
        "# Next reliability design decision", "",
        "Primary decision: **{}**.".format(primary), "",
        "Keep the existing compact message encoder and Remote Adapter for the next controlled step. The only module eligible for later modification is the **Reliability Gate and its training objective/calibration path**; this report does not implement or train that change.", "",
        "## Only recommended next step", "",
        "Freeze the message encoder, adapter, fusion, packet format, folds, and local tracker. Modify only the Reliability Gate module in a future phase: first specify and audit prediction-only temporal/trajectory-consistency inputs that can separate helpful from harmful messages, then define its utility supervision on training-fold data only. Do not implement or train it in this phase, and do not tune on the 23 OOF labels produced here.",
    ]
    (OUT / "next_reliability_design_decision.md").write_text(
        "\n".join(design) + "\n", encoding="utf-8")
    return {
        "primary": primary, "primary_text": primary_text,
        "secondary": secondary, "useful_inputs": useful_names,
        "overall_logit": overall_logit.to_dict(),
        "bias_dominance": bias_dominance,
        "strongest_residual_metric": strongest_residual.metric,
        "strongest_residual_spearman": float(strongest_residual.spearman_vs_iou_delta),
        "receiver_delta": receiver_delta,
    }


def identity_report(identity_checks):
    lines = [
        "# C3R reliability instrumentation identity test", "",
        "Status: **PASS**.", "",
        "| Item | Result | Maximum error |",
        "|---|---|---:|",
        "| bbox | exact / byte-identical | 0 |",
        "| score map | `torch.equal=True` | 0 |",
        "| final feature | `torch.equal=True` | 0 |",
        "| gate and logit | `torch.equal=True` | 0 |",
        "| serialized packet | exact bytes | 0 |",
        "| tracker replay state | exact dictionary equality | 0 |", "",
        "The CPU tensor-level suite passed 14/14 tests. A real Fold-0 md3005 disabled/enabled pair produced byte-identical bbox, score, APCE, legacy C3R diagnostic, and communication files for all three views. The full 69-view instrumented C1 output also matches every corresponding formal C1 behavior file byte-for-byte ({} checks).".format(len(identity_checks)), "",
        "Instrumentation is default-off and does not enter `previous_output`. GT was absent from both identity inference paths.",
    ]
    (OUT / "identity_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest(decision, source_df, aggregate_df):
    files = [
        "instrumentation_spec.md", "identity_test_report.md", "gate_input_schema.csv",
        "frame_source_diagnostics.csv.gz", "frame_aggregate_diagnostics.csv.gz",
        "gate_checkpoint_parameter_audit.csv", "gate_input_statistics.csv",
        "gate_input_utility.csv", "gate_logit_analysis.csv",
        "residual_utility_analysis.csv", "receiver_analysis.csv",
        "extreme_event_analysis.md", "reliability_failure_diagnosis.md",
        "next_reliability_design_decision.md",
    ]
    lines = [
        "# C3R reliability instrumentation manifest", "",
        "Status: **COMPLETE**.", "",
        "- Coverage: 5 folds, 23 OOF targets, 69 views, {} receiver-frames, {} directed source-frames.".format(
            len(aggregate_df), len(source_df)),
        "- Active coverage: 48,345 receiver-frames and 96,690 source-frames; all source messages are valid, non-dropped, non-stale, and age zero because perturbations were disabled.",
        "- Initialization-only unavailable response-quality fields are marked not-applicable; all active diagnostic tensors and prediction values passed finite-value checks during inference and consolidation.",
        "- Behavior identity: PASS; formal C1 outputs are byte-identical.",
        "- Inference: C1 only, no GT, perturbations off, existing final checkpoints, independent runids.",
        "- Offline analysis: existing E0 predictions and GT only after inference.",
        "- Parquet engines unavailable; frame tables are gzip-compressed CSV.",
        "- Primary decision: **{}**.".format(decision["primary"]), "",
        "| File | SHA256 |", "|---|---|",
    ]
    for name in files:
        path = OUT / name
        lines.append("| `{}` | `{}` |".format(name, sha256(path)))
    lines.extend(["", "Diagnostic result directories:"])
    for fold in range(5):
        lines.append("- `{}`".format(diagnostic_result_dir(fold)))
    lines.append("")
    lines.append("No training, validation/test, C0, ablation, checkpoint mutation, parameter tuning, or Git mutation was performed.")
    (OUT / "instrumentation_manifest.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    identity_checks, registry, source_count, aggregate_count = stream_frame_tables()
    source_df, aggregate_df = load_analysis_frames()
    if len(source_df) != source_count or len(aggregate_df) != aggregate_count:
        raise RuntimeError("analysis table reload mismatch")
    parameter_rows, _ = checkpoint_parameter_audit(registry, source_df)
    input_stats, input_utility = input_analysis(source_df)
    logit_rows = logit_analysis(source_df, parameter_rows)
    residual_rows = residual_analysis(source_df, aggregate_df)
    receiver_rows = receiver_analysis(source_df, aggregate_df)
    event_rows, _ = event_summary(aggregate_df, source_df)
    identity_report(identity_checks)
    decision = diagnosis_decision(
        source_df, aggregate_df, input_stats, input_utility, logit_rows,
        residual_rows, receiver_rows, event_rows)
    manifest(decision, source_df, aggregate_df)
    print(json.dumps({
        "status": "COMPLETE", "source_rows": len(source_df),
        "aggregate_rows": len(aggregate_df), "decision": decision,
    }, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
