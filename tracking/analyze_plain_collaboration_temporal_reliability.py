"""E2A causal temporal sender reliability audit.

The ``freeze`` phase reads prediction-only E1.5 counterfactuals and never loads
annotations.  The ``analyze`` phase verifies the frozen feature digest before
joining post-hoc labels and Three-MDOT inner-val ground truth.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tracking.analyze_plain_collaboration_safe_commit import iou_xywh, write_csv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PREDICTION_SHA256 = (
    "75ebb3cb292202346619e6f81cc46d050fa394a73e7c0d01ad9376545ba95f43")
IDENTITY = ("target_id", "receiver_view", "sender_view", "frame_id")
VIEWS = ("A", "B", "C")
WINDOWS = (4, 8)
FORBIDDEN_DATASETS = {"threemdot", "threemdot_test", "three_mdot_test"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _float(row, name):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean(values):
    values = _finite(values)
    return float(np.mean(values)) if values else float("nan")


def _std(values):
    values = _finite(values)
    return float(np.std(values, ddof=0)) if values else float("nan")


def _slope(values, frames):
    pairs = [(float(frame), float(value)) for frame, value in zip(frames, values)
             if math.isfinite(float(value))]
    if len(pairs) < 2:
        return float("nan")
    x = np.asarray([item[0] for item in pairs], dtype=float)
    y = np.asarray([item[1] for item in pairs], dtype=float)
    x = x - x[0]
    denominator = float(np.sum((x - x.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denominator)


def temporal_summary(values, frames, index, window):
    """Return a strictly causal prefix summary, including current frame."""
    start = max(0, index - int(window) + 1)
    prefix = values[start:index + 1]
    prefix_frames = frames[start:index + 1]
    return {
        "mean": _mean(prefix),
        "std": _std(prefix),
        "slope": _slope(prefix, prefix_frames),
    }


def _delta_one(values, index):
    if index < 1:
        return float("nan")
    current = float(values[index])
    previous = float(values[index - 1])
    return (current - previous
            if math.isfinite(current) and math.isfinite(previous)
            else float("nan"))


def _bbox(row, prefix):
    return np.asarray([
        _float(row, "{}_bbox_{}".format(prefix, field))
        for field in ("x", "y", "w", "h")
    ], dtype=float)


def motion_series(boxes):
    """Prediction-only constant-velocity diagnostics for a causal bbox series."""
    count = len(boxes)
    fields = {
        name: [float("nan")] * count for name in (
            "velocity_x", "velocity_y", "velocity_norm",
            "acceleration_x", "acceleration_y", "acceleration_norm",
            "normalized_motion_residual", "log_area_scale_change",
            "log_area")
    }
    raw_velocities = [None] * count
    normalized_velocities = [None] * count
    for index, box in enumerate(boxes):
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() \
                or box[2] <= 0.0 or box[3] <= 0.0:
            continue
        center = box[:2] + 0.5 * box[2:]
        fields["log_area"][index] = math.log(max(box[2] * box[3], 1e-12))
        if index < 1:
            continue
        previous = np.asarray(boxes[index - 1], dtype=float)
        if previous.shape != (4,) or not np.isfinite(previous).all() \
                or previous[2] <= 0.0 or previous[3] <= 0.0:
            continue
        previous_center = previous[:2] + 0.5 * previous[2:]
        raw_velocity = center - previous_center
        normalization = max(float(np.linalg.norm(previous[2:])), 1e-12)
        velocity = raw_velocity / normalization
        raw_velocities[index] = raw_velocity
        normalized_velocities[index] = velocity
        fields["velocity_x"][index] = float(velocity[0])
        fields["velocity_y"][index] = float(velocity[1])
        fields["velocity_norm"][index] = float(np.linalg.norm(velocity))
        fields["log_area_scale_change"][index] = (
            fields["log_area"][index] - fields["log_area"][index - 1])
        if index >= 2 and normalized_velocities[index - 1] is not None:
            acceleration = velocity - normalized_velocities[index - 1]
            fields["acceleration_x"][index] = float(acceleration[0])
            fields["acceleration_y"][index] = float(acceleration[1])
            fields["acceleration_norm"][index] = float(
                np.linalg.norm(acceleration))
        if index >= 2 and raw_velocities[index - 1] is not None:
            predicted = previous_center + raw_velocities[index - 1]
            current_diag = max(float(np.linalg.norm(box[2:])), 1e-12)
            fields["normalized_motion_residual"][index] = float(
                np.linalg.norm(center - predicted) / current_diag)
    return fields


GROUP_A = (
    "sender_score_t", "sender_apce_t", "sender_entropy_t",
    "sender_bbox_width_t", "sender_bbox_height_t",
    "sender_center_motion_t", "sender_scale_change_t",
)

GROUP_B = tuple(
    "sender_{}_{}".format(metric, statistic)
    for metric in ("score", "apce", "entropy")
    for statistic in (
        "mean_4", "mean_8", "std_4", "std_8",
        "slope_4", "slope_8", "delta_1"))

GROUP_D = tuple([
    "receiver_score_t", "receiver_apce_t", "receiver_entropy_t",
    "receiver_bbox_width_t", "receiver_bbox_height_t",
    "receiver_center_motion_t", "receiver_scale_change_t",
] + [
    "receiver_{}_{}".format(metric, statistic)
    for metric in ("score", "apce", "entropy")
    for statistic in (
        "mean_4", "mean_8", "std_4", "std_8",
        "slope_4", "slope_8", "delta_1")
])

GROUP_E = (
    "diff_score_t", "diff_apce_t", "diff_entropy_t",
    "diff_score_mean_8", "diff_apce_mean_8", "diff_entropy_mean_8",
    "diff_score_slope_8", "diff_apce_slope_8", "diff_entropy_slope_8",
    "diff_normalized_motion_residual_t", "diff_velocity_norm_t",
    "diff_acceleration_norm_t", "diff_scale_change_t",
)

MOTION_NAMES = (
    "velocity_x_t", "velocity_y_t", "velocity_norm_t",
    "acceleration_x_t", "acceleration_y_t", "acceleration_norm_t",
    "normalized_motion_residual_t", "log_area_scale_change_t",
    "log_area_slope_4", "log_area_slope_8",
    "log_area_variance_4", "log_area_variance_8",
)
GROUP_C = tuple(
    "{}_{}".format(side, name)
    for side in ("sender", "receiver") for name in MOTION_NAMES)

FEATURE_GROUPS = {
    "A": GROUP_A,
    "B": GROUP_B,
    "C": GROUP_C,
    "D": GROUP_D,
    "E": GROUP_E,
}
FEATURE_SETS = {
    "T0": GROUP_A,
    "T1": GROUP_A + GROUP_B,
    "T2": GROUP_A + GROUP_B + GROUP_D,
    "T3": GROUP_A + GROUP_B + GROUP_D + GROUP_E,
    "T4": GROUP_A + GROUP_B + GROUP_D + GROUP_E + GROUP_C,
}


def _source_local_maps(rows):
    local = {}
    both = {}
    groups = defaultdict(set)
    for row in rows:
        if str(row.get("uses_gt", "")).lower() not in ("false", "0"):
            raise RuntimeError("E1.5 source contains uses_gt=true")
        key = (row["target_id"], row["receiver_view"], int(row["frame_id"]))
        groups[key].add(row["branch_name"])
        if row["branch_name"] == "local":
            local[key] = row
        elif row["branch_name"] == "both":
            both[key] = row
    expected = {"local", "sender0_only", "sender1_only", "both"}
    bad = [key for key, branches in groups.items() if branches != expected]
    if bad:
        raise RuntimeError("E1.5 source has incomplete branch groups: {}".format(
            bad[:5]))
    if len(local) != len(groups) or len(both) != len(groups):
        raise RuntimeError("E1.5 local/both map is incomplete")
    return local, both


def _base_sender_records(rows):
    local_map, both_map = _source_local_maps(rows)
    records = []
    branch_to_slot = {"sender0_only": 0, "sender1_only": 1}
    for row in rows:
        if row["branch_name"] not in branch_to_slot:
            continue
        slot = branch_to_slot[row["branch_name"]]
        sender_view = row["sender_0_view"]
        frame_id = int(row["frame_id"])
        receiver_key = (row["target_id"], row["receiver_view"], frame_id)
        sender_key = (row["target_id"], sender_view, frame_id)
        if sender_key not in local_map:
            raise RuntimeError("sender local candidate is missing: {}".format(
                sender_key))
        sender = local_map[sender_key]
        receiver = local_map[receiver_key]
        both = both_map[receiver_key]
        if frame_id > 0:
            for raw_name, sender_name in (
                    ("sender_0_max_score", "local_max_score"),
                    ("sender_0_apce", "local_apce"),
                    ("sender_0_entropy", "local_entropy")):
                left, right = _float(row, raw_name), _float(sender, sender_name)
                if not (math.isfinite(left) and math.isfinite(right)
                        and abs(left - right) <= 1e-5):
                    raise RuntimeError(
                        "sender provenance mismatch {} {}".format(
                            receiver_key, raw_name))
        records.append({
            "sequence_name": row["sequence_name"],
            "target_id": row["target_id"],
            "frame_id": frame_id,
            "receiver_view": row["receiver_view"],
            "sender_view": sender_view,
            "sender_slot": slot,
            "uses_gt": False,
            "source_prediction_sha256": SOURCE_PREDICTION_SHA256,
            "sender_score": _float(sender, "local_max_score"),
            "sender_apce": _float(sender, "local_apce"),
            "sender_entropy": _float(sender, "local_entropy"),
            "sender_center_motion": _float(sender, "local_center_motion"),
            "sender_scale_change": _float(sender, "local_scale_change"),
            "receiver_score": _float(receiver, "local_max_score"),
            "receiver_apce": _float(receiver, "local_apce"),
            "receiver_entropy": _float(receiver, "local_entropy"),
            "receiver_center_motion": _float(
                receiver, "local_center_motion"),
            "receiver_scale_change": _float(receiver, "local_scale_change"),
            "sender_bbox": _bbox(sender, "local"),
            "receiver_bbox": _bbox(receiver, "local"),
            "candidate_bbox": _bbox(row, "branch"),
            "both_bbox": _bbox(both, "branch"),
        })
    return records


def _metric_temporal_features(output, prefix, metrics, frames, index):
    for metric_name, values in metrics.items():
        output["{}_{}_t".format(prefix, metric_name)] = values[index]
        for window in WINDOWS:
            summary = temporal_summary(values, frames, index, window)
            for statistic in ("mean", "std", "slope"):
                output["{}_{}_{}_{}".format(
                    prefix, metric_name, statistic, window)] = summary[statistic]
        output["{}_{}_delta_1".format(prefix, metric_name)] = _delta_one(
            values, index)


def build_temporal_features(source_rows):
    """Build branch-level temporal rows without reading labels or GT."""
    base = _base_sender_records(source_rows)
    grouped = defaultdict(list)
    for record in base:
        grouped[(record["target_id"], record["receiver_view"],
                 record["sender_view"])].append(record)
    output = []
    for history_key, history in sorted(grouped.items()):
        history.sort(key=lambda item: item["frame_id"])
        frames = [item["frame_id"] for item in history]
        if frames != list(range(frames[0], frames[-1] + 1)):
            raise RuntimeError("non-contiguous history {}".format(history_key))
        sender_metrics = {
            name: [item["sender_" + name] for item in history]
            for name in ("score", "apce", "entropy")}
        receiver_metrics = {
            name: [item["receiver_" + name] for item in history]
            for name in ("score", "apce", "entropy")}
        sender_boxes = [item["sender_bbox"] for item in history]
        receiver_boxes = [item["receiver_bbox"] for item in history]
        sender_motion = motion_series(sender_boxes)
        receiver_motion = motion_series(receiver_boxes)
        for index, item in enumerate(history):
            row = {
                name: item[name] for name in (
                    "sequence_name", "target_id", "frame_id",
                    "receiver_view", "sender_view", "sender_slot", "uses_gt",
                    "source_prediction_sha256")}
            for name, box in (
                    ("local", item["receiver_bbox"]),
                    ("sender", item["sender_bbox"]),
                    ("candidate", item["candidate_bbox"]),
                    ("both", item["both_bbox"])):
                for offset, field in enumerate(("x", "y", "w", "h")):
                    row["{}_bbox_{}".format(name, field)] = float(box[offset])
            row.update({
                "sender_score_t": item["sender_score"],
                "sender_apce_t": item["sender_apce"],
                "sender_entropy_t": item["sender_entropy"],
                "sender_bbox_width_t": float(item["sender_bbox"][2]),
                "sender_bbox_height_t": float(item["sender_bbox"][3]),
                "sender_center_motion_t": item["sender_center_motion"],
                "sender_scale_change_t": item["sender_scale_change"],
                "receiver_score_t": item["receiver_score"],
                "receiver_apce_t": item["receiver_apce"],
                "receiver_entropy_t": item["receiver_entropy"],
                "receiver_bbox_width_t": float(item["receiver_bbox"][2]),
                "receiver_bbox_height_t": float(item["receiver_bbox"][3]),
                "receiver_center_motion_t": item["receiver_center_motion"],
                "receiver_scale_change_t": item["receiver_scale_change"],
            })
            _metric_temporal_features(
                row, "sender", sender_metrics, frames, index)
            _metric_temporal_features(
                row, "receiver", receiver_metrics, frames, index)
            for side, values in (
                    ("sender", sender_motion),
                    ("receiver", receiver_motion)):
                for name in (
                        "velocity_x", "velocity_y", "velocity_norm",
                        "acceleration_x", "acceleration_y",
                        "acceleration_norm", "normalized_motion_residual",
                        "log_area_scale_change"):
                    row["{}_{}_t".format(side, name)] = values[name][index]
                for window in WINDOWS:
                    log_area = temporal_summary(
                        values["log_area"], frames, index, window)
                    row["{}_log_area_slope_{}".format(
                        side, window)] = log_area["slope"]
                    row["{}_log_area_variance_{}".format(
                        side, window)] = log_area["std"] ** 2
            for metric in ("score", "apce", "entropy"):
                row["diff_{}_t".format(metric)] = (
                    row["sender_{}_t".format(metric)]
                    - row["receiver_{}_t".format(metric)])
                row["diff_{}_mean_8".format(metric)] = (
                    row["sender_{}_mean_8".format(metric)]
                    - row["receiver_{}_mean_8".format(metric)])
                row["diff_{}_slope_8".format(metric)] = (
                    row["sender_{}_slope_8".format(metric)]
                    - row["receiver_{}_slope_8".format(metric)])
            row["diff_normalized_motion_residual_t"] = (
                row["sender_normalized_motion_residual_t"]
                - row["receiver_normalized_motion_residual_t"])
            row["diff_velocity_norm_t"] = (
                row["sender_velocity_norm_t"]
                - row["receiver_velocity_norm_t"])
            row["diff_acceleration_norm_t"] = (
                row["sender_acceleration_norm_t"]
                - row["receiver_acceleration_norm_t"])
            row["diff_scale_change_t"] = (
                row["sender_log_area_scale_change_t"]
                - row["receiver_log_area_scale_change_t"])
            output.append(row)
    output.sort(key=lambda row: (
        row["target_id"], row["receiver_view"], int(row["frame_id"]),
        int(row["sender_slot"])))
    return output


def _feature_definition_rows():
    descriptions = {
        "A": "sender current-frame prediction",
        "B": "sender causal score/APCE/entropy prefix",
        "C": "bbox-derived sender/receiver causal motion",
        "D": "receiver current and causal reliability prefix",
        "E": "sender minus receiver causal stability",
    }
    rows = []
    for group in ("A", "B", "D", "E", "C"):
        for name in FEATURE_GROUPS[group]:
            rows.append({
                "feature_group": group,
                "feature_name": name,
                "available": True,
                "prediction_only": True,
                "causal": True,
                "window": 8 if "_8" in name else (4 if "_4" in name else 1),
                "definition": descriptions[group],
                "missing_reason": "",
            })
    for name, reason in (
            ("response_top1_top2_gap", "response map absent from E1.5 freeze"),
            ("response_peak_sharpness", "response map absent from E1.5 freeze"),
            ("bbox_border_proximity", "image dimensions absent from E1.5 freeze"),
            ("search_region_border_proximity", "response grid absent from E1.5 freeze"),
            ("target_consistency", "dense/compact target feature absent; future E2B")):
        rows.append({
            "feature_group": "F" if name == "target_consistency" else "C",
            "feature_name": name,
            "available": False,
            "prediction_only": True,
            "causal": True,
            "window": 8,
            "definition": "not constructed",
            "missing_reason": reason,
        })
    return rows


def freeze_features(source_predictions, source_manifest, output_dir):
    source_predictions = _resolve_path(source_predictions)
    source_manifest = _resolve_path(source_manifest)
    output_dir = _resolve_path(output_dir)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_sha = sha256_file(source_predictions)
    if source_sha != SOURCE_PREDICTION_SHA256 \
            or source_sha != manifest["prediction_sha256"]:
        raise RuntimeError("E1.5 frozen prediction SHA256 mismatch")
    with source_predictions.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    forbidden = {
        "target_visible", "iou", "delta_iou", "label", "ground_truth", "gt"}
    source_fields = set(source_rows[0]) if source_rows else set()
    if forbidden.intersection(source_fields):
        raise RuntimeError("source prediction schema contains post-hoc fields")
    temporal_rows = build_temporal_features(source_rows)
    if len(temporal_rows) != 28224:
        raise RuntimeError("expected 28,224 sender temporal rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "temporal_prediction_only_features.csv"
    write_csv(feature_path, temporal_rows)
    write_csv(output_dir / "temporal_feature_definition.csv",
              _feature_definition_rows())
    temporal_manifest = {
        "schema_version": 1,
        "phase": "temporal_prediction_freeze_before_gt_join",
        "uses_gt": False,
        "causal_history_key": ["target_id", "receiver_view", "sender_view"],
        "primary_window": 8,
        "secondary_window": 4,
        "source_prediction_file": _manifest_path(source_predictions),
        "source_prediction_manifest": _manifest_path(source_manifest),
        "source_prediction_sha256": source_sha,
        "temporal_prediction_file": _manifest_path(feature_path),
        "temporal_prediction_rows": len(temporal_rows),
        "temporal_prediction_sha256": sha256_file(feature_path),
        "target_count": len({row["target_id"] for row in temporal_rows}),
        "receiver_frame_groups": len({
            (row["target_id"], row["receiver_view"], row["frame_id"])
            for row in temporal_rows}),
        "sender_histories": len({
            (row["target_id"], row["receiver_view"], row["sender_view"])
            for row in temporal_rows}),
        "feature_sets": {
            name: list(features) for name, features in FEATURE_SETS.items()},
        "target_consistency_available": False,
        "tracker_rollout_repeated": False,
    }
    (output_dir / "temporal_prediction_manifest.json").write_text(
        json.dumps(temporal_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(temporal_manifest, indent=2, sort_keys=True))


def _pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline((
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            random_state=42, solver="liblinear")),
    ))


def _matrix(rows, names):
    return np.asarray([[_float(row, name) for name in names] for row in rows],
                      dtype=float)


def loto_probabilities(rows, names, labels, predict_rows=None):
    """Fit on four targets and return target-held-out probabilities."""
    if predict_rows is None:
        predict_rows = rows
    x_train_all = _matrix(rows, names)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray([row["target_id"] for row in rows])
    predict_groups = np.asarray([row["target_id"] for row in predict_rows])
    x_predict = _matrix(predict_rows, names)
    output = np.full(len(predict_rows), np.nan, dtype=float)
    for target in sorted(set(predict_groups)):
        train_mask = groups != target
        test_mask = predict_groups == target
        if len(set(y[train_mask])) < 2:
            continue
        if not bool(np.isfinite(x_train_all[train_mask]).any(axis=0).any()):
            continue
        model = _pipeline()
        model.fit(x_train_all[train_mask], y[train_mask])
        output[test_mask] = model.predict_proba(x_predict[test_mask])[:, 1]
    return output


def _metric_row(task, feature_set, scope, labels, probabilities,
                eligible_rows, active_rows, feature_count):
    from sklearn.metrics import (
        average_precision_score, confusion_matrix, precision_score,
        recall_score, roc_auc_score)
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(probabilities)
    labels, probabilities = labels[valid], probabilities[valid]
    predicted = probabilities >= 0.5
    if len(labels):
        tn, fp, fn, tp = confusion_matrix(
            labels, predicted, labels=[0, 1]).reshape(-1)
    else:
        tn = fp = fn = tp = 0
    return {
        "task": task,
        "feature_set": feature_set,
        "scope": scope,
        "feature_count": feature_count,
        "cv": "leave_one_target_out",
        "target_folds": 5,
        "eligible_rows": int(eligible_rows),
        "oof_rows": int(len(labels)),
        "label_coverage": (float(eligible_rows) / active_rows
                           if active_rows else float("nan")),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "roc_auc": (float(roc_auc_score(labels, probabilities))
                    if len(set(labels)) == 2 else float("nan")),
        "pr_auc": (float(average_precision_score(labels, probabilities))
                   if len(set(labels)) == 2 else float("nan")),
        "precision_at_0_5": float(precision_score(
            labels, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(
            labels, predicted, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "uses_gt_at_decision": False,
        "posthoc_diagnostic_only": True,
    }


def _scoped_metrics(task, feature_set, rows, labels, probabilities,
                    active_rows, feature_count):
    output = [_metric_row(
        task, feature_set, "overall", labels, probabilities,
        len(rows), len(active_rows), feature_count)]
    for view in VIEWS:
        mask = np.asarray([row["receiver_view"] == view for row in rows])
        active_view_count = sum(
            row["receiver_view"] == view for row in active_rows)
        output.append(_metric_row(
            task, feature_set, "receiver_{}".format(view),
            np.asarray(labels)[mask], np.asarray(probabilities)[mask],
            int(mask.sum()), active_view_count, feature_count))
    return output


def _join_labels(feature_rows, label_rows, temporal_sha):
    label_map = {
        (row["target_id"], row["receiver_view"], int(row["frame_id"])): row
        for row in label_rows}
    output = []
    for row in feature_rows:
        key = (row["target_id"], row["receiver_view"], int(row["frame_id"]))
        label = label_map[key]
        slot = int(row["sender_slot"])
        output.append({
            **row,
            "sequence_name": row["sequence_name"],
            "target_id": row["target_id"],
            "receiver_view": row["receiver_view"],
            "sender_view": row["sender_view"],
            "sender_slot": slot,
            "frame_id": int(row["frame_id"]),
            "target_visible": _bool(label["target_visible"]),
            "valid_for_analysis": _bool(label["valid_for_analysis"]),
            "delta_iou": float(label["delta_iou_sender{}".format(slot)]),
            "label": label["label_sender{}".format(slot)],
            "remote_help_available": _bool(label["remote_help_available"]),
            "temporal_prediction_sha256": temporal_sha,
            "source_prediction_sha256": row["source_prediction_sha256"],
        })
    return output


def _frame_records(joined_rows):
    grouped = defaultdict(list)
    for row in joined_rows:
        grouped[(row["target_id"], row["receiver_view"],
                 int(row["frame_id"]))].append(row)
    frames = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["sender_slot"]))
        if [int(row["sender_slot"]) for row in rows] != [0, 1]:
            raise RuntimeError("frame does not contain two sender slots")
        frames.append({
            "target_id": key[0], "receiver_view": key[1], "frame_id": key[2],
            "sequence_name": rows[0]["sequence_name"],
            "sender0": rows[0], "sender1": rows[1],
            "valid_for_analysis": bool(rows[0]["valid_for_analysis"]),
            "remote_help_available": bool(rows[0]["remote_help_available"]),
        })
    return frames


def _aggregate_frame_features(frames, names):
    aggregate_names = []
    for name in names:
        if name.startswith("receiver_"):
            aggregate_names.append(name)
        else:
            aggregate_names.extend(
                "{}_{}".format(name, statistic)
                for statistic in ("min", "max", "mean", "absdiff"))
    rows = []
    for frame in frames:
        row = {
            "target_id": frame["target_id"],
            "receiver_view": frame["receiver_view"],
            "frame_id": frame["frame_id"],
        }
        for name in names:
            first = _float(frame["sender0"], name)
            second = _float(frame["sender1"], name)
            if name.startswith("receiver_"):
                row[name] = first
            else:
                finite = math.isfinite(first) and math.isfinite(second)
                row[name + "_min"] = min(first, second) if finite else float("nan")
                row[name + "_max"] = max(first, second) if finite else float("nan")
                row[name + "_mean"] = 0.5 * (first + second) if finite else float("nan")
                row[name + "_absdiff"] = abs(first - second) if finite else float("nan")
        rows.append(row)
    return rows, tuple(aggregate_names)


def _ranking_probabilities(frames, names, with_identity=False):
    eligible = [frame for frame in frames
                if frame["valid_for_analysis"] and frame["frame_id"] > 0
                and abs(float(frame["sender0"]["delta_iou"])
                        - float(frame["sender1"]["delta_iou"])) > 0.02]
    base_names = tuple("diff_{}".format(index) for index in range(len(names)))
    identity_names = tuple(
        "{}_{}".format(role, view)
        for role in ("receiver", "sender0", "sender1") for view in VIEWS)
    feature_names = base_names + (identity_names if with_identity else ())
    rows = []
    labels = []
    raw_identity = []
    for frame in eligible:
        row = {"target_id": frame["target_id"],
               "receiver_view": frame["receiver_view"]}
        for index, name in enumerate(names):
            row["diff_{}".format(index)] = (
                _float(frame["sender0"], name)
                - _float(frame["sender1"], name))
        if with_identity:
            sender0_view = frame["sender0"]["sender_view"]
            sender1_view = frame["sender1"]["sender_view"]
            for view in VIEWS:
                row["receiver_" + view] = int(frame["receiver_view"] == view)
                row["sender0_" + view] = int(sender0_view == view)
                row["sender1_" + view] = int(sender1_view == view)
            raw_identity.append((frame["receiver_view"], sender0_view, sender1_view))
        rows.append(row)
        labels.append(int(float(frame["sender0"]["delta_iou"])
                          > float(frame["sender1"]["delta_iou"])))
    x = _matrix(rows, feature_names)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray([row["target_id"] for row in rows])
    oof = np.full(len(rows), np.nan, dtype=float)
    for target in sorted(set(groups)):
        train = groups != target
        test = groups == target
        train_x = x[train]
        train_y = y[train]
        swapped = train_x.copy()
        swapped[:, :len(base_names)] *= -1.0
        if with_identity:
            offset = len(base_names)
            sender0 = swapped[:, offset + 3:offset + 6].copy()
            swapped[:, offset + 3:offset + 6] = swapped[:, offset + 6:offset + 9]
            swapped[:, offset + 6:offset + 9] = sender0
        augmented_x = np.concatenate((train_x, swapped), axis=0)
        augmented_y = np.concatenate((train_y, 1 - train_y), axis=0)
        model = _pipeline()
        model.fit(augmented_x, augmented_y)
        oof[test] = model.predict_proba(x[test])[:, 1]
    return eligible, labels, oof, len(feature_names)


def policy_choice(probability0, probability1, frame_id):
    if int(frame_id) == 0 or not (
            math.isfinite(float(probability0))
            and math.isfinite(float(probability1))):
        return "local"
    probability0, probability1 = float(probability0), float(probability1)
    if max(probability0, probability1) < 0.5:
        return "local"
    if abs(probability0 - probability1) <= 1e-12:
        return "local"
    return "sender0_only" if probability0 > probability1 else "sender1_only"


def _policy_predictions(joined_rows, feature_set):
    active_train = [row for row in joined_rows
                    if row["valid_for_analysis"] and int(row["frame_id"]) > 0]
    labels = [int(row["label"] == "helpful") for row in active_train]
    predict = list(joined_rows)
    probabilities = loto_probabilities(
        active_train, FEATURE_SETS[feature_set], labels, predict_rows=predict)
    probability_map = {
        (row["target_id"], row["receiver_view"], int(row["frame_id"]),
         int(row["sender_slot"])): probability
        for row, probability in zip(predict, probabilities)}
    frames = _frame_records(joined_rows)
    output = []
    for frame in frames:
        key = (frame["target_id"], frame["receiver_view"], frame["frame_id"])
        p0 = probability_map[key + (0,)]
        p1 = probability_map[key + (1,)]
        output.append({
            **frame,
            "probability_sender0": p0,
            "probability_sender1": p1,
            "selected_branch": policy_choice(p0, p1, frame["frame_id"]),
        })
    return output


def _box(row, prefix):
    return np.asarray([
        _float(row, "{}_bbox_{}".format(prefix, name))
        for name in ("x", "y", "w", "h")], dtype=float).astype(int).astype(float)


def _tracking_metrics(joined_rows, policies, dataset):
    from lib.test.analysis.fcvc_results import _curves
    sequence_map = {sequence.name: sequence for sequence in dataset}
    frame_map = {
        (row["target_id"], row["receiver_view"], int(row["frame_id"]),
         int(row["sender_slot"])): row for row in joined_rows}
    policy_maps = {
        name: {(row["target_id"], row["receiver_view"], row["frame_id"]): row
               for row in values}
        for name, values in policies.items()}
    groups = defaultdict(list)
    for key in sorted({item[:3] for item in frame_map}):
        first, second = frame_map[key + (0,)], frame_map[key + (1,)]
        groups[first["sequence_name"]].append((key, first, second))
    variants = (
        "local", "both", "sender0_only", "sender1_only",
        "policy_T0", "policy_T1", "policy_T2", "policy_T3", "policy_T4",
        "oracle_single")
    sequence_rows = []
    for sequence_name, frames in sorted(groups.items()):
        sequence = sequence_map[sequence_name]
        target = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        visible_data = getattr(sequence, "target_visible", None)
        visible = (None if visible_data is None else
                   np.asarray(visible_data).reshape(-1).astype(bool))
        frames.sort(key=lambda item: item[0][2])
        predictions = {name: [] for name in variants}
        for key, first, second in frames:
            local = _box(first, "local")
            sender0 = _box(first, "candidate")
            sender1 = _box(second, "candidate")
            both = _box(first, "both")
            candidates = {
                "local": local, "sender0_only": sender0,
                "sender1_only": sender1, "both": both}
            predictions["local"].append(local)
            predictions["both"].append(both)
            predictions["sender0_only"].append(sender0)
            predictions["sender1_only"].append(sender1)
            for name in ("T0", "T1", "T2", "T3", "T4"):
                choice = policy_maps[name][key]["selected_branch"]
                predictions["policy_" + name].append(candidates[choice])
            frame_id = key[2]
            is_visible = True if visible is None else bool(visible[frame_id])
            if is_visible:
                oracle = max(
                    (local, sender0, sender1),
                    key=lambda box: iou_xywh(box, target[frame_id]))
            else:
                oracle = local
            predictions["oracle_single"].append(oracle)
        for variant, boxes in predictions.items():
            metrics = _curves(
                np.asarray(boxes), target, target_visible=visible_data,
                dataset=sequence.dataset)
            sequence_rows.append({
                "sequence_name": sequence_name,
                "target_id": sequence_name.rsplit("-", 1)[0],
                "receiver_view": {"1": "A", "2": "B", "3": "C"}[
                    sequence_name.rsplit("-", 1)[1]],
                "variant": variant,
                **metrics,
            })
    return sequence_rows


def _macro_metrics(sequence_rows, group_field=None):
    groups = ["all"] if group_field is None else sorted({
        row[group_field] for row in sequence_rows})
    variants = []
    for row in sequence_rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])
    output = []
    for group in groups:
        selected_group = [row for row in sequence_rows
                          if group_field is None or row[group_field] == group]
        local_auc = float(np.mean([
            row["auc"] for row in selected_group if row["variant"] == "local"]))
        for variant in variants:
            selected = [row for row in selected_group
                        if row["variant"] == variant]
            result = {
                (group_field or "scope"): group,
                "variant": variant,
                "sequence_count": len(selected),
            }
            for metric in ("auc", "precision", "normalized_precision", "mean_iou"):
                result[metric] = float(np.mean([row[metric] for row in selected]))
            result["auc_delta_vs_local"] = result["auc"] - local_auc
            result["primary_policy"] = variant == "policy_T4"
            result["gt_oracle"] = variant == "oracle_single"
            output.append(result)
    return output


def analyze(output_dir, labels_path, dataset_name):
    if dataset_name.lower() in FORBIDDEN_DATASETS \
            or "test" in dataset_name.lower():
        raise RuntimeError("official/outer test analysis is forbidden")
    from lib.test.evaluation import get_dataset

    output_dir = _resolve_path(output_dir)
    labels_path = _resolve_path(labels_path)
    manifest = json.loads((
        output_dir / "temporal_prediction_manifest.json").read_text(
            encoding="utf-8"))
    feature_path = _resolve_path(manifest["temporal_prediction_file"])
    temporal_sha = sha256_file(feature_path)
    if temporal_sha != manifest["temporal_prediction_sha256"]:
        raise RuntimeError("temporal feature SHA256 mismatch; GT join refused")
    with feature_path.open(newline="", encoding="utf-8") as handle:
        feature_rows = list(csv.DictReader(handle))
    with labels_path.open(newline="", encoding="utf-8") as handle:
        label_rows = list(csv.DictReader(handle))
    source_shas = {row["prediction_sha256"] for row in label_rows}
    if source_shas != {manifest["source_prediction_sha256"]}:
        raise RuntimeError("post-hoc labels do not match frozen E1.5 prediction")
    joined = _join_labels(feature_rows, label_rows, temporal_sha)
    label_fields = (
        "sequence_name", "target_id", "receiver_view", "sender_view",
        "sender_slot", "frame_id", "target_visible", "valid_for_analysis",
        "delta_iou", "label", "remote_help_available",
        "temporal_prediction_sha256", "source_prediction_sha256")
    write_csv(output_dir / "temporal_posthoc_labels.csv", [
        {name: row[name] for name in label_fields} for row in joined])
    active = [row for row in joined
              if row["valid_for_analysis"] and int(row["frame_id"]) > 0]
    non_tie = [row for row in active if row["label"] in ("helpful", "harmful")]

    ablation = []
    task_a_probabilities = {}
    task_a_labels = np.asarray([int(row["label"] == "helpful")
                                for row in non_tie], dtype=int)
    for name, features in FEATURE_SETS.items():
        probabilities = loto_probabilities(
            non_tie, features, task_a_labels)
        task_a_probabilities[name] = probabilities
        ablation.extend(_scoped_metrics(
            "A_helpful_vs_harmful", name, non_tie, task_a_labels,
            probabilities, active, len(features)))
    write_csv(output_dir / "temporal_ablation_group_cv.csv", ablation)

    single_feature_rows = []
    for group in ("A", "B", "D", "E", "C"):
        for feature in FEATURE_GROUPS[group]:
            probabilities = loto_probabilities(
                non_tie, (feature,), task_a_labels)
            result = _metric_row(
                "A_helpful_vs_harmful", feature, "overall",
                task_a_labels, probabilities, len(non_tie), len(active), 1)
            result["feature_group"] = group
            single_feature_rows.append(result)
    write_csv(output_dir / "temporal_single_feature_analysis.csv",
              single_feature_rows)

    frames = [frame for frame in _frame_records(joined)
              if frame["valid_for_analysis"] and frame["frame_id"] > 0]
    remote_help_results = []
    for name, features in FEATURE_SETS.items():
        aggregated, aggregate_names = _aggregate_frame_features(frames, features)
        labels = np.asarray([int(frame["remote_help_available"])
                             for frame in frames], dtype=int)
        probabilities = loto_probabilities(aggregated, aggregate_names, labels)
        remote_help_results.extend(_scoped_metrics(
            "B_remote_help_available", name, aggregated, labels,
            probabilities, aggregated, len(aggregate_names)))
    write_csv(output_dir / "remote_help_group_cv.csv", remote_help_results)

    ranking_results = []
    for name, features in FEATURE_SETS.items():
        for with_identity in ((False, True) if name == "T4" else (False,)):
            eligible, labels, probabilities, feature_count = \
                _ranking_probabilities(frames, features, with_identity)
            ranking_results.extend(_scoped_metrics(
                "C_sender_ranking", name + (
                    "_with_pair_identity" if with_identity else "_no_identity"),
                eligible, labels, probabilities, frames, feature_count))
    write_csv(output_dir / "sender_ranking_group_cv.csv", ranking_results)

    all_frames = _frame_records(joined)
    policies = {
        name: _policy_predictions(joined, name) for name in FEATURE_SETS}
    primary_rows = []
    for index, frame in enumerate(policies["T4"]):
        row = {
            "sequence_name": frame["sequence_name"],
            "target_id": frame["target_id"],
            "receiver_view": frame["receiver_view"],
            "frame_id": frame["frame_id"],
            "sender0_view": frame["sender0"]["sender_view"],
            "sender1_view": frame["sender1"]["sender_view"],
            "probability_sender0": frame["probability_sender0"],
            "probability_sender1": frame["probability_sender1"],
            "selected_branch": frame["selected_branch"],
            "uses_gt_at_decision": False,
            "safe_report_only": True,
            "feature_set": "T4_preregistered_primary",
            "threshold": 0.5,
            "temporal_prediction_sha256": temporal_sha,
        }
        source = frame["sender0"]
        for prefix in ("local", "candidate", "both"):
            output_prefix = (
                "sender0" if prefix == "candidate" else prefix)
            for field in ("x", "y", "w", "h"):
                row["{}_bbox_{}".format(output_prefix, field)] = _float(
                    source, "{}_bbox_{}".format(prefix, field))
        for field in ("x", "y", "w", "h"):
            row["sender1_bbox_" + field] = _float(
                frame["sender1"], "candidate_bbox_" + field)
        for descriptive in ("T0", "T1", "T2", "T3"):
            row["selected_branch_" + descriptive] = policies[descriptive][
                index]["selected_branch"]
        primary_rows.append(row)
    write_csv(output_dir / "oof_policy_predictions.csv", primary_rows)

    dataset = get_dataset(dataset_name)
    sequence_rows = _tracking_metrics(joined, policies, dataset)
    summary = _macro_metrics(sequence_rows)
    per_view = _macro_metrics(sequence_rows, "receiver_view")
    per_target = _macro_metrics(sequence_rows, "target_id")
    write_csv(output_dir / "oof_tracking_summary.csv", summary)
    write_csv(output_dir / "oof_per_view.csv", per_view)
    write_csv(output_dir / "oof_per_target.csv", per_target)

    def auc(variant):
        return next(row["auc"] for row in summary if row["variant"] == variant)
    local_auc = auc("local")
    selector_auc = auc("policy_T4")
    oracle_auc = auc("oracle_single")
    gain = selector_auc - local_auc
    headroom = oracle_auc - local_auc
    target_policy = [row for row in per_target if row["variant"] == "policy_T4"]
    max_negative = min(row["auc_delta_vs_local"] for row in target_policy)
    task_a_t4 = next(row for row in ablation
                     if row["feature_set"] == "T4"
                     and row["scope"] == "overall")
    gates = {
        "primary_auc_gain_ge_0_003": gain >= 0.003,
        "secondary_roc_auc_ge_0_65": task_a_t4["roc_auc"] >= 0.65,
        "secondary_pr_above_prior": (
            task_a_t4["pr_auc"] > task_a_t4["positive_rate"]),
        "safety_no_target_le_minus_0_05": max_negative > -0.05,
    }
    decision_case = (
        "A" if all(gates.values()) else
        "B" if task_a_t4["roc_auc"] >= 0.65 and gain < 0.003 else
        "C" if task_a_t4["roc_auc"] < 0.65 and gain < 0.003 else
        "MIXED")
    utilization = {
        "dataset": dataset_name,
        "local_auc": local_auc,
        "both_safe_auc": auc("both"),
        "fixed_sender0_auc": auc("sender0_only"),
        "fixed_sender1_auc": auc("sender1_only"),
        "oof_temporal_selector_auc": selector_auc,
        "gt_oracle_single_auc": oracle_auc,
        "selector_gain": gain,
        "oracle_headroom": headroom,
        "oracle_utilization": gain / headroom if headroom else float("nan"),
        "max_per_target_negative_delta": max_negative,
        "task_a_t4_roc_auc": task_a_t4["roc_auc"],
        "task_a_t4_pr_auc": task_a_t4["pr_auc"],
        "task_a_t4_positive_rate": task_a_t4["positive_rate"],
        "preregistered_gates": gates,
        "decision_case": decision_case,
        "gt_oracle_upper_bound_only": True,
        "communication_trigger": False,
    }
    (output_dir / "oracle_utilization.json").write_text(
        json.dumps(utilization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    analysis_manifest = {
        "phase": "posthoc_gt_join_loto_oof_safe_report",
        "dataset": dataset_name,
        "temporal_prediction_sha256_verified": temporal_sha,
        "temporal_prediction_rows": len(feature_rows),
        "posthoc_sender_rows": len(joined),
        "active_sender_rows": len(active),
        "non_tie_sender_rows": len(non_tie),
        "receiver_frames": len(all_frames),
        "active_receiver_frames": len(frames),
        "target_folds": 5,
        "tracker_rollout_repeated": False,
        "official_test_used": False,
    }
    (output_dir / "posthoc_analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({**analysis_manifest, **utilization}, indent=2,
                     sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--source-predictions", required=True)
    freeze.add_argument("--source-manifest", required=True)
    freeze.add_argument("--output-dir", required=True)
    posthoc = commands.add_parser("analyze")
    posthoc.add_argument("--output-dir", required=True)
    posthoc.add_argument("--posthoc-labels", required=True)
    posthoc.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_features(
            args.source_predictions, args.source_manifest, args.output_dir)
    else:
        analyze(args.output_dir, args.posthoc_labels, args.dataset)


if __name__ == "__main__":
    main()
