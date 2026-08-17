"""Utilities for offline PCUM reliability-selector experiments."""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation.datasets import get_dataset
from lib.test.utils.load_text import load_text


BASE_FEATURE_COLUMNS = [
    "local_score",
    "collab_score",
    "local_apce",
    "collab_apce",
    "score_delta",
    "apce_delta",
    "remote_quality",
    "remote_weight_entropy",
    "remote_max_weight",
    "motion_consistency",
    "bbox_area_change",
    "center_displacement",
]

ENHANCED_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "local_response_max",
    "collab_response_max",
    "local_collab_bbox_iou",
    "local_collab_center_distance_norm",
    "local_collab_area_ratio_log",
    "local_collab_score_ratio",
    "local_collab_apce_ratio",
    "local_center_velocity",
    "collab_center_velocity",
    "local_velocity_change",
    "collab_velocity_change",
    "local_scale_change",
    "collab_scale_change",
    "local_temporal_smoothness",
    "collab_temporal_smoothness",
    "temporal_smoothness_delta",
    "remote_weight_entropy_delta",
    "remote_max_weight_delta",
    "remote_quality_delta",
    "selected_remote_uav_changed",
]

FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)

FORBIDDEN_FEATURE_NAMES = {
    "target_visible",
    "gt_visible",
    "visibility",
    "annotation_visibility",
    "gt_iou",
    "test_iou",
    "oracle_iou",
    "oracle",
    "test_gt",
}

THRESHOLD_OVERLAP = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
THRESHOLD_CENTER = torch.arange(0, 51, dtype=torch.float64)
THRESHOLD_CENTER_NORM = torch.arange(0, 51, dtype=torch.float64) / 100.0


def result_dir(root, tracker, runid):
    return Path(root) / "output" / "test" / "tracking_results" / "entertrack" / "{}_{:03d}".format(tracker, int(runid))


def view_name(seq_name):
    suffix = seq_name.rsplit("-", 1)[-1]
    return {"1": "Drone A", "2": "Drone B", "3": "Drone C"}.get(suffix, "Unknown")


def load_bbox_file(path):
    data = load_text(str(path), delimiter=("\t", ","), dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :4]


def load_vector_file(path, length, default=0.0):
    path = Path(path)
    if not path.is_file():
        return np.full((length,), float(default), dtype=np.float64)
    data = load_text(str(path), delimiter=("\t", ","), dtype=np.float64)
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if data.size < length:
        data = np.pad(data, (0, length - data.size), constant_values=float(default))
    return data[:length]


def load_remote_weight_file(path, length):
    # Columns written by pcum remote aggregation diagnostics:
    # entropy, max_weight, mean_weight, selected_uav, valid_count,
    # quality_mean, quality_min, quality_max, fallback, weight_A, weight_B, weight_C
    default = np.zeros((length, 12), dtype=np.float64)
    default[:, 0] = math.log(2.0)
    default[:, 1] = 0.5
    default[:, 2] = 0.5
    default[:, 3] = -1
    default[:, 4] = 0
    default[:, 8] = 1
    path = Path(path)
    if not path.is_file():
        return default
    data = np.loadtxt(str(path), delimiter="\t")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 12:
        padded = np.zeros((data.shape[0], 12), dtype=np.float64)
        padded[:, : data.shape[1]] = data
        data = padded
    if data.shape[0] < length:
        data = np.vstack([data, default[data.shape[0] : length]])
    data = data[:length]
    bad = ~np.isfinite(data)
    if bad.any():
        data[bad] = default[bad]
    return data


def box_area(box):
    return max(float(box[2]), 0.0) * max(float(box[3]), 0.0)


def box_center(box):
    return np.array([float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0], dtype=np.float64)


def box_iou_xywh(a, b):
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    ax2, ay2 = ax1 + max(aw, 0.0), ay1 + max(ah, 0.0)
    bx2, by2 = bx1 + max(bw, 0.0), by1 + max(bh, 0.0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def gt_label_loss(pred, gt):
    # GT-derived supervision for offline label construction only.
    iou = box_iou_xywh(pred, gt)
    scale = math.sqrt(max(box_area(gt), 1.0))
    pred_c, gt_c = box_center(pred), box_center(gt)
    center_l1 = float(np.abs(pred_c - gt_c).sum()) / scale
    size_l1 = (abs(float(pred[2]) - float(gt[2])) + abs(float(pred[3]) - float(gt[3]))) / scale
    return (1.0 - iou) + 0.05 * (center_l1 + size_l1)


def prediction_feature_values(local_box, collab_box, prev_local_box, local_score, collab_score,
                              local_apce, collab_apce, remote_weights):
    score_delta = float(collab_score) - float(local_score)
    apce_delta = float(collab_apce) - float(local_apce)
    scale = math.sqrt(max(box_area(local_box), 1.0))
    center_displacement = float(np.linalg.norm(box_center(collab_box) - box_center(local_box))) / scale
    area_ratio = (box_area(collab_box) + 1e-6) / (box_area(local_box) + 1e-6)
    bbox_area_change = abs(math.log(max(area_ratio, 1e-6)))

    if prev_local_box is None:
        motion_consistency = 1.0
    else:
        prev_scale = math.sqrt(max(box_area(prev_local_box), 1.0))
        motion = float(np.linalg.norm(box_center(collab_box) - box_center(prev_local_box))) / prev_scale
        motion_consistency = max(0.0, min(1.0, 1.0 - motion / 2.0))

    return {
        "local_score": float(local_score),
        "collab_score": float(collab_score),
        "local_apce": float(local_apce),
        "collab_apce": float(collab_apce),
        "score_delta": score_delta,
        "apce_delta": apce_delta,
        "remote_quality": float(remote_weights[5]),
        "remote_weight_entropy": float(remote_weights[0]),
        "remote_max_weight": float(remote_weights[1]),
        "motion_consistency": float(motion_consistency),
        "bbox_area_change": float(bbox_area_change),
        "center_displacement": float(center_displacement),
    }


def _center_velocity(box, prev_box):
    if prev_box is None:
        return 0.0
    scale = math.sqrt(max(box_area(prev_box), 1.0))
    return float(np.linalg.norm(box_center(box) - box_center(prev_box))) / scale


def _scale_change(box, prev_box):
    if prev_box is None:
        return 0.0
    ratio = (box_area(box) + 1e-6) / (box_area(prev_box) + 1e-6)
    return abs(math.log(max(ratio, 1e-6)))


def enhanced_prediction_feature_values(local_box, collab_box, prev_local_box, prev_collab_box,
                                       prev2_local_box, prev2_collab_box, local_score, collab_score,
                                       local_apce, collab_apce, remote_weights, prev_remote_weights):
    features = prediction_feature_values(
        local_box, collab_box, prev_local_box, local_score, collab_score,
        local_apce, collab_apce, remote_weights
    )
    local_velocity = _center_velocity(local_box, prev_local_box)
    collab_velocity = _center_velocity(collab_box, prev_collab_box)
    prev_local_velocity = _center_velocity(prev_local_box, prev2_local_box) if prev_local_box is not None else 0.0
    prev_collab_velocity = _center_velocity(prev_collab_box, prev2_collab_box) if prev_collab_box is not None else 0.0
    local_velocity_change = abs(local_velocity - prev_local_velocity)
    collab_velocity_change = abs(collab_velocity - prev_collab_velocity)
    local_scale_change = _scale_change(local_box, prev_local_box)
    collab_scale_change = _scale_change(collab_box, prev_collab_box)
    local_smooth = 1.0 / (1.0 + local_velocity_change + local_scale_change)
    collab_smooth = 1.0 / (1.0 + collab_velocity_change + collab_scale_change)
    selected_changed = 0.0
    if prev_remote_weights is not None:
        selected_changed = 1.0 if int(remote_weights[3]) != int(prev_remote_weights[3]) else 0.0
    features.update({
        # Existing outputs save max response score, not full response maps.
        "local_response_max": float(local_score),
        "collab_response_max": float(collab_score),
        "local_collab_bbox_iou": float(box_iou_xywh(local_box, collab_box)),
        "local_collab_center_distance_norm": features["center_displacement"],
        "local_collab_area_ratio_log": features["bbox_area_change"],
        "local_collab_score_ratio": float(collab_score + 1e-6) / float(local_score + 1e-6),
        "local_collab_apce_ratio": float(collab_apce + 1e-6) / float(local_apce + 1e-6),
        "local_center_velocity": local_velocity,
        "collab_center_velocity": collab_velocity,
        "local_velocity_change": local_velocity_change,
        "collab_velocity_change": collab_velocity_change,
        "local_scale_change": local_scale_change,
        "collab_scale_change": collab_scale_change,
        "local_temporal_smoothness": local_smooth,
        "collab_temporal_smoothness": collab_smooth,
        "temporal_smoothness_delta": collab_smooth - local_smooth,
        "remote_weight_entropy_delta": float(remote_weights[0] - prev_remote_weights[0]) if prev_remote_weights is not None else 0.0,
        "remote_max_weight_delta": float(remote_weights[1] - prev_remote_weights[1]) if prev_remote_weights is not None else 0.0,
        "remote_quality_delta": float(remote_weights[5] - prev_remote_weights[5]) if prev_remote_weights is not None else 0.0,
        "selected_remote_uav_changed": selected_changed,
    })
    return features


def get_feature_columns(feature_set="base"):
    if feature_set == "base":
        return list(BASE_FEATURE_COLUMNS)
    if feature_set == "enhanced":
        return list(ENHANCED_FEATURE_COLUMNS)
    raise ValueError("Unsupported feature_set: {}".format(feature_set))


def validate_feature_columns(feature_columns=FEATURE_COLUMNS):
    lowered = {name.lower() for name in feature_columns}
    forbidden_hits = []
    for name in lowered:
        for forbidden in FORBIDDEN_FEATURE_NAMES:
            if forbidden in name:
                forbidden_hits.append(name)
    if forbidden_hits:
        raise ValueError("Forbidden selector feature names: {}".format(", ".join(sorted(forbidden_hits))))


def read_selector_csv(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def parse_bbox(value):
    return np.asarray(json.loads(value), dtype=np.float64)


def rows_to_feature_matrix(rows, feature_columns=FEATURE_COLUMNS):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    validate_feature_columns(feature_columns)
    return np.asarray([[float(row[col]) for col in feature_columns] for row in rows], dtype=np.float64)


def non_ignore_rows(rows):
    return [row for row in rows if int(row["ignore"]) == 0]


def normalize_features(x, stats):
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    return (x - mean) / np.maximum(std, 1e-6)


def compute_norm_stats(x, feature_columns=FEATURE_COLUMNS):
    return {
        "mean": x.mean(axis=0).tolist(),
        "std": np.maximum(x.std(axis=0), 1e-6).tolist(),
        "feature_columns": list(feature_columns),
    }


def seq_metrics_from_array(pred_bb, seq):
    pred_bb = torch.tensor(np.asarray(pred_bb, dtype=np.float64), dtype=torch.float64)
    anno_bb = torch.tensor(seq.ground_truth_rect, dtype=torch.float64)
    target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if getattr(seq, "target_visible", None) is not None else None
    err_overlap, err_center, err_center_normalized, _ = calc_seq_err_robust(
        pred_bb, anno_bb, seq.dataset, target_visible
    )
    seq_length = anno_bb.shape[0]
    auc = (err_overlap.view(-1, 1) > THRESHOLD_OVERLAP.view(1, -1)).sum(0).float().mean().item() / seq_length * 100.0
    precision = (err_center.view(-1, 1) <= THRESHOLD_CENTER.view(1, -1)).sum(0).float()[20].item() / seq_length * 100.0
    norm_precision = (
        (err_center_normalized.view(-1, 1) <= THRESHOLD_CENTER_NORM.view(1, -1)).sum(0).float()[20].item()
        / seq_length
        * 100.0
    )
    return auc, precision, norm_precision


def summarize_metric_rows(rows):
    return {
        "auc": float(np.mean([r["auc"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "norm_precision": float(np.mean([r["norm_precision"] for r in rows])),
    }


def group_rows_by_sequence(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    for sequence in grouped:
        grouped[sequence].sort(key=lambda r: int(r["frame"]))
    return grouped


def evaluate_rows_as_predictions(rows, dataset_name, mode):
    dataset = {seq.name: seq for seq in get_dataset(dataset_name)}
    grouped = group_rows_by_sequence(rows)
    metric_rows = []
    for sequence, seq_rows in grouped.items():
        seq = dataset[sequence]
        pred = []
        for row in seq_rows:
            if mode == "local":
                pred.append(parse_bbox(row["local_bbox"]))
            elif mode == "collab":
                pred.append(parse_bbox(row["collab_bbox"]))
            elif mode == "oracle":
                local_loss = float(row["local_loss"])
                collab_loss = float(row["collab_loss"])
                pred.append(parse_bbox(row["collab_bbox"] if collab_loss < local_loss else row["local_bbox"]))
            elif mode.startswith("prob:"):
                threshold = float(mode.split(":", 1)[1])
                pred.append(parse_bbox(row["collab_bbox"] if float(row["selector_prob"]) > threshold else row["local_bbox"]))
            else:
                raise ValueError("Unsupported prediction mode: {}".format(mode))
        auc, precision, norm_precision = seq_metrics_from_array(np.asarray(pred, dtype=np.float64), seq)
        metric_rows.append({
            "sequence": sequence,
            "view": view_name(sequence),
            "auc": auc,
            "precision": precision,
            "norm_precision": norm_precision,
        })
    return metric_rows


def delta_stats(rows, base_rows):
    base = {row["sequence"]: row for row in base_rows}
    deltas = []
    for row in rows:
        delta = row["auc"] - base[row["sequence"]]["auc"]
        deltas.append((row["sequence"], row["view"], delta))
    pos = sum(1 for _, _, d in deltas if d > 1e-6)
    neg = sum(1 for _, _, d in deltas if d < -1e-6)
    same = len(deltas) - pos - neg
    return {
        "positive": pos,
        "negative": neg,
        "same": same,
        "negative_rate": neg / max(len(deltas), 1) * 100.0,
        "top_positive": sorted(deltas, key=lambda x: x[2], reverse=True)[:5],
        "top_negative": sorted(deltas, key=lambda x: x[2])[:5],
    }


def summarize_by_view(rows):
    out = {}
    for view in ("Drone A", "Drone B", "Drone C"):
        view_rows = [row for row in rows if row["view"] == view]
        out[view] = summarize_metric_rows(view_rows) if view_rows else None
    return out


def format_metric(summary):
    return "{auc:.3f} / {precision:.3f} / {norm_precision:.3f}".format(**summary)


def format_delta(summary, base):
    return "{:+.3f} / {:+.3f} / {:+.3f}".format(
        summary["auc"] - base["auc"],
        summary["precision"] - base["precision"],
        summary["norm_precision"] - base["norm_precision"],
    )


def format_top(items):
    return ", ".join("{} {:+.3f}".format(seq, delta) for seq, _, delta in items)
