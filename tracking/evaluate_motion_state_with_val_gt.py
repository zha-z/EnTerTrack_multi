#!/usr/bin/env python3
"""Offline validation-only GT audit for M0 motion-state shadow diagnostics."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml


VALIDATION_DATASET = "threemdot_val"
SIGNAL_DIRECTIONS = {
    "max_score": "low",
    "apce": "low",
    "response_entropy": "high",
    "response_top1_top2_gap": "low",
    "response_peak_sharpness": "low",
    "normalized_motion_residual": "high",
    "bbox_border_proximity": "low",
    "search_region_border_proximity": "low",
    "remote_quality": "low",
    "remote_weight_entropy": "high",
    "remote_max_weight": "low",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Audit M0 shadow states against threemdot_val GT offline. "
            "This command never runs a tracker."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--iou-fail", type=float, default=0.1)
    parser.add_argument("--iou-recover", type=float, default=0.5)
    parser.add_argument("--center-error-fail", type=float, default=1.0)
    parser.add_argument("--k-fail", type=int, default=3)
    parser.add_argument("--k-recover", type=int, default=3)
    parser.add_argument(
        "--failure-mode",
        choices=("iou", "center", "either", "both"),
        default="either",
    )
    parser.add_argument("--warning-window", type=int, default=10)
    parser.add_argument("--refailure-window", type=int, default=30)
    return parser.parse_args(argv)


def validate_dataset(dataset):
    if str(dataset).lower() != VALIDATION_DATASET:
        raise ValueError(
            "validation-only audit refuses dataset {!r}; expected {!r}".format(
                dataset, VALIDATION_DATASET
            )
        )


def ensure_output_separate(diagnostics_dir, output_dir):
    diagnostics = Path(diagnostics_dir).resolve()
    output = Path(output_dir).resolve()
    if output == diagnostics or diagnostics in output.parents:
        raise ValueError(
            "output-dir must be separate from diagnostics-dir; GT audit must "
            "never write into prediction-only diagnostics"
        )


def resolve_dataset_root(dataset_root=None):
    if dataset_root:
        return Path(dataset_root).resolve()
    from lib.test.evaluation.local import local_env_settings
    return Path(local_env_settings().threemdot_val_path).resolve()


def validation_sequences():
    split_path = (
        Path(__file__).resolve().parents[1]
        / "lib/train/data_specs/threemdot/threemdot_val.txt"
    )
    with split_path.open(encoding="utf-8") as file_handle:
        return [line.strip() for line in file_handle if line.strip()]


def sequence_paths(dataset_root, sequence):
    class_name = sequence.split("-", 1)[0]
    sequence_dir = Path(dataset_root) / class_name / sequence
    return {
        "sequence_dir": sequence_dir,
        "gt": sequence_dir / "groundtruth.txt",
        "occlusion": sequence_dir / "occlusion.txt",
        "out_of_view": sequence_dir / "out_of_view.txt",
        "images": sequence_dir / "img",
    }


def load_bbox_file(path):
    values = np.loadtxt(str(path), delimiter=",", dtype=np.float64, ndmin=2)
    if values.shape[1] != 4:
        raise ValueError("bbox file must have four xywh columns: {}".format(path))
    if not np.isfinite(values).all():
        raise ValueError("bbox file contains non-finite values: {}".format(path))
    return values


def load_prediction_file(path):
    values = np.loadtxt(str(path), dtype=np.float64, ndmin=2)
    if values.shape[1] != 4:
        raise ValueError("prediction file must have four xywh columns: {}".format(path))
    if not np.isfinite(values).all():
        raise ValueError("prediction file contains non-finite values: {}".format(path))
    return values


def load_binary_annotation(path):
    values = np.loadtxt(str(path), delimiter=",", dtype=np.float64, ndmin=1)
    values = np.asarray(values).reshape(-1)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("visibility annotation must contain only 0/1: {}".format(path))
    return values.astype(bool)


def load_visibility(sequence_path, expected_length):
    occlusion_path = Path(sequence_path) / "occlusion.txt"
    out_of_view_path = Path(sequence_path) / "out_of_view.txt"
    if not occlusion_path.is_file() or not out_of_view_path.is_file():
        return {
            "available": False,
            "visible": None,
            "occluded": None,
            "out_of_view": None,
            "reason": "occlusion.txt or out_of_view.txt unavailable",
        }
    occluded = load_binary_annotation(occlusion_path)
    out_of_view = load_binary_annotation(out_of_view_path)
    validate_lengths(
        expected_length,
        occlusion=occluded,
        out_of_view=out_of_view,
    )
    return {
        "available": True,
        "visible": np.logical_not(np.logical_or(occluded, out_of_view)),
        "occluded": occluded,
        "out_of_view": out_of_view,
        "reason": None,
    }


def resolve_diagnostics_file(diagnostics_dir, sequence):
    root = Path(diagnostics_dir)
    direct = root / "{}.jsonl".format(sequence)
    nested = root / "motion_state_diagnostics" / "{}.jsonl".format(sequence)
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError("missing diagnostics JSONL for {}".format(sequence))


def load_diagnostics(path):
    records = []
    with Path(path).open() as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "frame_id" not in record or "state" not in record:
                raise ValueError(
                    "invalid diagnostics record at {}:{}".format(path, line_number)
                )
            records.append(record)
    frame_ids = [int(record["frame_id"]) for record in records]
    if frame_ids != list(range(len(records))):
        raise ValueError("diagnostics frame_id must be contiguous from zero: {}".format(path))
    return records


def validate_lengths(expected_length, **arrays):
    mismatches = {
        name: len(values)
        for name, values in arrays.items()
        if len(values) != int(expected_length)
    }
    if mismatches:
        raise ValueError(
            "frame length mismatch; expected {}, got {}".format(
                expected_length, mismatches
            )
        )


def bbox_iou_xywh(prediction, ground_truth):
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    top_left = np.maximum(prediction[:, :2], ground_truth[:, :2])
    bottom_right = np.minimum(
        prediction[:, :2] + prediction[:, 2:4],
        ground_truth[:, :2] + ground_truth[:, 2:4],
    )
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    prediction_area = np.maximum(prediction[:, 2] * prediction[:, 3], 0.0)
    gt_area = np.maximum(ground_truth[:, 2] * ground_truth[:, 3], 0.0)
    union = prediction_area + gt_area - intersection
    return np.divide(
        intersection,
        np.maximum(union, 1e-12),
        out=np.zeros_like(intersection),
    )


def normalized_center_error(prediction, ground_truth):
    prediction_center = prediction[:, :2] + 0.5 * prediction[:, 2:4]
    gt_center = ground_truth[:, :2] + 0.5 * ground_truth[:, 2:4]
    gt_diagonal = np.linalg.norm(ground_truth[:, 2:4], axis=1)
    return np.linalg.norm(prediction_center - gt_center, axis=1) / np.maximum(
        gt_diagonal, 1e-12
    )


def consecutive_true(values, minimum_length):
    values = np.asarray(values, dtype=bool)
    minimum_length = max(1, int(minimum_length))
    result = np.zeros(values.shape, dtype=bool)
    count = 0
    for index, value in enumerate(values):
        count = count + 1 if value else 0
        result[index] = count >= minimum_length
    return result


def event_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    events = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(mask) - 1):
            end = index if active and index == len(mask) - 1 else index - 1
            events.append((start, end))
            start = None
    return events


def construct_event_labels(
    iou,
    center_error,
    visible=None,
    iou_fail=0.1,
    iou_recover=0.5,
    center_error_fail=1.0,
    k_fail=3,
    k_recover=3,
    failure_mode="either",
):
    iou_failure = np.asarray(iou) < float(iou_fail)
    center_failure = np.asarray(center_error) > float(center_error_fail)
    if failure_mode == "iou":
        failure_proxy = iou_failure
    elif failure_mode == "center":
        failure_proxy = center_failure
    elif failure_mode == "both":
        failure_proxy = np.logical_and(iou_failure, center_failure)
    else:
        failure_proxy = np.logical_or(iou_failure, center_failure)

    if visible is None:
        evaluation_mask = np.ones(failure_proxy.shape, dtype=bool)
    else:
        evaluation_mask = np.asarray(visible, dtype=bool)
        failure_proxy = np.logical_and(failure_proxy, evaluation_mask)
    stable_failure = consecutive_true(failure_proxy, k_fail)
    recovery_proxy = np.logical_and(np.asarray(iou) > float(iou_recover), evaluation_mask)
    stable_recovery_all = consecutive_true(recovery_proxy, k_recover)

    # A recovery event is accepted only after at least one stable failure.
    stable_recovery = np.zeros(stable_recovery_all.shape, dtype=bool)
    failure_seen = False
    recovered_for_current_failure = False
    for index in range(len(stable_failure)):
        if stable_failure[index]:
            failure_seen = True
            recovered_for_current_failure = False
        if failure_seen and not recovered_for_current_failure and stable_recovery_all[index]:
            stable_recovery[index] = True
            recovered_for_current_failure = True
    return {
        "iou_failure": iou_failure,
        "center_failure": center_failure,
        "failure_proxy": failure_proxy,
        "stable_failure": stable_failure,
        "stable_recovery": stable_recovery,
        "evaluation_mask": evaluation_mask,
    }


def binary_metrics(prediction, target, evaluation_mask=None):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    mask = (
        np.ones(target.shape, dtype=bool)
        if evaluation_mask is None else np.asarray(evaluation_mask, dtype=bool)
    )
    prediction = prediction[mask]
    target = target[mask]
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    normal_frames = int((~target).sum())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_triggers": false_positive,
        "false_triggers_per_1000": 1000.0 * false_positive / max(normal_frames, 1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def detection_delays(target_events, detection_mask, warning_window=0):
    detection_mask = np.asarray(detection_mask, dtype=bool)
    detection_onsets = [start for start, _ in event_runs(detection_mask)]
    delays = []
    missed = 0
    first_after = []
    for start, end in target_events:
        candidates = [
            onset for onset in detection_onsets
            if start - int(warning_window) <= onset <= end
        ]
        if candidates:
            delays.append(min(candidates, key=lambda value: abs(value - start)) - start)
        else:
            missed += 1
        after = np.flatnonzero(detection_mask[start:])
        first_after.append(int(after[0]) if after.size else None)
    return {"delays": delays, "missed": missed, "first_after": first_after}


def false_trigger_count(detection_mask, target_mask, warning_window=0):
    target_mask = np.asarray(target_mask, dtype=bool)
    allowed = target_mask.copy()
    for start, _ in event_runs(target_mask):
        allowed[max(0, start - int(warning_window)):start] = True
    return sum(
        not allowed[start]
        for start, _ in event_runs(np.asarray(detection_mask, dtype=bool))
    )


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[labels].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order].astype(np.float64)
    true_positive = np.cumsum(ordered)
    false_positive = np.cumsum(1.0 - ordered)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / labels.sum()
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def finite_signal(records, name):
    values = np.full(len(records), np.nan, dtype=np.float64)
    for index, record in enumerate(records):
        value = record.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[index] = float(value)
    return values


def quantile_text(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return "unavailable"
    quantiles = np.quantile(values, [0.1, 0.5, 0.9])
    return "q10={:.4g}, q50={:.4g}, q90={:.4g}".format(*quantiles)


def state_durations(states, state):
    return [end - start + 1 for start, end in event_runs(np.asarray(states) == state)]


def segmented_consecutive_true(values, sequence_ids, minimum_length):
    values = np.asarray(values, dtype=bool)
    sequence_ids = np.asarray(sequence_ids)
    validate_lengths(len(values), sequence_ids=sequence_ids)
    result = np.zeros(values.shape, dtype=bool)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sequence_ids[end] == sequence_ids[start]:
            end += 1
        result[start:end] = consecutive_true(values[start:end], minimum_length)
        start = end
    return result


def segmented_event_runs(mask, sequence_ids):
    mask = np.asarray(mask, dtype=bool)
    sequence_ids = np.asarray(sequence_ids)
    validate_lengths(len(mask), sequence_ids=sequence_ids)
    events = []
    start = 0
    while start < len(mask):
        end = start + 1
        while end < len(mask) and sequence_ids[end] == sequence_ids[start]:
            end += 1
        events.extend(
            (event_start + start, event_end + start)
            for event_start, event_end in event_runs(mask[start:end])
        )
        start = end
    return events


def segmented_detection_delays(target_events, detection_mask, sequence_ids,
                               warning_window=0):
    detection_mask = np.asarray(detection_mask, dtype=bool)
    sequence_ids = np.asarray(sequence_ids)
    detection_onsets = [
        start for start, _ in segmented_event_runs(detection_mask, sequence_ids)
    ]
    delays = []
    missed = 0
    first_after = []
    for start, end in target_events:
        sequence = sequence_ids[start]
        sequence_end = start
        while (sequence_end + 1 < len(sequence_ids)
               and sequence_ids[sequence_end + 1] == sequence):
            sequence_end += 1
        candidates = [
            onset for onset in detection_onsets
            if sequence_ids[onset] == sequence
            and start - int(warning_window) <= onset <= end
        ]
        if candidates:
            delays.append(min(candidates, key=lambda value: abs(value - start)) - start)
        else:
            missed += 1
        after = np.flatnonzero(detection_mask[start:sequence_end + 1])
        first_after.append(int(after[0]) if after.size else None)
    return {"delays": delays, "missed": missed, "first_after": first_after}


def segmented_state_durations(states, sequence_ids, state):
    states = np.asarray(states)
    sequence_ids = np.asarray(sequence_ids)
    return [
        end - start + 1
        for start, end in segmented_event_runs(states == state, sequence_ids)
    ]


def sequence_metrics(sequence, frame_rows, records, labels, visibility,
                     warning_window=10, refailure_window=30):
    states = np.asarray([record["state"] for record in records])
    failure = labels["stable_failure"]
    recovery = labels["stable_recovery"]
    evaluation_mask = labels["evaluation_mask"]
    failure_events = event_runs(failure)
    lost = states == "LOST"
    uncertain = states == "UNCERTAIN"
    recover = states == "RECOVER"
    lost_timing = detection_delays(
        failure_events, lost, warning_window=warning_window
    )
    recovery_events = event_runs(recovery)
    recovery_timing = detection_delays(recovery_events, recover, warning_window=0)

    row = {
        "sequence": sequence,
        "frame_count": len(records),
        "visibility_available": visibility["available"],
        "visible_frames": int(evaluation_mask.sum()),
        "occluded_frames": int(visibility["occluded"].sum())
        if visibility["available"] else None,
        "out_of_view_frames": int(visibility["out_of_view"].sum())
        if visibility["available"] else None,
        "failure_event_count": len(failure_events),
        "recovery_event_count": len(recovery_events),
        "lost_missed_failure_events": lost_timing["missed"],
        "lost_false_trigger_events": false_trigger_count(
            lost, failure, warning_window
        ),
        "lost_mean_detection_delay": float(np.mean(lost_timing["delays"]))
        if lost_timing["delays"] else None,
        "lost_median_detection_delay": float(np.median(lost_timing["delays"]))
        if lost_timing["delays"] else None,
        "lost_mean_first_entry_after_failure": float(np.mean([
            value for value in lost_timing["first_after"] if value is not None
        ])) if any(
            value is not None for value in lost_timing["first_after"]
        ) else None,
        "failure_with_uncertain_warning": sum(
            uncertain[max(0, start - int(warning_window)):start].any()
            for start, _ in failure_events
        ),
        "recovery_missed_events": recovery_timing["missed"],
        "recovery_false_events": false_trigger_count(recover, recovery, 0),
        "recovery_mean_detection_delay": float(np.mean(recovery_timing["delays"]))
        if recovery_timing["delays"] else None,
    }
    for state, target in (
        ("UNCERTAIN", failure),
        ("LOST", failure),
        ("RECOVER", recovery),
    ):
        state_mask = states == state
        metrics = binary_metrics(state_mask, target, evaluation_mask)
        durations = state_durations(states, state)
        prefix = state.lower()
        row.update({
            "{}_precision".format(prefix): metrics["precision"],
            "{}_recall".format(prefix): metrics["recall"],
            "{}_f1".format(prefix): metrics["f1"],
            "{}_pr_auc".format(prefix): pr_auc(
                target[evaluation_mask], state_mask[evaluation_mask].astype(float)
            ),
            "{}_false_triggers_per_1000".format(prefix): metrics[
                "false_triggers_per_1000"
            ],
            "{}_frame_ratio".format(prefix): float(state_mask.mean()),
            "{}_transition_count".format(prefix): len(durations),
            "{}_mean_duration".format(prefix): float(np.mean(durations))
            if durations else 0.0,
            "{}_median_duration".format(prefix): float(np.median(durations))
            if durations else 0.0,
        })
    row["recover_refailure_ratio"] = refailure_ratio(
        recover, failure, window=refailure_window
    )
    return row


def refailure_ratio(recover_mask, failure_mask, window=30):
    recover_events = event_runs(recover_mask)
    if not recover_events:
        return 0.0
    refailed = 0
    for _, end in recover_events:
        if np.asarray(failure_mask, dtype=bool)[end + 1:end + 1 + int(window)].any():
            refailed += 1
    return refailed / len(recover_events)


def simulate_threshold_candidate(records, sequence_ids, score_low, apce_low,
                                 residual_high, k_fail):
    score = finite_signal(records, "max_score")
    apce = finite_signal(records, "apce")
    residual = finite_signal(records, "normalized_motion_residual")
    alert = (
        (np.isfinite(score) & (score < score_low))
        | (np.isfinite(apce) & (apce < apce_low))
        | (np.isfinite(residual) & (residual > residual_high))
        | np.asarray([bool(record.get("bbox_border_event")) for record in records])
        | np.asarray([
            bool(record.get("search_region_border_event")) for record in records
        ])
    )
    return segmented_consecutive_true(alert, sequence_ids, k_fail)


def threshold_candidates(all_records, sequence_ids, failure_labels,
                         evaluation_mask, k_fail):
    score = finite_signal(all_records, "max_score")
    apce = finite_signal(all_records, "apce")
    residual = finite_signal(all_records, "normalized_motion_residual")
    if not (np.isfinite(score).any() and np.isfinite(apce).any()
            and np.isfinite(residual).any()):
        return []
    low_quantiles = (0.05, 0.1, 0.2, 0.3, 0.4)
    high_quantiles = (0.6, 0.7, 0.8, 0.9, 0.95)
    score_grid = np.quantile(score[np.isfinite(score)], low_quantiles)
    apce_grid = np.quantile(apce[np.isfinite(apce)], low_quantiles)
    residual_grid = np.quantile(residual[np.isfinite(residual)], high_quantiles)
    candidates = []
    failure_events = segmented_event_runs(failure_labels, sequence_ids)
    for score_low in score_grid:
        for apce_low in apce_grid:
            for residual_high in residual_grid:
                predicted = simulate_threshold_candidate(
                    all_records, sequence_ids, score_low, apce_low,
                    residual_high, k_fail
                )
                metrics = binary_metrics(predicted, failure_labels, evaluation_mask)
                timing = segmented_detection_delays(
                    failure_events,
                    predicted,
                    sequence_ids,
                    warning_window=10,
                )
                mean_delay = float(np.mean(timing["delays"])) if timing["delays"] else None
                transitions = len(segmented_event_runs(predicted, sequence_ids))
                transitions_per_1000 = 1000.0 * transitions / max(len(predicted), 1)
                candidates.append({
                    "score_low": float(score_low),
                    "apce_low": float(apce_low),
                    "motion_residual_high": float(residual_high),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "false_triggers_per_1000": metrics["false_triggers_per_1000"],
                    "average_detection_delay": mean_delay,
                    "state_transitions_per_1000": transitions_per_1000,
                })

    def delay_penalty(candidate):
        delay = candidate["average_detection_delay"]
        return 20.0 if delay is None else max(float(delay), 0.0)

    rankings = {
        "conservative": lambda item: (
            -item["false_triggers_per_1000"],
            item["precision"],
            item["recall"],
            -delay_penalty(item),
        ),
        "balanced": lambda item: (
            item["f1"]
            - 0.001 * item["false_triggers_per_1000"]
            - 0.002 * delay_penalty(item)
            - 0.0005 * item["state_transitions_per_1000"],
        ),
        "sensitive": lambda item: (
            item["recall"],
            -delay_penalty(item),
            item["precision"],
        ),
    }
    selected = []
    used = set()
    for profile, key_fn in rankings.items():
        for candidate in sorted(candidates, key=key_fn, reverse=True):
            key = (
                candidate["score_low"],
                candidate["apce_low"],
                candidate["motion_residual_high"],
            )
            if key not in used:
                used.add(key)
                selected.append((profile, candidate))
                break
    return selected


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_signal_discrimination(path, all_frame_rows, all_records):
    labels = np.asarray([row["stable_failure"] for row in all_frame_rows], dtype=bool)
    evaluation = np.asarray([row["evaluation_frame"] for row in all_frame_rows], dtype=bool)
    rows = []
    for signal, direction in SIGNAL_DIRECTIONS.items():
        values = finite_signal(all_records, signal)
        available = evaluation & np.isfinite(values)
        failure_values = values[available & labels]
        normal_values = values[available & ~labels]
        oriented = -values[available] if direction == "low" else values[available]
        rows.append({
            "signal": signal,
            "recommended_trigger_direction": direction,
            "roc_auc": roc_auc(labels[available], oriented),
            "pr_auc": pr_auc(labels[available], oriented),
            "missing_rate": 1.0 - float(np.isfinite(values).mean()),
            "failure_mean": float(failure_values.mean()) if failure_values.size else None,
            "failure_quantiles": quantile_text(failure_values),
            "normal_mean": float(normal_values.mean()) if normal_values.size else None,
            "normal_quantiles": quantile_text(normal_values),
            "failure_count": int(failure_values.size),
            "normal_count": int(normal_values.size),
        })
    write_csv(path, rows)
    return rows


def example_record(sequence_data, frame, category, range_start=None, range_end=None):
    row = sequence_data["frame_rows"][frame]
    record = sequence_data["records"][frame]
    return {
        "category": category,
        "sequence": sequence_data["sequence"],
        "frame_zero_based": frame,
        "range_start_zero_based": frame if range_start is None else range_start,
        "range_end_zero_based": frame if range_end is None else range_end,
        "image_frame_one_based": frame + 1,
        "state": record["state"],
        "iou": row["iou"],
        "center_error_norm": row["center_error_norm"],
        "max_score": record.get("max_score"),
        "apce": record.get("apce"),
        "response_entropy": record.get("response_entropy"),
        "normalized_motion_residual": record.get("normalized_motion_residual"),
        "prediction_path": str(sequence_data["prediction_path"]),
        "gt_path": str(sequence_data["gt_path"]),
        "diagnostics_path": str(sequence_data["diagnostics_path"]),
    }


def select_examples(sequence_data_list):
    categories = {
        "LOST correct trigger": [],
        "LOST missed failure": [],
        "LOST false trigger": [],
        "RECOVER correct": [],
        "RECOVER false trigger": [],
    }
    for data in sequence_data_list:
        states = np.asarray([record["state"] for record in data["records"]])
        failure = data["labels"]["stable_failure"]
        recovery = data["labels"]["stable_recovery"]
        for start, end in event_runs(failure):
            lost_frames = np.flatnonzero(states[start:end + 1] == "LOST")
            category = "LOST correct trigger" if lost_frames.size else "LOST missed failure"
            frame = start + int(lost_frames[0]) if lost_frames.size else start
            categories[category].append(
                example_record(data, frame, category, start, end)
            )
        for start, end in event_runs(states == "LOST"):
            if not failure[start]:
                categories["LOST false trigger"].append(
                    example_record(
                        data, start, "LOST false trigger", start, end
                    )
                )
        for start, end in event_runs(recovery):
            recover_frames = np.flatnonzero(states[start:end + 1] == "RECOVER")
            if recover_frames.size:
                frame = start + int(recover_frames[0])
                categories["RECOVER correct"].append(
                    example_record(data, frame, "RECOVER correct", start, end)
                )
        for start, end in event_runs(states == "RECOVER"):
            if not recovery[start]:
                categories["RECOVER false trigger"].append(
                    example_record(
                        data, start, "RECOVER false trigger", start, end
                    )
                )
    return {name: values[:3] for name, values in categories.items()}


def write_examples(path, examples):
    with Path(path).open("w") as file_handle:
        file_handle.write("# Failure Event Examples\n\n")
        file_handle.write(
            "Validation-only offline GT audit. Frame indexes and paths are "
            "reported; no images are generated.\n\n"
        )
        for category, records in examples.items():
            file_handle.write("## {}\n\n".format(category))
            if not records:
                file_handle.write("- No matching event.\n\n")
                continue
            for record in records:
                file_handle.write(
                    "- `{sequence}` frame range `{range_start_zero_based}-"
                    "{range_end_zero_based}`; representative frame "
                    "`{frame_zero_based}` (image "
                    "`{image_frame_one_based:08d}.jpg`), state `{state}`, "
                    "IoU `{iou:.4f}`, center error `{center_error_norm:.4f}`, "
                    "score `{max_score}`, APCE `{apce}`, entropy "
                    "`{response_entropy}`, motion residual "
                    "`{normalized_motion_residual}`; prediction "
                    "`{prediction_path}`; GT `{gt_path}`; diagnostics "
                    "`{diagnostics_path}`.\n".format(**record)
                )
            file_handle.write("\n")


def aggregate_state_metrics(frame_rows, records, state, target_name):
    states = np.asarray([record["state"] for record in records])
    predicted = states == state
    target = np.asarray([row[target_name] for row in frame_rows], dtype=bool)
    evaluation = np.asarray([row["evaluation_frame"] for row in frame_rows], dtype=bool)
    metrics = binary_metrics(predicted, target, evaluation)
    sequence_ids = [row["sequence"] for row in frame_rows]
    durations = segmented_state_durations(states, sequence_ids, state)
    metrics.update({
        "state": state,
        "state_frame_ratio": float(predicted.mean()),
        "transition_count": len(durations),
        "mean_duration": float(np.mean(durations)) if durations else 0.0,
        "median_duration": float(np.median(durations)) if durations else 0.0,
        "pr_auc": pr_auc(target[evaluation], predicted[evaluation].astype(float)),
    })
    return metrics


def aggregate_event_summary(sequence_rows):
    def finite_values(name):
        return [
            float(row[name]) for row in sequence_rows
            if row.get(name) is not None and math.isfinite(float(row[name]))
        ]

    failure_events = sum(row["failure_event_count"] for row in sequence_rows)
    recovery_events = sum(row["recovery_event_count"] for row in sequence_rows)
    lost_delays = finite_values("lost_mean_detection_delay")
    recovery_delays = finite_values("recovery_mean_detection_delay")
    return {
        "failure_events": failure_events,
        "lost_missed": sum(row["lost_missed_failure_events"] for row in sequence_rows),
        "lost_false": sum(row["lost_false_trigger_events"] for row in sequence_rows),
        "lost_delay_mean": float(np.mean(lost_delays)) if lost_delays else None,
        "uncertain_warning": sum(
            row["failure_with_uncertain_warning"] for row in sequence_rows
        ),
        "recovery_events": recovery_events,
        "recovery_missed": sum(row["recovery_missed_events"] for row in sequence_rows),
        "recovery_false": sum(row["recovery_false_events"] for row in sequence_rows),
        "recovery_delay_mean": float(np.mean(recovery_delays))
        if recovery_delays else None,
        "recover_refailure_ratio": float(np.mean([
            row["recover_refailure_ratio"] for row in sequence_rows
        ])) if sequence_rows else 0.0,
    }


def write_report(path, args, dataset_root, sequence_rows, state_metrics,
                 signal_rows, candidates, visibility_available_count):
    event_summary = aggregate_event_summary(sequence_rows)
    with Path(path).open("w") as file_handle:
        file_handle.write("# M0 Motion State Validation GT Audit\n\n")
        file_handle.write("- Result label: **validation diagnostic result**.\n")
        file_handle.write("- Dataset: `threemdot_val`; test support: refused.\n")
        file_handle.write("- Tracker execution: none; predictions are read-only.\n")
        file_handle.write("- GT root: `{}`.\n".format(dataset_root))
        file_handle.write(
            "- GT is used only by this offline script and is never written "
            "into diagnostics or predictions.\n"
        )
        file_handle.write(
            "- Visibility available: `{}/{}` sequences; invisible frames are "
            "excluded from visible localization-failure metrics and reported "
            "separately.\n\n".format(
                visibility_available_count, len(sequence_rows)
            )
        )
        file_handle.write("## Event Definition\n\n")
        file_handle.write(
            "- Failure mode `{}`; IoU fail `< {}`; center error / GT bbox "
            "diagonal `> {}`; stable for `{}` frames.\n".format(
                args.failure_mode, args.iou_fail, args.center_error_fail, args.k_fail
            )
        )
        file_handle.write(
            "- Recovery IoU `> {}` for `{}` visible frames after a stable "
            "failure.\n\n".format(args.iou_recover, args.k_recover)
        )
        file_handle.write("## Shadow State Metrics\n\n")
        file_handle.write("| State | Precision | Recall | F1 | PR-AUC | False/1000 | Ratio | Transitions | Mean/median duration |\n")
        file_handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for metrics in state_metrics:
            file_handle.write(
                "| {state} | {precision:.4f} | {recall:.4f} | {f1:.4f} | "
                "{pr_auc} | {false_triggers_per_1000:.2f} | "
                "{state_frame_ratio:.4f} | {transition_count} | "
                "{mean_duration:.2f}/{median_duration:.2f} |\n".format(**metrics)
            )
        file_handle.write("\n## Event Timing\n\n")
        file_handle.write(
            "- Stable failure events: `{failure_events}`; LOST missed: "
            "`{lost_missed}`; LOST false trigger events: `{lost_false}`; "
            "mean sequence-level detection delay: `{lost_delay_mean}` frames.\n"
            "- Failure events with UNCERTAIN in the preceding warning window: "
            "`{uncertain_warning}`.\n"
            "- Recovery events: `{recovery_events}`; RECOVER missed: "
            "`{recovery_missed}`; false recovery events: `{recovery_false}`; "
            "mean sequence-level recovery delay: `{recovery_delay_mean}` frames.\n"
            "- Mean per-sequence RECOVER refailure ratio: "
            "`{recover_refailure_ratio}`.\n".format(**event_summary)
        )
        file_handle.write("\n## Signal Discrimination\n\n")
        file_handle.write("| Signal | Direction | ROC-AUC | PR-AUC | Missing |\n")
        file_handle.write("|---|---|---:|---:|---:|\n")
        for row in signal_rows:
            file_handle.write(
                "| {signal} | {recommended_trigger_direction} | {roc_auc} | "
                "{pr_auc} | {missing_rate:.4f} |\n".format(**row)
            )
        file_handle.write("\n## Threshold Boundary\n\n")
        file_handle.write(
            "The candidate thresholds are **validation-derived provisional "
            "diagnostics**. They are not written back to any tracker YAML and "
            "must never be tuned using `threemdot_test`. Candidate count: "
            "`{}`.\n".format(len(candidates))
        )


def main(argv=None):
    args = parse_args(argv)
    validate_dataset(args.dataset)
    ensure_output_separate(args.diagnostics_dir, args.output_dir)
    dataset_root = resolve_dataset_root(args.dataset_root)
    prediction_dir = Path(args.prediction_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frame_rows = []
    all_records = []
    sequence_rows = []
    sequence_data_list = []
    visibility_available_count = 0
    for sequence in validation_sequences():
        paths = sequence_paths(dataset_root, sequence)
        prediction_path = prediction_dir / "{}.txt".format(sequence)
        diagnostics_path = resolve_diagnostics_file(args.diagnostics_dir, sequence)
        gt = load_bbox_file(paths["gt"])
        prediction = load_prediction_file(prediction_path)
        records = load_diagnostics(diagnostics_path)
        validate_lengths(len(gt), prediction=prediction, diagnostics=records)
        visibility = load_visibility(paths["sequence_dir"], len(gt))
        visibility_available_count += int(visibility["available"])
        iou = bbox_iou_xywh(prediction, gt)
        center_error = normalized_center_error(prediction, gt)
        labels = construct_event_labels(
            iou,
            center_error,
            visible=visibility["visible"],
            iou_fail=args.iou_fail,
            iou_recover=args.iou_recover,
            center_error_fail=args.center_error_fail,
            k_fail=args.k_fail,
            k_recover=args.k_recover,
            failure_mode=args.failure_mode,
        )
        frame_rows = []
        for frame_id, record in enumerate(records):
            frame_row = {
                "sequence": sequence,
                "frame_id": frame_id,
                "image_frame_one_based": frame_id + 1,
                "state": record["state"],
                "iou": float(iou[frame_id]),
                "center_error_norm": float(center_error[frame_id]),
                "iou_failure": bool(labels["iou_failure"][frame_id]),
                "center_failure": bool(labels["center_failure"][frame_id]),
                "stable_failure": bool(labels["stable_failure"][frame_id]),
                "stable_recovery": bool(labels["stable_recovery"][frame_id]),
                "evaluation_frame": bool(labels["evaluation_mask"][frame_id]),
                "visibility_available": visibility["available"],
                "visible": bool(visibility["visible"][frame_id])
                if visibility["available"] else None,
                "occluded": bool(visibility["occluded"][frame_id])
                if visibility["available"] else None,
                "out_of_view": bool(visibility["out_of_view"][frame_id])
                if visibility["available"] else None,
            }
            for signal in SIGNAL_DIRECTIONS:
                value = record.get(signal)
                frame_row[signal] = (
                    float(value)
                    if isinstance(value, (int, float))
                    and math.isfinite(float(value)) else None
                )
            frame_rows.append(frame_row)
        sequence_row = sequence_metrics(
            sequence,
            frame_rows,
            records,
            labels,
            visibility,
            warning_window=args.warning_window,
            refailure_window=args.refailure_window,
        )
        sequence_rows.append(sequence_row)
        all_frame_rows.extend(frame_rows)
        all_records.extend(records)
        sequence_data_list.append({
            "sequence": sequence,
            "frame_rows": frame_rows,
            "records": records,
            "labels": labels,
            "prediction_path": prediction_path,
            "gt_path": paths["gt"],
            "diagnostics_path": diagnostics_path,
        })

    write_csv(output_dir / "frame_level_metrics.csv", all_frame_rows)
    write_csv(output_dir / "sequence_event_metrics.csv", sequence_rows)
    signal_rows = write_signal_discrimination(
        output_dir / "signal_failure_discrimination.csv",
        all_frame_rows,
        all_records,
    )
    state_metrics = [
        aggregate_state_metrics(all_frame_rows, all_records, "UNCERTAIN", "stable_failure"),
        aggregate_state_metrics(all_frame_rows, all_records, "LOST", "stable_failure"),
        aggregate_state_metrics(all_frame_rows, all_records, "RECOVER", "stable_recovery"),
    ]
    failure_labels = np.asarray(
        [row["stable_failure"] for row in all_frame_rows], dtype=bool
    )
    evaluation_mask = np.asarray(
        [row["evaluation_frame"] for row in all_frame_rows], dtype=bool
    )
    sequence_ids = [row["sequence"] for row in all_frame_rows]
    candidates = threshold_candidates(
        all_records, sequence_ids, failure_labels, evaluation_mask, args.k_fail
    )
    threshold_output = {
        "provenance": "validation-derived offline GT audit on threemdot_val",
        "warning": (
            "provisional diagnostics only; not written to tracker config and "
            "not eligible for test-derived tuning"
        ),
        "failure_proxy": {
            "mode": args.failure_mode,
            "iou_fail": args.iou_fail,
            "iou_recover": args.iou_recover,
            "center_error_fail": args.center_error_fail,
            "k_fail": args.k_fail,
            "k_recover": args.k_recover,
        },
        "candidates": [
            {
                "profile": profile,
                "TEST.MOTION_STATE": {
                    "SCORE_LOW": candidate["score_low"],
                    "APCE_LOW": candidate["apce_low"],
                    "MOTION_RESIDUAL_HIGH": candidate["motion_residual_high"],
                    "K_LOST": args.k_fail,
                },
                "offline_metrics": {
                    key: value for key, value in candidate.items()
                    if key not in ("score_low", "apce_low", "motion_residual_high")
                },
            }
            for profile, candidate in candidates
        ],
    }
    with (output_dir / "threshold_candidates.yaml").open("w") as file_handle:
        yaml.safe_dump(threshold_output, file_handle, sort_keys=False)
    write_examples(
        output_dir / "failure_event_examples.md",
        select_examples(sequence_data_list),
    )
    write_report(
        output_dir / "motion_state_gt_audit.md",
        args,
        dataset_root,
        sequence_rows,
        state_metrics,
        signal_rows,
        candidates,
        visibility_available_count,
    )


if __name__ == "__main__":
    main()
