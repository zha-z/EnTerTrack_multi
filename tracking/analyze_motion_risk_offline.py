#!/usr/bin/env python3
"""Offline prediction-only motion risk computation and validation GT audit."""

import argparse
import csv
import inspect
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from tracking.evaluate_motion_state_with_val_gt import (
    bbox_iou_xywh,
    construct_event_labels,
    event_runs,
    load_bbox_file,
    load_prediction_file,
    load_visibility,
    normalized_center_error,
    pr_auc,
    roc_auc,
    validate_dataset,
    validate_lengths,
    validation_sequences,
)


VALIDATION_DATASET = "threemdot_val"
HIGH_RISK_THRESHOLD = 0.70
RISK_CLIP = 4.0
MAD_SCALE = 1.4826
EPSILON = 1e-8

RAW_SIGNAL_ALIASES = {
    "max_score": ("max_score",),
    "apce": ("apce", "APCE"),
    "response_entropy": ("response_entropy",),
    "normalized_motion_residual": ("normalized_motion_residual",),
    "bbox_border_proximity": ("bbox_border_proximity",),
    "remote_quality": ("remote_quality",),
    "remote_weight_entropy": ("remote_weight_entropy",),
    "remote_max_weight": ("remote_max_weight", "remote_weight_max"),
    "top1_top2_gap": ("top1_top2_gap", "response_top1_top2_gap"),
    "peak_sharpness": ("peak_sharpness", "response_peak_sharpness"),
}

RISK_DIRECTIONS = {
    "max_score": "low",
    "apce": "low",
    "response_entropy": "high",
    "normalized_motion_residual": "high",
    "bbox_border_proximity": "low",
    "remote_quality": "low",
    "top1_top2_gap": "low",
    "peak_sharpness": "low",
    # Diagnostic only. These observed directions are not used by the main score.
    "remote_weight_entropy": "low",
    "remote_max_weight": "high",
}

CORE_WEIGHTS = {
    "motion_risk": 0.25,
    "entropy_risk": 0.20,
    "score_risk": 0.20,
    "apce_risk": 0.15,
    "remote_quality_risk": 0.10,
    "auxiliary_response_risk": 0.10,
}

PROFILE_DEFAULTS = {
    "conservative": {
        "uncertain_enter_threshold": 0.60,
        "lost_enter_threshold": 0.80,
        "lost_exit_threshold": 0.30,
        "k_uncertain": 3,
        "k_lost": 5,
        "k_exit": 5,
        "minimum_high_risk_signal_count": 3,
    },
    "balanced": {
        "uncertain_enter_threshold": 0.55,
        "lost_enter_threshold": 0.75,
        "lost_exit_threshold": 0.35,
        "k_uncertain": 2,
        "k_lost": 4,
        "k_exit": 4,
        "minimum_high_risk_signal_count": 2,
    },
    "sensitive": {
        "uncertain_enter_threshold": 0.50,
        "lost_enter_threshold": 0.70,
        "lost_exit_threshold": 0.40,
        "k_uncertain": 1,
        "k_lost": 3,
        "k_exit": 3,
        "minimum_high_risk_signal_count": 2,
    },
}

SEARCH_GRID = {
    "uncertain_enter_threshold": (0.50, 0.55, 0.60),
    "lost_enter_threshold": (0.70, 0.75, 0.80),
    "lost_exit_threshold": (0.30, 0.35, 0.40),
    "k_lost": (3, 4, 5),
}

FORBIDDEN_RISK_OUTPUT_FIELDS = {
    "gt", "gt_bbox", "iou", "visibility", "visible", "occlusion",
    "out_of_view", "oracle_failure", "failure_label", "target_visible",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute causal prediction-only motion risk, then evaluate it "
            "offline using threemdot_val GT. This script never runs a tracker."
        )
    )
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", default="/data2/Three-MDOT")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalization-window", type=int, default=60)
    parser.add_argument("--min-history", type=int, default=20)
    parser.add_argument("--ema-alpha", type=float, default=0.8)
    parser.add_argument(
        "--provisional-profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="balanced",
    )
    return parser.parse_args(argv)


def finite_float(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def extract_prediction_signals(record):
    signals = {}
    for canonical, aliases in RAW_SIGNAL_ALIASES.items():
        value = None
        for alias in aliases:
            value = finite_float(record.get(alias))
            if value is not None:
                break
        signals[canonical] = value
    return signals


def causal_robust_risk(value, history, direction, clip=RISK_CLIP):
    """Map one value to [0, 1] using previous values only."""
    value = finite_float(value)
    history = np.asarray(
        [item for item in history if finite_float(item) is not None],
        dtype=np.float64,
    )
    if value is None or history.size == 0:
        return None
    median = float(np.median(history))
    mad = float(np.median(np.abs(history - median)))
    sign = 1.0 if direction == "high" else -1.0
    if mad <= EPSILON:
        if abs(value - median) <= EPSILON:
            return 0.5
        percentile = (
            float(np.mean(history <= value))
            if direction == "high" else float(np.mean(history >= value))
        )
        return float(np.clip(percentile, 0.0, 1.0))
    robust_z = sign * (value - median) / (MAD_SCALE * mad)
    robust_z = float(np.clip(robust_z, -float(clip), float(clip)))
    return float(1.0 / (1.0 + math.exp(-robust_z)))


def weighted_available_mean(components, weights):
    available = [
        (float(components[name]), float(weight))
        for name, weight in weights.items()
        if components.get(name) is not None
    ]
    if not available:
        return 0.0
    numerator = sum(value * weight for value, weight in available)
    denominator = sum(weight for _, weight in available)
    return numerator / max(denominator, EPSILON)


def combine_risk_components(normalized_risks):
    auxiliary_values = [
        normalized_risks[name]
        for name in ("top1_top2_gap_risk", "peak_sharpness_risk")
        if normalized_risks.get(name) is not None
    ]
    components = {
        "motion_risk": normalized_risks.get("normalized_motion_residual_risk"),
        "entropy_risk": normalized_risks.get("response_entropy_risk"),
        "score_risk": normalized_risks.get("max_score_risk"),
        "apce_risk": normalized_risks.get("apce_risk"),
        "remote_quality_risk": normalized_risks.get("remote_quality_risk"),
        "auxiliary_response_risk": (
            float(np.mean(auxiliary_values)) if auxiliary_values else None
        ),
    }
    core_risk = weighted_available_mean(components, CORE_WEIGHTS)
    motion = components["motion_risk"] or 0.0
    entropy = components["entropy_risk"] or 0.0
    border = normalized_risks.get("bbox_border_proximity_risk") or 0.0
    border_interaction = border * max(motion, entropy)
    high_risk_count = sum(
        value is not None and value > HIGH_RISK_THRESHOLD
        for value in components.values()
    )
    instantaneous = 0.85 * core_risk + 0.15 * border_interaction
    consistency_reduced = high_risk_count < 2
    if consistency_reduced:
        instantaneous *= 0.5
    return {
        "core_risk": float(core_risk),
        "border_interaction": float(border_interaction),
        "instantaneous_risk": float(np.clip(instantaneous, 0.0, 1.0)),
        "high_risk_signal_count": int(high_risk_count),
        "consistency_reduced": bool(consistency_reduced),
        **components,
    }


def apply_risk_ema(previous, current, alpha):
    if previous is None:
        return float(current)
    return float(alpha) * float(previous) + (1.0 - float(alpha)) * float(current)


class ProvisionalRiskStateMachine:
    def __init__(self, profile):
        self.profile = dict(profile)
        if not (
            self.profile["lost_enter_threshold"]
            > self.profile["uncertain_enter_threshold"]
            > self.profile["lost_exit_threshold"]
        ):
            raise ValueError("risk profile thresholds violate hysteresis ordering")
        self.state = "NORMAL"
        self.uncertain_count = 0
        self.lost_count = 0
        self.exit_count = 0

    def update(self, risk_ema, high_risk_count, warmup=False):
        previous = self.state
        reason = "hold"
        if warmup:
            self.state = "NORMAL"
            self.uncertain_count = self.lost_count = self.exit_count = 0
            return self.state, "warmup"
        enough_signals = (
            int(high_risk_count)
            >= self.profile["minimum_high_risk_signal_count"]
        )
        uncertain_high = (
            enough_signals
            and risk_ema >= self.profile["uncertain_enter_threshold"]
        )
        lost_high = (
            enough_signals and risk_ema >= self.profile["lost_enter_threshold"]
        )

        if self.state == "NORMAL":
            self.uncertain_count = self.uncertain_count + 1 if uncertain_high else 0
            if self.uncertain_count >= self.profile["k_uncertain"]:
                self.state = "UNCERTAIN"
                reason = "risk_above_uncertain"
                self.lost_count = 0
        elif self.state == "UNCERTAIN":
            self.lost_count = self.lost_count + 1 if lost_high else 0
            self.exit_count = self.exit_count + 1 if not uncertain_high else 0
            if self.lost_count >= self.profile["k_lost"]:
                self.state = "LOST"
                reason = "sustained_multisignal_risk"
                self.exit_count = 0
            elif self.exit_count >= self.profile["k_exit"]:
                self.state = "NORMAL"
                reason = "risk_below_uncertain"
                self.uncertain_count = self.lost_count = self.exit_count = 0
        elif self.state == "LOST":
            self.exit_count = (
                self.exit_count + 1
                if risk_ema <= self.profile["lost_exit_threshold"] else 0
            )
            if self.exit_count >= self.profile["k_exit"]:
                self.state = "RECOVER"
                reason = "sustained_low_risk_exit"
                self.exit_count = 0
        else:  # RECOVER
            self.lost_count = self.lost_count + 1 if lost_high else 0
            self.exit_count = (
                self.exit_count + 1
                if risk_ema <= self.profile["uncertain_enter_threshold"] else 0
            )
            if self.lost_count >= self.profile["k_lost"]:
                self.state = "LOST"
                reason = "risk_rebounded"
                self.lost_count = self.exit_count = 0
            elif self.exit_count >= self.profile["k_exit"]:
                self.state = "NORMAL"
                reason = "recovery_stable"
                self.lost_count = self.exit_count = 0
        if self.state == previous and reason == "hold":
            reason = "{} hold".format(self.state.lower())
        return self.state, reason


def compute_prediction_only_risk(
    diagnostics_records,
    normalization_window=60,
    min_history=20,
    ema_alpha=0.8,
    provisional_profile=None,
):
    """Stage A interface: intentionally accepts no GT or annotation inputs."""
    if normalization_window < 1 or min_history < 1:
        raise ValueError("normalization window and min history must be positive")
    if not 0.0 <= ema_alpha < 1.0:
        raise ValueError("ema alpha must be in [0, 1)")
    profile = provisional_profile or PROFILE_DEFAULTS["balanced"]
    machine = ProvisionalRiskStateMachine(profile)
    histories = {name: [] for name in RAW_SIGNAL_ALIASES}
    outputs = []
    previous_ema = None
    readiness_signals = (
        "max_score",
        "apce",
        "response_entropy",
        "normalized_motion_residual",
        "remote_quality",
    )
    for expected_frame, record in enumerate(diagnostics_records):
        frame_id = int(record.get("frame_id", expected_frame))
        if frame_id != expected_frame:
            raise ValueError("diagnostics frame_id must be contiguous from zero")
        raw = extract_prediction_signals(record)
        ready_count = sum(
            len(histories[name]) >= int(min_history)
            for name in readiness_signals
        )
        warmup = ready_count < profile["minimum_high_risk_signal_count"]
        normalized = {}
        for name, value in raw.items():
            history = histories[name][-int(normalization_window):]
            normalized[name + "_risk"] = (
                None if len(history) < int(min_history) else causal_robust_risk(
                    value, history, RISK_DIRECTIONS[name]
                )
            )
        combined = combine_risk_components(normalized) if not warmup else {
            "core_risk": 0.0,
            "border_interaction": 0.0,
            "instantaneous_risk": 0.0,
            "high_risk_signal_count": 0,
            "consistency_reduced": True,
            "motion_risk": None,
            "entropy_risk": None,
            "score_risk": None,
            "apce_risk": None,
            "remote_quality_risk": None,
            "auxiliary_response_risk": None,
        }
        current_ema = apply_risk_ema(
            previous_ema, combined["instantaneous_risk"], ema_alpha
        )
        state, reason = machine.update(
            current_ema,
            combined["high_risk_signal_count"],
            warmup=warmup,
        )
        available_count = sum(value is not None for value in raw.values())
        output = {
            "frame_id": frame_id,
            "warmup": warmup,
            "raw_signals": raw,
            "normalized_risks": normalized,
            "available_signal_count": int(available_count),
            "high_risk_signal_count": combined["high_risk_signal_count"],
            "core_risk": combined["core_risk"],
            "border_interaction": combined["border_interaction"],
            "instantaneous_risk": combined["instantaneous_risk"],
            "risk_ema": current_ema,
            "provisional_state": state,
            "transition_reason": reason,
            "consistency_reduced": combined["consistency_reduced"],
        }
        if FORBIDDEN_RISK_OUTPUT_FIELDS.intersection(output):
            raise AssertionError("GT-derived field leaked into risk output")
        outputs.append(output)
        previous_ema = current_ema
        for name, value in raw.items():
            if value is not None:
                histories[name].append(value)
    return outputs


def load_diagnostics_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_prediction_only_phase(
    diagnostics_dir,
    output_dir,
    normalization_window=60,
    min_history=20,
    ema_alpha=0.8,
    provisional_profile="balanced",
):
    """Stage A: read diagnostics only; do not accept dataset or GT paths."""
    diagnostics_dir = Path(diagnostics_dir).resolve()
    risk_dir = Path(output_dir).resolve() / "prediction_only_risk_scores"
    risk_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(diagnostics_dir.glob("*.jsonl"))
    if not paths:
        nested = diagnostics_dir / "motion_state_diagnostics"
        paths = sorted(nested.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError("no motion diagnostics JSONL files found")
    profile = PROFILE_DEFAULTS[provisional_profile]
    for path in paths:
        risk = compute_prediction_only_risk(
            load_diagnostics_jsonl(path),
            normalization_window=normalization_window,
            min_history=min_history,
            ema_alpha=ema_alpha,
            provisional_profile=profile,
        )
        write_jsonl(risk_dir / path.name, risk)
    return risk_dir


def prediction_only_interface_is_gt_free():
    parameters = set(inspect.signature(compute_prediction_only_risk).parameters)
    parameters.update(inspect.signature(run_prediction_only_phase).parameters)
    forbidden = {
        "gt", "ground_truth", "visibility", "iou", "failure_label",
        "target_visible", "oracle_mask",
    }
    return not parameters.intersection(forbidden)


def target_id(sequence):
    return str(sequence).rsplit("-", 1)[0]


def target_group_folds(sequences):
    groups = defaultdict(list)
    for sequence in sequences:
        groups[target_id(sequence)].append(sequence)
    folds = []
    all_sequences = set(sequences)
    for held_out in sorted(groups):
        held_sequences = sorted(groups[held_out])
        folds.append({
            "held_out_target": held_out,
            "train_sequences": sorted(all_sequences - set(held_sequences)),
            "held_out_sequences": held_sequences,
        })
    return folds


def sequence_dataset_paths(dataset_root, sequence):
    target = target_id(sequence)
    root = Path(dataset_root) / target / sequence
    return {
        "root": root,
        "gt": root / "groundtruth.txt",
        "prediction": None,
    }


def load_risk_jsonl(path):
    records = load_diagnostics_jsonl(path)
    for expected_frame, record in enumerate(records):
        if int(record.get("frame_id", -1)) != expected_frame:
            raise ValueError("risk score frame_id mismatch: {}".format(path))
        forbidden = FORBIDDEN_RISK_OUTPUT_FIELDS.intersection(record)
        if forbidden:
            raise ValueError(
                "prediction-only risk file contains GT fields: {}".format(
                    sorted(forbidden)
                )
            )
    return records


def load_validation_sequence_data(args, risk_dir):
    """Stage B: GT access starts here, after risk JSONL exists on disk."""
    data = {}
    for sequence in validation_sequences():
        paths = sequence_dataset_paths(args.dataset_root, sequence)
        gt = load_bbox_file(paths["gt"])
        prediction_path = Path(args.prediction_dir) / "{}.txt".format(sequence)
        prediction = load_prediction_file(prediction_path)
        risk_path = Path(risk_dir) / "{}.jsonl".format(sequence)
        risk_records = load_risk_jsonl(risk_path)
        validate_lengths(len(gt), prediction=prediction, risk=risk_records)
        visibility = load_visibility(paths["root"], len(gt))
        iou = bbox_iou_xywh(prediction, gt)
        center_error = normalized_center_error(prediction, gt)
        labels = construct_event_labels(
            iou,
            center_error,
            visible=visibility["visible"],
            iou_fail=0.1,
            iou_recover=0.5,
            center_error_fail=1.0,
            k_fail=3,
            k_recover=3,
            failure_mode="either",
        )
        data[sequence] = {
            "sequence": sequence,
            "target": target_id(sequence),
            "risk": risk_records,
            "gt": gt,
            "prediction": prediction,
            "prediction_path": prediction_path,
            "gt_path": paths["gt"],
            "risk_path": risk_path,
            "visibility": visibility,
            "iou": iou,
            "center_error": center_error,
            "labels": labels,
        }
    return data


def simulate_profile(risk_records, profile):
    machine = ProvisionalRiskStateMachine(profile)
    states = []
    reasons = []
    for record in risk_records:
        state, reason = machine.update(
            float(record["risk_ema"]),
            int(record["high_risk_signal_count"]),
            warmup=bool(record.get("warmup", False)),
        )
        states.append(state)
        reasons.append(reason)
    return np.asarray(states), reasons


def safe_ratio(numerator, denominator):
    return float(numerator) / max(int(denominator), 1)


def binary_frame_metrics(predicted, target, scores, mask):
    predicted = np.asarray(predicted, dtype=bool)[mask]
    target = np.asarray(target, dtype=bool)[mask]
    scores = np.asarray(scores, dtype=np.float64)[mask]
    true_positive = int(np.logical_and(predicted, target).sum())
    false_positive = int(np.logical_and(predicted, ~target).sum())
    false_negative = int(np.logical_and(~predicted, target).sum())
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(precision + recall, EPSILON)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(target, scores),
        "pr_auc": pr_auc(target, scores),
        "false_lost_per_1000": 1000.0 * false_positive / max(int((~target).sum()), 1),
        "lost_frame_ratio": float(predicted.mean()) if len(predicted) else 0.0,
        "frame_count": int(len(predicted)),
        "failure_frame_count": int(target.sum()),
    }


def state_onsets(states, state):
    return [start for start, _ in event_runs(np.asarray(states) == state)]


def event_metrics_for_sequence(data, states):
    failure_events = event_runs(data["labels"]["stable_failure"])
    recovery_events = event_runs(data["labels"]["stable_recovery"])
    lost_onsets = state_onsets(states, "LOST")
    uncertain_mask = states == "UNCERTAIN"
    recover_onsets = state_onsets(states, "RECOVER")
    matched_lost = set()
    delays = []
    warning_5 = 0
    warning_10 = 0
    for event_index, (start, end) in enumerate(failure_events):
        candidates = [
            onset for onset in lost_onsets
            if start - 10 <= onset <= end and onset not in matched_lost
        ]
        if candidates:
            onset = min(candidates, key=lambda item: abs(item - start))
            matched_lost.add(onset)
            delays.append(onset - start)
        warning_5 += int(uncertain_mask[max(0, start - 5):start].any())
        warning_10 += int(uncertain_mask[max(0, start - 10):start].any())

    matched_recover = set()
    recovery_delays = []
    for start, _ in recovery_events:
        candidates = [
            onset for onset in recover_onsets
            if start <= onset <= start + 10 and onset not in matched_recover
        ]
        if candidates:
            onset = min(candidates)
            matched_recover.add(onset)
            recovery_delays.append(onset - start)

    state_durations = [
        end - start + 1 for start, end in event_runs(states == "LOST")
    ]
    transitions = int(np.sum(states[1:] != states[:-1])) if len(states) > 1 else 0
    detected = len(delays)
    false_events = len(lost_onsets) - len(matched_lost)
    event_precision = safe_ratio(detected, detected + false_events)
    event_recall = safe_ratio(detected, len(failure_events))
    recover_detected = len(recovery_delays)
    recover_false = len(recover_onsets) - len(matched_recover)
    return {
        "sequence": data["sequence"],
        "target": data["target"],
        "failure_events": len(failure_events),
        "detected_failure_events": detected,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "false_event_triggers": false_events,
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "median_detection_delay": float(np.median(delays)) if delays else None,
        "detection_delays": delays,
        "uncertain_warning_1_5": warning_5,
        "uncertain_warning_1_10": warning_10,
        "recovery_events": len(recovery_events),
        "detected_recovery_events": recover_detected,
        "recovery_precision": safe_ratio(
            recover_detected, recover_detected + recover_false
        ),
        "recovery_recall": safe_ratio(recover_detected, len(recovery_events)),
        "recovery_delay": float(np.mean(recovery_delays))
        if recovery_delays else None,
        "recovery_delays": recovery_delays,
        "false_recovery_triggers": recover_false,
        "lost_state_count": len(state_durations),
        "lost_mean_duration": float(np.mean(state_durations))
        if state_durations else 0.0,
        "lost_median_duration": float(np.median(state_durations))
        if state_durations else 0.0,
        "transition_count": transitions,
        "transition_rate": safe_ratio(transitions, max(len(states) - 1, 1)),
    }


def aggregate_event_metrics(rows):
    failure_events = sum(row["failure_events"] for row in rows)
    detected = sum(row["detected_failure_events"] for row in rows)
    false_events = sum(row["false_event_triggers"] for row in rows)
    recovery_events = sum(row["recovery_events"] for row in rows)
    recovery_detected = sum(row["detected_recovery_events"] for row in rows)
    false_recovery = sum(row["false_recovery_triggers"] for row in rows)
    delays = [value for row in rows for value in row["detection_delays"]]
    recovery_delays = [
        value for row in rows for value in row["recovery_delays"]
    ]
    return {
        "failure_events": failure_events,
        "detected_failure_events": detected,
        "event_precision": safe_ratio(detected, detected + false_events),
        "event_recall": safe_ratio(detected, failure_events),
        "false_event_triggers": false_events,
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "median_detection_delay": float(np.median(delays)) if delays else None,
        "uncertain_warning_1_5_ratio": safe_ratio(
            sum(row["uncertain_warning_1_5"] for row in rows), failure_events
        ),
        "uncertain_warning_1_10_ratio": safe_ratio(
            sum(row["uncertain_warning_1_10"] for row in rows), failure_events
        ),
        "recovery_events": recovery_events,
        "detected_recovery_events": recovery_detected,
        "recovery_precision": safe_ratio(
            recovery_detected, recovery_detected + false_recovery
        ),
        "recovery_recall": safe_ratio(recovery_detected, recovery_events),
        "recovery_delay": float(np.mean(recovery_delays))
        if recovery_delays else None,
        "false_recovery_triggers": false_recovery,
        "transition_count": sum(row["transition_count"] for row in rows),
        "max_transition_rate": max(
            [row["transition_rate"] for row in rows] or [0.0]
        ),
    }


def evaluate_config(sequence_data, sequences, profile):
    collected = {
        "states": [], "scores": [], "failure": [], "visible": [],
    }
    event_rows = []
    states_by_sequence = {}
    reasons_by_sequence = {}
    for sequence in sequences:
        data = sequence_data[sequence]
        states, reasons = simulate_profile(data["risk"], profile)
        states_by_sequence[sequence] = states
        reasons_by_sequence[sequence] = reasons
        collected["states"].extend(states)
        collected["scores"].extend(record["risk_ema"] for record in data["risk"])
        collected["failure"].extend(data["labels"]["stable_failure"])
        visible = data["visibility"]["visible"]
        if visible is None:
            visible = np.ones(len(states), dtype=bool)
        collected["visible"].extend(visible)
        event_rows.append(event_metrics_for_sequence(data, states))

    states = np.asarray(collected["states"])
    predicted = states == "LOST"
    failure = np.asarray(collected["failure"], dtype=bool)
    scores = np.asarray(collected["scores"], dtype=np.float64)
    visible = np.asarray(collected["visible"], dtype=bool)
    scopes = {
        "visible": visible,
        "occluded_or_out_of_view": ~visible,
        "overall": np.ones(len(states), dtype=bool),
    }
    frame_metrics = {
        scope: binary_frame_metrics(predicted, failure, scores, mask)
        for scope, mask in scopes.items()
    }
    return {
        "frame": frame_metrics,
        "event": aggregate_event_metrics(event_rows),
        "event_rows": event_rows,
        "states": states_by_sequence,
        "reasons": reasons_by_sequence,
    }


def profile_candidates():
    candidates = []
    grid_keys = tuple(SEARCH_GRID)
    for profile_name, defaults in PROFILE_DEFAULTS.items():
        for values in itertools.product(*(SEARCH_GRID[key] for key in grid_keys)):
            candidate = dict(defaults)
            candidate.update(dict(zip(grid_keys, values)))
            if not (
                candidate["lost_enter_threshold"]
                > candidate["uncertain_enter_threshold"]
                > candidate["lost_exit_threshold"]
            ):
                continue
            candidate["profile_family"] = profile_name
            candidates.append(candidate)
    return candidates


def selection_score(result):
    frame = result["frame"]["visible"]
    event = result["event"]
    delay = event["median_detection_delay"]
    delay_penalty = 15.0 if delay is None else max(float(delay), 0.0)
    return (
        1.5 * frame["f1"]
        + event["event_recall"]
        + 0.25 * event["uncertain_warning_1_10_ratio"]
        - 0.002 * frame["false_lost_per_1000"]
        - 0.01 * delay_penalty
        - 0.5 * max(frame["lost_frame_ratio"] - 0.15, 0.0)
        - 0.2 * event["max_transition_rate"]
    )


def choose_fold_profiles(sequence_data):
    folds = target_group_folds(sorted(sequence_data))
    candidates = profile_candidates()
    selections = []
    for fold in folds:
        best_profile = None
        best_score = None
        best_train = None
        for candidate in candidates:
            result = evaluate_config(
                sequence_data, fold["train_sequences"], candidate
            )
            score = selection_score(result)
            tie_key = (
                score,
                result["event"]["event_recall"],
                result["frame"]["visible"]["precision"],
                -result["frame"]["visible"]["false_lost_per_1000"],
            )
            if best_score is None or tie_key > best_score:
                best_score = tie_key
                best_profile = dict(candidate)
                best_train = result
        held_out = evaluate_config(
            sequence_data, fold["held_out_sequences"], best_profile
        )
        selections.append({
            **fold,
            "selected_profile": best_profile,
            "selection_score": best_score[0],
            "train_result": best_train,
            "held_out_result": held_out,
        })
    return selections


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(prefix, metrics):
    return {"{}_{}".format(prefix, key): value for key, value in metrics.items()}


def build_oof_outputs(sequence_data, selections):
    frame_rows = []
    event_rows = []
    collected_states = []
    collected_scores = []
    collected_failure = []
    collected_visible = []
    for selection in selections:
        profile = selection["selected_profile"]
        profile_name = profile["profile_family"]
        result = selection["held_out_result"]
        for event_row in result["event_rows"]:
            row = dict(event_row)
            row["held_out_target"] = selection["held_out_target"]
            row["selected_profile_family"] = profile_name
            row["selected_profile"] = json.dumps(profile, sort_keys=True)
            event_rows.append(row)
        for sequence in selection["held_out_sequences"]:
            data = sequence_data[sequence]
            states = result["states"][sequence]
            reasons = result["reasons"][sequence]
            visible = data["visibility"]["visible"]
            if visible is None:
                visible = np.ones(len(states), dtype=bool)
            for frame_id, (record, state, reason) in enumerate(
                zip(data["risk"], states, reasons)
            ):
                frame_rows.append({
                    "sequence": sequence,
                    "target": data["target"],
                    "held_out_target": selection["held_out_target"],
                    "frame_id": frame_id,
                    "visible": bool(visible[frame_id]),
                    "occluded": bool(data["visibility"]["occluded"][frame_id])
                    if data["visibility"]["available"] else None,
                    "out_of_view": bool(
                        data["visibility"]["out_of_view"][frame_id]
                    ) if data["visibility"]["available"] else None,
                    "iou": float(data["iou"][frame_id]),
                    "center_error_norm": float(data["center_error"][frame_id]),
                    "stable_failure": bool(
                        data["labels"]["stable_failure"][frame_id]
                    ),
                    "stable_recovery": bool(
                        data["labels"]["stable_recovery"][frame_id]
                    ),
                    "risk_ema": float(record["risk_ema"]),
                    "instantaneous_risk": float(record["instantaneous_risk"]),
                    "high_risk_signal_count": int(
                        record["high_risk_signal_count"]
                    ),
                    "warmup": bool(record["warmup"]),
                    "provisional_state": state,
                    "transition_reason": reason,
                    "selected_profile_family": profile_name,
                })
            collected_states.extend(states)
            collected_scores.extend(record["risk_ema"] for record in data["risk"])
            collected_failure.extend(data["labels"]["stable_failure"])
            collected_visible.extend(visible)

    states = np.asarray(collected_states)
    predicted = states == "LOST"
    scores = np.asarray(collected_scores, dtype=np.float64)
    failure = np.asarray(collected_failure, dtype=bool)
    visible = np.asarray(collected_visible, dtype=bool)
    frame_metrics = {
        "visible": binary_frame_metrics(predicted, failure, scores, visible),
        "occluded_or_out_of_view": binary_frame_metrics(
            predicted, failure, scores, ~visible
        ),
        "overall": binary_frame_metrics(
            predicted, failure, scores, np.ones(len(states), dtype=bool)
        ),
    }
    return {
        "frame_rows": frame_rows,
        "event_rows": event_rows,
        "frame_metrics": frame_metrics,
        "event_metrics": aggregate_event_metrics(event_rows),
    }


def loto_metric_rows(selections, oof):
    rows = []
    for selection in selections:
        result = selection["held_out_result"]
        for scope, metrics in result["frame"].items():
            rows.append({
                "held_out_target": selection["held_out_target"],
                "held_out_sequences": ",".join(selection["held_out_sequences"]),
                "scope": scope,
                "selected_profile_family": selection[
                    "selected_profile"
                ]["profile_family"],
                "selected_profile": json.dumps(
                    selection["selected_profile"], sort_keys=True
                ),
                "selection_score": selection["selection_score"],
                **metrics,
                **flatten_metrics("event", result["event"]),
            })
    for scope, metrics in oof["frame_metrics"].items():
        rows.append({
            "held_out_target": "OOF_AGGREGATE",
            "held_out_sequences": "all validation sequences, grouped OOF",
            "scope": scope,
            "selected_profile_family": "per-fold selected",
            "selected_profile": "see risk_profile_candidates.yaml",
            "selection_score": None,
            **metrics,
            **flatten_metrics("event", oof["event_metrics"]),
        })
    return rows


def component_statistics(sequence_data):
    rows = []
    fields = list(RAW_SIGNAL_ALIASES) + [
        name + "_risk" for name in RAW_SIGNAL_ALIASES
    ] + ["instantaneous_risk", "risk_ema"]
    for field in fields:
        values = []
        labels = []
        for data in sequence_data.values():
            for index, record in enumerate(data["risk"]):
                if field in RAW_SIGNAL_ALIASES:
                    value = record["raw_signals"].get(field)
                elif field.endswith("_risk"):
                    value = record["normalized_risks"].get(field)
                else:
                    value = record.get(field)
                value = finite_float(value)
                if value is not None:
                    values.append(value)
                    labels.append(bool(data["labels"]["stable_failure"][index]))
        values = np.asarray(values, dtype=np.float64)
        labels = np.asarray(labels, dtype=bool)
        failure_values = values[labels] if len(values) else np.asarray([])
        normal_values = values[~labels] if len(values) else np.asarray([])
        rows.append({
            "component": field,
            "available_count": int(len(values)),
            "failure_mean": float(failure_values.mean())
            if failure_values.size else None,
            "failure_median": float(np.median(failure_values))
            if failure_values.size else None,
            "normal_mean": float(normal_values.mean())
            if normal_values.size else None,
            "normal_median": float(np.median(normal_values))
            if normal_values.size else None,
            "roc_auc_high_value": roc_auc(labels, values) if len(values) else None,
            "pr_auc_high_value": pr_auc(labels, values) if len(values) else None,
        })
    return rows


def select_failure_examples(sequence_data, oof, limit=3):
    categories = defaultdict(list)
    frames_by_sequence = defaultdict(list)
    for row in oof["frame_rows"]:
        frames_by_sequence[row["sequence"]].append(row)
    for event_row in oof["event_rows"]:
        sequence = event_row["sequence"]
        rows = frames_by_sequence[sequence]
        data = sequence_data[sequence]
        failure_events = event_runs(data["labels"]["stable_failure"])
        for start, end in failure_events:
            lost = [
                row for row in rows[max(0, start - 10):end + 1]
                if row["provisional_state"] == "LOST"
            ]
            category = "detected failure" if lost else "missed failure"
            representative = lost[0] if lost else rows[start]
            categories[category].append((representative, start, end, data))
        for start, end in event_runs(
            np.asarray([row["provisional_state"] == "LOST" for row in rows])
        ):
            if not any(row["stable_failure"] for row in rows[start:end + 1]):
                categories["false LOST"].append((rows[start], start, end, data))
    return {
        category: examples[:limit] for category, examples in categories.items()
    }


def write_examples(path, examples):
    with Path(path).open("w", encoding="utf-8") as file_handle:
        file_handle.write("# Continuous Risk Failure Examples\n\n")
        file_handle.write(
            "Validation-only grouped OOF diagnostics; paths and frame ranges "
            "only, with no generated images.\n\n"
        )
        for category in ("detected failure", "missed failure", "false LOST"):
            file_handle.write("## {}\n\n".format(category))
            selected = examples.get(category, [])
            if not selected:
                file_handle.write("- No matching example.\n\n")
                continue
            for row, start, end, data in selected:
                file_handle.write(
                    "- `{}` frames `{}-{}`; representative frame `{}`; "
                    "state `{}`; risk EMA `{:.4f}`; IoU `{:.4f}`; center "
                    "error `{:.4f}`; prediction `{}`; GT `{}`; risk `{}`.\n".format(
                        row["sequence"], start, end, row["frame_id"],
                        row["provisional_state"], row["risk_ema"], row["iou"],
                        row["center_error_norm"], data["prediction_path"],
                        data["gt_path"], data["risk_path"],
                    )
                )
            file_handle.write("\n")


def yaml_safe_value(value):
    if isinstance(value, dict):
        return {key: yaml_safe_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe_value(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def pass_gate(oof, selections):
    frame = oof["frame_metrics"]["visible"]
    event = oof["event_metrics"]
    median_delay = event["median_detection_delay"]
    collapsed_targets = [
        selection["held_out_target"] for selection in selections
        if selection["held_out_result"]["frame"]["overall"]["lost_frame_ratio"]
        >= 0.95
    ]
    criteria = {
        "lost_precision_ge_0_60": frame["precision"] >= 0.60,
        "failure_event_recall_ge_0_40": event["event_recall"] >= 0.40,
        "false_lost_per_1000_le_50": frame["false_lost_per_1000"] <= 50.0,
        "median_delay_le_10": median_delay is not None and median_delay <= 10.0,
        "uncertain_warning_1_10_ge_0_25": (
            event["uncertain_warning_1_10_ratio"] >= 0.25
        ),
        "lost_frame_ratio_lt_0_15": frame["lost_frame_ratio"] < 0.15,
        "no_held_out_target_collapse": not collapsed_targets,
        "no_framewise_oscillation": event["max_transition_rate"] < 0.10,
    }
    return criteria, collapsed_targets


def write_summary(path, args, oof, selections):
    frame = oof["frame_metrics"]["visible"]
    event = oof["event_metrics"]
    criteria, collapsed_targets = pass_gate(oof, selections)
    passed = all(criteria.values())
    with Path(path).open("w", encoding="utf-8") as file_handle:
        file_handle.write("# M0-C Continuous Motion Risk Summary\n\n")
        file_handle.write("- Result label: **validation diagnostic result**.\n")
        file_handle.write(
            "- Risk computation is entirely prediction-only and causal; GT "
            "is read only after per-sequence risk JSONL files are saved.\n"
        )
        file_handle.write(
            "- GT is used only for grouped leave-one-target-out validation "
            "evaluation and cannot affect risk scores.\n"
        )
        file_handle.write(
            "- Split: `threemdot_val`; this report is not a test result and "
            "does not authorize real M1 expanded search.\n\n"
        )
        file_handle.write("## Grouped OOF Metrics\n\n")
        file_handle.write(
            "| Precision | Recall | F1 | ROC-AUC | PR-AUC | False LOST/1000 | "
            "LOST ratio | Event recall | Median delay | UNCERTAIN 1-10 |\n"
        )
        file_handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        file_handle.write(
            "| {precision:.4f} | {recall:.4f} | {f1:.4f} | {roc_auc} | "
            "{pr_auc} | {false_lost_per_1000:.2f} | {lost_frame_ratio:.4f} | "
            "{event_recall:.4f} | {median_detection_delay} | "
            "{uncertain_warning_1_10_ratio:.4f} |\n".format(
                **frame, **event
            )
        )
        file_handle.write("\n## Scope Breakdown\n\n")
        for scope, metrics in oof["frame_metrics"].items():
            file_handle.write(
                "- `{}`: precision `{:.4f}`, recall `{:.4f}`, false "
                "LOST/1000 `{:.2f}`, LOST ratio `{:.4f}`.\n".format(
                    scope, metrics["precision"], metrics["recall"],
                    metrics["false_lost_per_1000"], metrics["lost_frame_ratio"],
                )
            )
        file_handle.write("\n## Baseline Diagnostics\n\n")
        file_handle.write(
            "- Initial M0: LOST precision `0.8911`, recall `0.2308`, "
            "19/20 events missed, mean delay `31`.\n"
        )
        file_handle.write(
            "- M0.5: LOST precision `0.0574`, false LOST `488.23/1000`, "
            "LOST ratio `50.90%`, 0/20 events detected.\n"
        )
        file_handle.write(
            "- Relative conclusion: M0-C is considered better than both M0 "
            "and M0.5 only when every grouped OOF gate below passes; current "
            "result: **{}**. No in-sample metric may be substituted.\n"
            "- Recovery events are defined and scored on visible frames; "
            "occluded/out-of-view and overall state occupancy are reported "
            "separately above.\n\n".format("better" if passed else "not better")
        )
        file_handle.write("## Gate\n\n")
        for name, result in criteria.items():
            file_handle.write(
                "- `{}`: **{}**.\n".format(name, "PASS" if result else "FAIL")
            )
        file_handle.write(
            "- Held-out collapsed targets: `{}`.\n".format(collapsed_targets)
        )
        file_handle.write("\n## Decision\n\n")
        if passed:
            file_handle.write(
                "Grouped OOF gate passed. M0-C has value for a future Shadow "
                "Mode integration audit, but real M1 expanded search remains "
                "forbidden at this stage.\n"
            )
        else:
            file_handle.write(
                "Grouped OOF gate failed. Stop prediction-only state-detector "
                "weight/threshold search and move to low-frequency scheduled "
                "expanded search with independent candidate verification.\n"
            )


def run_validation_gt_phase(args, risk_dir):
    validate_dataset(args.dataset)
    output_dir = Path(args.output_dir).resolve()
    sequence_data = load_validation_sequence_data(args, risk_dir)
    selections = choose_fold_profiles(sequence_data)
    oof = build_oof_outputs(sequence_data, selections)

    write_csv(output_dir / "frame_level_risk_metrics.csv", oof["frame_rows"])
    write_csv(output_dir / "event_level_risk_metrics.csv", oof["event_rows"])
    write_csv(
        output_dir / "leave_one_target_out_metrics.csv",
        loto_metric_rows(selections, oof),
    )
    write_csv(
        output_dir / "signal_component_statistics.csv",
        component_statistics(sequence_data),
    )
    candidate_output = {
        "provenance": (
            "validation-derived grouped leave-one-target-out on threemdot_val"
        ),
        "warning": (
            "provisional offline diagnostics only; do not write to tracker "
            "config or use for test selection"
        ),
        "fixed_signal_weights": CORE_WEIGHTS,
        "fixed_profile_families": PROFILE_DEFAULTS,
        "search_grid": SEARCH_GRID,
        "folds": [
            {
                "held_out_target": item["held_out_target"],
                "train_sequences": item["train_sequences"],
                "held_out_sequences": item["held_out_sequences"],
                "selected_profile": item["selected_profile"],
                "selection_score": item["selection_score"],
            }
            for item in selections
        ],
    }
    with (output_dir / "risk_profile_candidates.yaml").open(
        "w", encoding="utf-8"
    ) as file_handle:
        yaml.safe_dump(
            yaml_safe_value(candidate_output), file_handle, sort_keys=False
        )
    write_examples(
        output_dir / "risk_failure_examples.md",
        select_failure_examples(sequence_data, oof),
    )
    write_summary(
        output_dir / "motion_risk_summary.md", args, oof, selections
    )


def main(argv=None):
    args = parse_args(argv)
    validate_dataset(args.dataset)
    if not prediction_only_interface_is_gt_free():
        raise AssertionError("prediction-only risk interface accepts GT inputs")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage A completes and persists before any GT loader is called.
    risk_dir = run_prediction_only_phase(
        diagnostics_dir=args.diagnostics_dir,
        output_dir=args.output_dir,
        normalization_window=args.normalization_window,
        min_history=args.min_history,
        ema_alpha=args.ema_alpha,
        provisional_profile=args.provisional_profile,
    )

    # Stage B reads only the persisted risk scores plus validation predictions/GT.
    run_validation_gt_phase(args, risk_dir)


if __name__ == "__main__":
    main()
