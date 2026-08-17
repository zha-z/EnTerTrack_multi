import json
import math
import os
from collections import Counter, deque
from enum import Enum

import numpy as np
import torch

from lib.test.utils.pcum_diagnostics import normalized_response_entropy


class MotionState(str, Enum):
    NORMAL = "NORMAL"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"
    RECOVER = "RECOVER"


MOTION_DIAGNOSTIC_FIELDS = (
    "frame_id",
    "predicted_bbox",
    "predicted_center",
    "predicted_motion_center",
    "normalized_motion_residual",
    "motion_residual_normalization",
    "bbox_width",
    "bbox_height",
    "max_score",
    "apce",
    "response_entropy",
    "response_top1_top2_gap",
    "response_peak_sharpness",
    "bbox_border_proximity",
    "bbox_border_event",
    "search_region_border_proximity",
    "search_region_border_event",
    "remote_quality",
    "remote_weight_entropy",
    "remote_max_weight",
    "valid_remote_count",
    "velocity",
    "low_quality_consecutive_count",
    "recover_consecutive_count",
    "lost_duration",
    "uncertainty_radius",
    "previous_state",
    "state",
    "transition_reason",
    "low_quality_reasons",
    "missing_fields",
)


def _finite_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.detach().reshape(-1)[0].cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bbox_values(bbox):
    if bbox is None:
        raise ValueError("predicted bbox is required")
    values = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if values.size < 4 or not np.isfinite(values[:4]).all():
        raise ValueError("predicted bbox must contain four finite values")
    return values[:4]


def _response_statistics(response, border_margin):
    unavailable = {
        "response_entropy": None,
        "response_top1_top2_gap": None,
        "response_peak_sharpness": None,
        "search_region_border_proximity": None,
        "search_region_border_event": None,
    }
    if response is None or not torch.is_tensor(response) or response.numel() == 0:
        return unavailable

    values = response.detach().float()
    if values.ndim < 2:
        return unavailable
    spatial = values.reshape(-1, values.shape[-2], values.shape[-1])[0]
    flat = spatial.reshape(-1)
    top_k = min(2, flat.numel())
    top_values, top_indices = torch.topk(flat, k=top_k)
    top1 = float(top_values[0].cpu().item())
    top2 = float(top_values[1].cpu().item()) if top_k == 2 else top1
    std = float(flat.std(unbiased=False).cpu().item())
    mean = float(flat.mean().cpu().item())

    height, width = spatial.shape
    peak_index = int(top_indices[0].cpu().item())
    peak_y, peak_x = divmod(peak_index, width)
    x_proximity = min(peak_x, width - 1 - peak_x) / max(width - 1, 1)
    y_proximity = min(peak_y, height - 1 - peak_y) / max(height - 1, 1)
    proximity = float(min(x_proximity, y_proximity))

    return {
        "response_entropy": _finite_float(normalized_response_entropy(response)),
        "response_top1_top2_gap": top1 - top2,
        "response_peak_sharpness": (top1 - mean) / max(std, 1e-12),
        "search_region_border_proximity": proximity,
        "search_region_border_event": bool(proximity <= border_margin),
    }


class MotionStateManager:
    """Prediction-only motion state estimator used only for shadow logging."""

    def __init__(
        self,
        velocity_ema=0.8,
        max_history=10,
        score_low=0.0,
        score_recover=0.0,
        apce_low=0.0,
        apce_recover=0.0,
        motion_residual_high=0.0,
        border_margin=0.1,
        k_lost=3,
        k_normal=2,
        k_recover=3,
    ):
        if not 0.0 <= float(velocity_ema) <= 1.0:
            raise ValueError("velocity_ema must be in [0, 1]")
        self.velocity_ema = float(velocity_ema)
        self.max_history = max(2, int(max_history))
        self.score_low = float(score_low)
        self.score_recover = float(score_recover)
        self.apce_low = float(apce_low)
        self.apce_recover = float(apce_recover)
        self.motion_residual_high = float(motion_residual_high)
        self.border_margin = max(0.0, float(border_margin))
        self.k_lost = max(1, int(k_lost))
        self.k_normal = max(1, int(k_normal))
        self.k_recover = max(1, int(k_recover))
        self.reset()

    @classmethod
    def from_config(cls, config):
        return cls(
            velocity_ema=getattr(config, "VELOCITY_EMA", 0.8),
            max_history=getattr(config, "MAX_HISTORY", 10),
            score_low=getattr(config, "SCORE_LOW", 0.0),
            score_recover=getattr(config, "SCORE_RECOVER", 0.0),
            apce_low=getattr(config, "APCE_LOW", 0.0),
            apce_recover=getattr(config, "APCE_RECOVER", 0.0),
            motion_residual_high=getattr(config, "MOTION_RESIDUAL_HIGH", 0.0),
            border_margin=getattr(config, "BORDER_MARGIN", 0.1),
            k_lost=getattr(config, "K_LOST", 3),
            k_normal=getattr(config, "K_NORMAL", 2),
            k_recover=getattr(config, "K_RECOVER", 3),
        )

    def reset(self, initial_bbox=None, image_size=None):
        self.state = MotionState.NORMAL
        self.previous_state = MotionState.NORMAL
        self.reliable_centers = deque(maxlen=self.max_history)
        self.reliable_sizes = deque(maxlen=self.max_history)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.low_quality_count = 0
        self.recover_count = 0
        self.lost_duration = 0
        self.latest_diagnostics = None
        if initial_bbox is not None:
            bbox = _bbox_values(initial_bbox)
            center = bbox[:2] + 0.5 * bbox[2:4]
            self.reliable_centers.append(center)
            self.reliable_sizes.append(bbox[2:4].copy())
            self.latest_diagnostics = self._initial_record(bbox, image_size)
        return self.get_diagnostics()

    def _initial_record(self, bbox, image_size):
        center = bbox[:2] + 0.5 * bbox[2:4]
        border_proximity, border_event = self._bbox_border(bbox, image_size)
        record = {
            "frame_id": 0,
            "predicted_bbox": bbox.tolist(),
            "predicted_center": center.tolist(),
            "predicted_motion_center": center.tolist(),
            "normalized_motion_residual": 0.0,
            "motion_residual_normalization": "predicted_bbox_diagonal",
            "bbox_width": float(bbox[2]),
            "bbox_height": float(bbox[3]),
            "max_score": None,
            "apce": None,
            "response_entropy": None,
            "response_top1_top2_gap": None,
            "response_peak_sharpness": None,
            "bbox_border_proximity": border_proximity,
            "bbox_border_event": border_event,
            "search_region_border_proximity": None,
            "search_region_border_event": None,
            "remote_quality": None,
            "remote_weight_entropy": None,
            "remote_max_weight": None,
            "valid_remote_count": None,
            "velocity": self.velocity.tolist(),
            "low_quality_consecutive_count": 0,
            "recover_consecutive_count": 0,
            "lost_duration": 0,
            "uncertainty_radius": 0.5 * float(np.linalg.norm(bbox[2:4])),
            "previous_state": MotionState.NORMAL.value,
            "state": MotionState.NORMAL.value,
            "transition_reason": "initialization",
            "low_quality_reasons": [],
            "missing_fields": [
                "max_score", "apce", "response_entropy",
                "response_top1_top2_gap", "response_peak_sharpness",
                "search_region_border_proximity", "remote_quality",
                "remote_weight_entropy", "remote_max_weight",
                "valid_remote_count",
            ],
        }
        return record

    def _bbox_border(self, bbox, image_size):
        if image_size is None:
            return None, None
        height, width = image_size[:2]
        if height <= 0 or width <= 0:
            return None, None
        x, y, box_w, box_h = bbox
        distances = (
            x / width,
            y / height,
            (width - (x + box_w)) / width,
            (height - (y + box_h)) / height,
        )
        proximity = float(max(0.0, min(distances)))
        return proximity, bool(proximity <= self.border_margin)

    def _threshold_low(self, value, threshold):
        return threshold > 0.0 and value is not None and value < threshold

    def _threshold_recovered(self, value, threshold):
        return threshold <= 0.0 or (value is not None and value >= threshold)

    def _transition(self, low_quality, recovered, reasons):
        old_state = self.state
        reason = "stable"

        if self.state == MotionState.NORMAL:
            if low_quality:
                self.state = MotionState.UNCERTAIN
                self.low_quality_count = 1
                self.recover_count = 0
                reason = "normal_to_uncertain:" + ",".join(reasons)
            else:
                self.low_quality_count = 0
        elif self.state == MotionState.UNCERTAIN:
            if low_quality:
                self.low_quality_count += 1
                self.recover_count = 0
                if self.low_quality_count >= self.k_lost:
                    self.state = MotionState.LOST
                    self.lost_duration = 1
                    reason = "uncertain_to_lost:k_low_quality"
                else:
                    reason = "uncertain_low_quality:" + ",".join(reasons)
            elif recovered:
                self.recover_count += 1
                if self.recover_count >= self.k_normal:
                    self.state = MotionState.NORMAL
                    self.low_quality_count = 0
                    self.recover_count = 0
                    reason = "uncertain_to_normal:k_recovered"
                else:
                    reason = "uncertain_recovery_pending"
        elif self.state == MotionState.LOST:
            if recovered and not low_quality:
                self.state = MotionState.RECOVER
                self.recover_count = 1
                self.low_quality_count = 0
                reason = "lost_to_recover:quality_recovered"
            else:
                self.lost_duration += 1
                reason = "lost_persistent"
        elif self.state == MotionState.RECOVER:
            if low_quality or not recovered:
                self.state = MotionState.LOST
                self.low_quality_count = 1
                self.recover_count = 0
                self.lost_duration = max(1, self.lost_duration + 1)
                reason = "recover_to_lost:" + ",".join(reasons or ["quality_not_recovered"])
            else:
                self.recover_count += 1
                if self.recover_count >= self.k_recover:
                    self.state = MotionState.NORMAL
                    self.recover_count = 0
                    self.lost_duration = 0
                    reason = "recover_to_normal:k_stable"
                else:
                    reason = "recover_stability_pending"

        self.previous_state = old_state
        return reason

    def update_prediction_only(
        self,
        frame_id,
        predicted_bbox,
        max_score=None,
        apce=None,
        response=None,
        image_size=None,
        remote_quality=None,
        remote_weight_entropy=None,
        remote_max_weight=None,
        valid_remote_count=None,
    ):
        bbox = _bbox_values(predicted_bbox)
        center = bbox[:2] + 0.5 * bbox[2:4]

        if len(self.reliable_centers) >= 2:
            observed_velocity = self.reliable_centers[-1] - self.reliable_centers[-2]
            beta = self.velocity_ema
            self.velocity = beta * self.velocity + (1.0 - beta) * observed_velocity
        predicted_center = (
            self.reliable_centers[-1] + self.velocity
            if self.reliable_centers else center.copy()
        )
        bbox_diagonal = max(float(np.linalg.norm(bbox[2:4])), 1e-6)
        motion_residual = float(np.linalg.norm(center - predicted_center) / bbox_diagonal)

        score = _finite_float(max_score)
        apce_value = _finite_float(apce)
        remote_quality_value = _finite_float(remote_quality)
        remote_entropy_value = _finite_float(remote_weight_entropy)
        remote_max_value = _finite_float(remote_max_weight)
        valid_remote_value = _finite_float(valid_remote_count)
        response_stats = _response_statistics(response, self.border_margin)
        bbox_proximity, bbox_border_event = self._bbox_border(bbox, image_size)

        reasons = []
        if self._threshold_low(score, self.score_low):
            reasons.append("score_low")
        if self._threshold_low(apce_value, self.apce_low):
            reasons.append("apce_low")
        if self.motion_residual_high > 0.0 and motion_residual > self.motion_residual_high:
            reasons.append("motion_residual_high")
        if bbox_border_event:
            reasons.append("bbox_border")
        if response_stats["search_region_border_event"]:
            reasons.append("response_peak_border")
        low_quality = bool(reasons)
        recovered = (
            self._threshold_recovered(score, self.score_recover)
            and self._threshold_recovered(apce_value, self.apce_recover)
            and (
                self.motion_residual_high <= 0.0
                or motion_residual <= self.motion_residual_high
            )
            and not bool(bbox_border_event)
            and not bool(response_stats["search_region_border_event"])
        )
        transition_reason = self._transition(low_quality, recovered, reasons)

        if not low_quality and self.state != MotionState.LOST:
            self.reliable_centers.append(center.copy())
            self.reliable_sizes.append(bbox[2:4].copy())

        uncertainty_multiplier = 1.0 + 0.5 * self.low_quality_count + self.lost_duration
        uncertainty_radius = 0.5 * bbox_diagonal * uncertainty_multiplier
        record = {
            "frame_id": int(frame_id),
            "predicted_bbox": bbox.tolist(),
            "predicted_center": center.tolist(),
            "predicted_motion_center": predicted_center.tolist(),
            "normalized_motion_residual": motion_residual,
            "motion_residual_normalization": "predicted_bbox_diagonal",
            "bbox_width": float(bbox[2]),
            "bbox_height": float(bbox[3]),
            "max_score": score,
            "apce": apce_value,
            **response_stats,
            "bbox_border_proximity": bbox_proximity,
            "bbox_border_event": bbox_border_event,
            "remote_quality": remote_quality_value,
            "remote_weight_entropy": remote_entropy_value,
            "remote_max_weight": remote_max_value,
            "valid_remote_count": valid_remote_value,
            "velocity": self.velocity.tolist(),
            "low_quality_consecutive_count": int(self.low_quality_count),
            "recover_consecutive_count": int(self.recover_count),
            "lost_duration": int(self.lost_duration),
            "uncertainty_radius": float(uncertainty_radius),
            "previous_state": self.previous_state.value,
            "state": self.state.value,
            "transition_reason": transition_reason,
            "low_quality_reasons": reasons,
        }
        missing_candidates = (
            "max_score", "apce", "response_entropy",
            "response_top1_top2_gap", "response_peak_sharpness",
            "bbox_border_proximity", "search_region_border_proximity",
            "remote_quality", "remote_weight_entropy", "remote_max_weight",
            "valid_remote_count",
        )
        record["missing_fields"] = [
            field for field in missing_candidates if record.get(field) is None
        ]
        self.latest_diagnostics = record
        return self.get_diagnostics()

    def get_diagnostics(self):
        if self.latest_diagnostics is None:
            return None
        return dict(self.latest_diagnostics)


def summarize_motion_records(records):
    records = [record for record in records if isinstance(record, dict)]
    states = Counter(record.get("state", "UNKNOWN") for record in records)
    transitions = sum(
        1 for record in records
        if record.get("previous_state") != record.get("state")
    )
    longest_lost = 0
    current_lost = 0
    for record in records:
        if record.get("state") == MotionState.LOST.value:
            current_lost += 1
            longest_lost = max(longest_lost, current_lost)
        else:
            current_lost = 0

    def average(field):
        values = [_finite_float(record.get(field)) for record in records]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    missing = Counter()
    for record in records:
        missing.update(record.get("missing_fields", []))
    return {
        "frame_count": len(records),
        "state_counts": {
            state.value: int(states.get(state.value, 0)) for state in MotionState
        },
        "state_transition_count": int(transitions),
        "longest_lost_duration": int(longest_lost),
        "mean_score": average("max_score"),
        "mean_apce": average("apce"),
        "mean_response_entropy": average("response_entropy"),
        "mean_normalized_motion_residual": average("normalized_motion_residual"),
        "bbox_border_event_count": sum(
            bool(record.get("bbox_border_event")) for record in records
        ),
        "search_region_border_event_count": sum(
            bool(record.get("search_region_border_event")) for record in records
        ),
        "missing_diagnostic_fields": dict(sorted(missing.items())),
    }


def motion_diagnostics_file(results_dir, sequence_name):
    return os.path.join(
        results_dir,
        "motion_state_diagnostics",
        "{}.jsonl".format(sequence_name),
    )


def save_motion_diagnostics(results_dir, sequence_name, records):
    """Write one sequence JSONL and update the run-level summary safely."""
    diagnostics_dir = os.path.join(results_dir, "motion_state_diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = motion_diagnostics_file(results_dir, sequence_name)
    with open(jsonl_path, "w") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
            file_handle.write("\n")

    summary_path = os.path.join(diagnostics_dir, "summary.json")
    sequence_summary = summarize_motion_records(records)
    sequence_summary["sequence"] = str(sequence_name)
    try:
        import fcntl
        with open(summary_path, "a+") as summary_handle:
            fcntl.flock(summary_handle.fileno(), fcntl.LOCK_EX)
            summary_handle.seek(0)
            try:
                summary = json.load(summary_handle)
            except (json.JSONDecodeError, ValueError):
                summary = {"sequences": {}}
            summary.setdefault("sequences", {})[str(sequence_name)] = sequence_summary
            summary["sequence_count"] = len(summary["sequences"])
            summary_handle.seek(0)
            summary_handle.truncate()
            json.dump(summary, summary_handle, indent=2, sort_keys=True, allow_nan=False)
            summary_handle.write("\n")
            fcntl.flock(summary_handle.fileno(), fcntl.LOCK_UN)
    except ImportError:
        with open(summary_path, "w") as summary_handle:
            json.dump(
                {"sequence_count": 1, "sequences": {str(sequence_name): sequence_summary}},
                summary_handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            summary_handle.write("\n")
    return jsonl_path, summary_path
