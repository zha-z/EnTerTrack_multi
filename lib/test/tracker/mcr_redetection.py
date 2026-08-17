"""Prediction-only Motion-guided Periodic Collaborative Re-detection (MCR-v0).

The classes in this module are tracker agnostic.  A tracker adapter only needs
to provide a side-effect-free callback which searches around a requested image
center and returns a :class:`RedetectionCandidate`.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        if value.numel() == 0:
            return None
        value = value.detach().reshape(-1)[0].cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bbox(bbox: Sequence[float]) -> np.ndarray:
    value = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if value.size < 4 or not np.isfinite(value[:4]).all():
        raise ValueError("bbox must contain four finite values")
    return value[:4]


def bbox_center(bbox: Sequence[float]) -> np.ndarray:
    value = _bbox(bbox)
    return value[:2] + 0.5 * value[2:4]


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    first = _bbox(first)
    second = _bbox(second)
    top_left = np.maximum(first[:2], second[:2])
    bottom_right = np.minimum(first[:2] + first[2:4], second[:2] + second[2:4])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = float(size[0] * size[1])
    union = float(first[2] * first[3] + second[2] * second[3] - intersection)
    return intersection / max(union, 1e-12)


def normalized_center_distance(first: Sequence[float], second: Sequence[float]) -> float:
    first = _bbox(first)
    second = _bbox(second)
    scale = max(math.sqrt(float(first[2] * first[3])), 1.0)
    return float(np.linalg.norm(bbox_center(first) - bbox_center(second)) / scale)


@dataclass
class RedetectionCandidate:
    bbox: List[float]
    visual_score: float
    apce: Optional[float] = None
    response_entropy: Optional[float] = None
    anchor_type: str = "unknown"
    scale: float = 1.0
    remote_score: Optional[float] = None
    remote_diagnostics: Optional[Dict[str, Any]] = None
    feature: Any = None
    search_region: Optional[Dict[str, Any]] = None
    motion_consistency: Optional[float] = None
    geometry_consistency: Optional[float] = None
    total_score: Optional[float] = None
    available_weights: Dict[str, float] = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    source: Any = field(default=None, repr=False, compare=False)

    def diagnostic_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "bbox": self.bbox,
            "visual_score": self.visual_score,
            "apce": self.apce,
            "response_entropy": self.response_entropy,
            "anchor_type": self.anchor_type,
            "scale": self.scale,
            "remote_score": self.remote_score,
            "remote_diagnostics": self.remote_diagnostics,
            "search_region": self.search_region,
            "motion_consistency": self.motion_consistency,
            "geometry_consistency": self.geometry_consistency,
            "total_score": self.total_score,
            "available_weights": self.available_weights,
            "rejection_reason": self.rejection_reason,
        })


class EMAMotionPredictor:
    """EMA velocity model using only tracker predictions."""

    def __init__(self, velocity_ema: float = 0.8, max_history: int = 10):
        if not 0.0 <= float(velocity_ema) <= 1.0:
            raise ValueError("velocity_ema must be in [0, 1]")
        self.velocity_ema = float(velocity_ema)
        self.max_history = max(1, int(max_history))
        self.reset()

    def reset(self, bbox: Optional[Sequence[float]] = None) -> None:
        self.centers = deque(maxlen=self.max_history)
        self.sizes = deque(maxlen=self.max_history)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.last_reliable_center = None
        self.last_reliable_bbox = None
        if bbox is not None:
            self.observe(bbox, reliable=True)

    def observe(self, bbox: Sequence[float], reliable: bool = False) -> None:
        value = _bbox(bbox)
        center = bbox_center(value)
        if self.centers:
            displacement = center - self.centers[-1]
            beta = self.velocity_ema
            self.velocity = beta * self.velocity + (1.0 - beta) * displacement
        self.centers.append(center.copy())
        self.sizes.append(value[2:4].copy())
        if reliable:
            self.last_reliable_center = center.copy()
            self.last_reliable_bbox = value.copy()

    @property
    def current_center(self) -> Optional[np.ndarray]:
        return self.centers[-1].copy() if self.centers else None

    @property
    def predicted_center(self) -> Optional[np.ndarray]:
        center = self.current_center
        return None if center is None else center + self.velocity


class PeriodicSearchScheduler:
    def __init__(self, local_interval: int = 10, cooldown_after_switch: int = 5,
                 verify_every_frame_when_pending: bool = True):
        self.local_interval = max(1, int(local_interval))
        self.cooldown_after_switch = max(0, int(cooldown_after_switch))
        self.verify_every_frame_when_pending = bool(verify_every_frame_when_pending)
        self.reset()

    def reset(self) -> None:
        self.cooldown_remaining = 0

    def action(self, frame_id: int, pending: bool = False) -> Optional[str]:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return None
        if pending and self.verify_every_frame_when_pending:
            return "pending_verify"
        if int(frame_id) > 0 and int(frame_id) % self.local_interval == 0:
            return "periodic"
        return None

    def switched(self) -> None:
        self.cooldown_remaining = self.cooldown_after_switch


class MultiAnchorCandidateGenerator:
    def __init__(self, anchors: Iterable[str], scales: Iterable[float],
                 max_regions: int = 6, dedup_normalized_distance: float = 0.25):
        self.anchors = tuple(str(item) for item in anchors)
        self.scales = tuple(float(item) for item in scales)
        self.max_regions = max(1, int(max_regions))
        self.dedup_normalized_distance = max(0.0, float(dedup_normalized_distance))

    @staticmethod
    def _clip_center(center: Sequence[float], image_size: Sequence[int]) -> np.ndarray:
        height, width = int(image_size[0]), int(image_size[1])
        center = np.asarray(center, dtype=np.float64).reshape(2)
        return np.asarray([
            min(max(center[0], 0.0), max(width - 1.0, 0.0)),
            min(max(center[1], 0.0), max(height - 1.0, 0.0)),
        ])

    def generate(self, current_bbox: Sequence[float], motion_center: Optional[Sequence[float]],
                 last_reliable_center: Optional[Sequence[float]],
                 image_size: Sequence[int]) -> List[Dict[str, Any]]:
        current_bbox = _bbox(current_bbox)
        centers = {
            "current": bbox_center(current_bbox),
            "motion": motion_center,
            "last_reliable": last_reliable_center,
        }
        norm = max(math.sqrt(float(current_bbox[2] * current_bbox[3])), 1.0)
        unique = []
        for anchor in self.anchors:
            center = centers.get(anchor)
            if center is None:
                continue
            center = self._clip_center(center, image_size)
            if any(np.linalg.norm(center - item[1]) / norm <= self.dedup_normalized_distance
                   for item in unique):
                continue
            unique.append((anchor, center))

        regions = []
        for scale in self.scales:
            if scale <= 0.0:
                continue
            for anchor, center in unique:
                regions.append({
                    "anchor_type": anchor,
                    "center": center.tolist(),
                    "scale": float(scale),
                })
                if len(regions) >= self.max_regions:
                    return regions
        return regions


class FixedCandidateEvaluator:
    def __init__(self, visual_weight: float = 0.55, remote_weight: float = 0.20,
                 motion_weight: float = 0.15, geometry_weight: float = 0.10,
                 motion_distance_scale: float = 2.0):
        self.weights = {
            "visual": max(0.0, float(visual_weight)),
            "remote": max(0.0, float(remote_weight)),
            "motion": max(0.0, float(motion_weight)),
            "geometry": max(0.0, float(geometry_weight)),
        }
        self.motion_distance_scale = max(float(motion_distance_scale), 1e-6)

    @staticmethod
    def geometry_score(candidate_bbox: Sequence[float], reference_bbox: Sequence[float]) -> float:
        candidate = _bbox(candidate_bbox)
        reference = _bbox(reference_bbox)
        area_ratio = max(float(candidate[2] * candidate[3]), 1e-12) / max(
            float(reference[2] * reference[3]), 1e-12)
        aspect_ratio = max(float(candidate[2] / max(candidate[3], 1e-12)), 1e-12) / max(
            float(reference[2] / max(reference[3], 1e-12)), 1e-12)
        penalty = abs(math.log(area_ratio)) + abs(math.log(aspect_ratio))
        return float(math.exp(-penalty))

    def motion_score(self, candidate_bbox: Sequence[float], predicted_center: Sequence[float],
                     reference_bbox: Sequence[float]) -> float:
        norm = max(math.sqrt(float(_bbox(reference_bbox)[2] * _bbox(reference_bbox)[3])), 1.0)
        distance = float(np.linalg.norm(bbox_center(candidate_bbox) - np.asarray(predicted_center))) / norm
        return float(math.exp(-distance / self.motion_distance_scale))

    def evaluate(self, candidate: RedetectionCandidate, predicted_center: Optional[Sequence[float]],
                 reliable_bbox: Optional[Sequence[float]], reference_bbox: Sequence[float]) -> float:
        components = {"visual": min(max(float(candidate.visual_score), 0.0), 1.0)}
        remote = _finite_float(candidate.remote_score)
        if remote is not None:
            components["remote"] = min(max(remote, 0.0), 1.0)
        if predicted_center is not None:
            components["motion"] = self.motion_score(
                candidate.bbox, predicted_center, reference_bbox)
            candidate.motion_consistency = components["motion"]
        if reliable_bbox is not None:
            components["geometry"] = self.geometry_score(candidate.bbox, reliable_bbox)
            candidate.geometry_consistency = components["geometry"]

        available = {name: self.weights[name] for name in components if self.weights[name] > 0.0}
        denominator = sum(available.values())
        if denominator <= 0.0:
            candidate.total_score = 0.0
            candidate.available_weights = {}
            return 0.0
        normalized = {name: weight / denominator for name, weight in available.items()}
        candidate.available_weights = normalized
        candidate.total_score = float(sum(normalized[name] * components[name] for name in normalized))
        return candidate.total_score


class MultiFrameCandidateVerifier:
    def __init__(self, confirm_frames: int = 2, verify_window: int = 3,
                 min_confirm_iou: float = 0.30, max_confirm_center_distance: float = 1.0,
                 enabled: bool = True):
        self.confirm_frames = max(1, int(confirm_frames))
        self.verify_window = max(self.confirm_frames, int(verify_window))
        self.min_confirm_iou = float(min_confirm_iou)
        self.max_confirm_center_distance = float(max_confirm_center_distance)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.pending = None
        self.confirm_count = 0
        self.age = 0

    def start(self, candidate: RedetectionCandidate) -> bool:
        self.pending = candidate
        self.confirm_count = 1
        self.age = 1
        return not self.enabled or self.confirm_count >= self.confirm_frames

    def verify(self, candidate: RedetectionCandidate) -> Tuple[str, Optional[str]]:
        if self.pending is None:
            return "rejected", "no_pending_candidate"
        self.age += 1
        iou = bbox_iou(self.pending.bbox, candidate.bbox)
        distance = normalized_center_distance(self.pending.bbox, candidate.bbox)
        consistent = iou >= self.min_confirm_iou and distance <= self.max_confirm_center_distance
        if consistent:
            self.confirm_count += 1
            self.pending = candidate
            if self.confirm_count >= self.confirm_frames:
                return "confirmed", None
        if not consistent:
            reason = "confirmation_inconsistent:iou={:.4f},center_distance={:.4f}".format(iou, distance)
            self.reset()
            return "rejected", reason
        if self.age >= self.verify_window:
            self.reset()
            return "rejected", "confirmation_timeout"
        return "pending", None


class SafeBBoxSwitcher:
    def __init__(self, shadow_only: bool = True, freeze_frames: int = 3):
        self.shadow_only = bool(shadow_only)
        self.freeze_frames = max(0, int(freeze_frames))
        self.reset()

    def reset(self) -> None:
        self.update_freeze_remaining = 0
        self.previous_bbox = None

    @property
    def updates_allowed(self) -> bool:
        return self.update_freeze_remaining <= 0

    def begin_frame(self) -> None:
        if self.update_freeze_remaining > 0:
            self.update_freeze_remaining -= 1

    def switch(self, current_bbox: Sequence[float], confirmed_bbox: Sequence[float]) -> List[float]:
        self.previous_bbox = list(_bbox(current_bbox))
        if self.shadow_only:
            return list(_bbox(current_bbox))
        self.update_freeze_remaining = self.freeze_frames
        return list(_bbox(confirmed_bbox))


class MCRRedetectionManager:
    """Coordinate MCR scheduling, scoring, verification, and safe switching."""

    def __init__(self, config: Any):
        get = lambda name, default: getattr(config, name, default)
        self.enabled = bool(get("ENABLED", False))
        self.shadow_only = bool(get("SHADOW_ONLY", True))
        self.motion_enabled = bool(get("MOTION_ENABLED", True))
        self.local_enabled = bool(get("LOCAL_ENABLED", True))
        self.global_enabled = bool(get("GLOBAL_ENABLED", False))
        self.remote_verify_enabled = bool(get("REMOTE_VERIFY_ENABLED", True))
        self.min_visual_score = float(get("MIN_VISUAL_SCORE", 0.30))
        self.min_candidate_score = float(get("MIN_CANDIDATE_SCORE", 0.50))
        self.switch_margin = float(get("SWITCH_MARGIN", 0.05))
        self.max_area_ratio_change = float(get("MAX_AREA_RATIO_CHANGE", 4.0))
        self.max_aspect_ratio_change = float(get("MAX_ASPECT_RATIO_CHANGE", 3.0))
        geometry_guard = get("CURRENT_LARGE_SCALE_GEOMETRY_GUARD", None)
        guard_get = lambda name, default: getattr(geometry_guard, name, default)
        self.current_large_scale_geometry_guard_enabled = bool(
            guard_get("ENABLED", False))
        self.current_large_scale_geometry_guard_min_scale = float(
            guard_get("MIN_SCALE", 2.0))
        self.current_large_scale_geometry_guard_min_geometry = float(
            guard_get("MIN_GEOMETRY", 0.4))
        self.verify_search_scale = float(get("VERIFY_SEARCH_SCALE", 1.5))
        self.reliable_score_threshold = float(get("RELIABLE_SCORE_THRESHOLD", 0.30))
        self.reliable_apce_threshold = float(get("RELIABLE_APCE_THRESHOLD", 0.0))
        self.motion = EMAMotionPredictor(
            get("VELOCITY_EMA", 0.8), get("MAX_HISTORY", 10))
        self.scheduler = PeriodicSearchScheduler(
            get("LOCAL_INTERVAL", 10), get("COOLDOWN_AFTER_SWITCH", 5),
            get("VERIFY_EVERY_FRAME_WHEN_PENDING", True))
        self.generator = MultiAnchorCandidateGenerator(
            get("ANCHORS", ["current", "motion", "last_reliable"]),
            get("LOCAL_SCALES", [1.5, 2.0, 3.0]),
            get("MAX_REGIONS_PER_TRIGGER", 6),
            get("ANCHOR_DEDUP_NORMALIZED_DISTANCE", 0.25))
        self.evaluator = FixedCandidateEvaluator(
            get("VISUAL_WEIGHT", 0.55), get("REMOTE_WEIGHT", 0.20),
            get("MOTION_WEIGHT", 0.15), get("GEOMETRY_WEIGHT", 0.10),
            get("MOTION_DISTANCE_SCALE", 2.0))
        self.verifier = MultiFrameCandidateVerifier(
            get("CONFIRM_FRAMES", 2), get("VERIFY_WINDOW", 3),
            get("MIN_CONFIRM_IOU", 0.30), get("MAX_CONFIRM_CENTER_DISTANCE", 1.0),
            get("MULTIFRAME_CONFIRM_ENABLED", True))
        self.switcher = SafeBBoxSwitcher(
            self.shadow_only, get("UPDATE_FREEZE_FRAMES_AFTER_SWITCH", 3))
        self.reset()

    def reset(self, initial_bbox: Optional[Sequence[float]] = None) -> None:
        self.motion.reset(initial_bbox)
        self.scheduler.reset()
        self.verifier.reset()
        self.switcher.reset()
        self.stats = Counter()
        self.score_sum = 0.0
        self.anchor_selection = Counter()
        self.scale_selection = Counter()

    def _is_reliable(self, visual_score: float, apce: Optional[float]) -> bool:
        apce_value = _finite_float(apce)
        return (
            float(visual_score) >= self.reliable_score_threshold
            and (self.reliable_apce_threshold <= 0.0
                 or (apce_value is not None and apce_value >= self.reliable_apce_threshold))
        )

    def _geometry_valid(self, candidate: Sequence[float], reference: Sequence[float],
                        image_size: Sequence[int]) -> Tuple[bool, Optional[str]]:
        candidate = _bbox(candidate)
        reference = _bbox(reference)
        height, width = image_size[:2]
        if candidate[2] <= 0 or candidate[3] <= 0:
            return False, "invalid_bbox_size"
        if candidate[0] < 0 or candidate[1] < 0 or candidate[0] + candidate[2] > width or candidate[1] + candidate[3] > height:
            return False, "bbox_outside_image"
        area_ratio = float(candidate[2] * candidate[3] / max(reference[2] * reference[3], 1e-12))
        aspect_ratio = float((candidate[2] / candidate[3]) / max(reference[2] / reference[3], 1e-12))
        if max(area_ratio, 1.0 / max(area_ratio, 1e-12)) > self.max_area_ratio_change:
            return False, "area_ratio_change"
        if max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-12)) > self.max_aspect_ratio_change:
            return False, "aspect_ratio_change"
        return True, None

    def _eligible(self, candidate: RedetectionCandidate, reference_score: float,
                  reference_bbox: Sequence[float], image_size: Sequence[int]) -> Tuple[bool, Optional[str]]:
        valid, reason = self._geometry_valid(candidate.bbox, reference_bbox, image_size)
        if not valid:
            return False, reason
        geometry_score = _finite_float(candidate.geometry_consistency)
        if (
            self.current_large_scale_geometry_guard_enabled
            and candidate.anchor_type == "current"
            and float(candidate.scale) >= self.current_large_scale_geometry_guard_min_scale
            and geometry_score is not None
            and geometry_score < self.current_large_scale_geometry_guard_min_geometry
        ):
            return False, "current_large_scale_low_geometry"
        if candidate.visual_score < self.min_visual_score:
            return False, "visual_below_threshold"
        if candidate.total_score is None or candidate.total_score < self.min_candidate_score:
            return False, "candidate_score_below_threshold"
        if candidate.total_score < reference_score + self.switch_margin:
            return False, "switch_margin_not_met"
        return True, None

    def _score(self, candidate: RedetectionCandidate, reference_bbox: Sequence[float]) -> float:
        return self.evaluator.evaluate(
            candidate,
            self.motion.predicted_center if self.motion_enabled else None,
            self.motion.last_reliable_bbox,
            reference_bbox,
        )

    def _candidate_from_callback(self, callback: Callable[..., Any], center: Sequence[float],
                                 scale: float, anchor_type: str,
                                 reference_bbox: Sequence[float]) -> RedetectionCandidate:
        result = callback(
            center=list(center), scale=float(scale), anchor_type=str(anchor_type),
            reference_bbox=list(reference_bbox))
        if isinstance(result, RedetectionCandidate):
            candidate = result
        elif isinstance(result, dict):
            candidate = RedetectionCandidate(**result)
        else:
            raise TypeError("search callback must return RedetectionCandidate or dict")
        candidate.anchor_type = str(anchor_type)
        candidate.scale = float(scale)
        if not self.remote_verify_enabled:
            candidate.remote_score = None
            candidate.remote_diagnostics = None
        self._score(candidate, reference_bbox)
        return candidate

    def process(self, frame_id: int, current_bbox: Sequence[float], current_visual_score: float,
                current_apce: Optional[float], image_size: Sequence[int],
                search_callback: Callable[..., Any], current_remote_score: Optional[float] = None,
                current_remote_diagnostics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process one already-computed main-tracker prediction.

        No GT, visibility, or oracle argument is intentionally present.
        """
        current_bbox = list(_bbox(current_bbox))
        if not self.enabled:
            return {"bbox": current_bbox, "switched": False, "diagnostic": None,
                    "candidate": None, "updates_allowed": True}
        if self.global_enabled:
            raise NotImplementedError("MCR-v0 global tiled search is not implemented")

        self.switcher.begin_frame()
        reliable = self._is_reliable(current_visual_score, current_apce)
        self.motion.observe(current_bbox, reliable=reliable)
        current_reference = RedetectionCandidate(
            bbox=current_bbox, visual_score=float(current_visual_score), apce=_finite_float(current_apce),
            anchor_type="main", scale=1.0,
            remote_score=_finite_float(current_remote_score) if self.remote_verify_enabled else None,
            remote_diagnostics=current_remote_diagnostics if self.remote_verify_enabled else None,
        )
        reference_score = self._score(current_reference, current_bbox)
        action = self.scheduler.action(frame_id, pending=self.verifier.pending is not None)
        generated = []
        reject_reason = None
        switch_event = None
        selected = None
        additional_forwards = 0
        safety_rejections = []

        def record_safety_rejection(candidate, reason):
            candidate.rejection_reason = reason
            if reason != "current_large_scale_low_geometry":
                return
            self.stats["current_large_scale_geometry_reject_count"] += 1
            safety_rejections.append({
                "reason": reason,
                "anchor": candidate.anchor_type,
                "scale": float(candidate.scale),
                "geometry_score": _finite_float(candidate.geometry_consistency),
                "min_geometry": self.current_large_scale_geometry_guard_min_geometry,
                "min_scale": self.current_large_scale_geometry_guard_min_scale,
                "candidate_total_score": _finite_float(candidate.total_score),
                "current_reference_score": _finite_float(reference_score),
                "frame_id": int(frame_id),
            })

        if action == "pending_verify" and self.verifier.pending is not None:
            pending = self.verifier.pending
            candidate = self._candidate_from_callback(
                search_callback, bbox_center(pending.bbox), self.verify_search_scale,
                "pending", pending.bbox)
            generated.append(candidate)
            additional_forwards = 1
            eligible, reject_reason = self._eligible(
                candidate, reference_score, current_bbox, image_size)
            if not eligible and reject_reason:
                record_safety_rejection(candidate, reject_reason)
            if eligible:
                status, reject_reason = self.verifier.verify(candidate)
                if status == "confirmed":
                    selected = candidate
            else:
                self.verifier.reset()
        elif action == "periodic" and self.local_enabled:
            self.stats["periodic_trigger_count"] += 1
            regions = self.generator.generate(
                current_bbox, self.motion.predicted_center,
                self.motion.last_reliable_center, image_size)
            for region in regions:
                generated.append(self._candidate_from_callback(
                    search_callback, region["center"], region["scale"],
                    region["anchor_type"], current_bbox))
            additional_forwards = len(generated)
            eligible_candidates = []
            rejection_reasons = []
            for candidate in generated:
                eligible, reason = self._eligible(
                    candidate, reference_score, current_bbox, image_size)
                if eligible:
                    eligible_candidates.append(candidate)
                elif reason:
                    record_safety_rejection(candidate, reason)
                    rejection_reasons.append(reason)
            if eligible_candidates:
                best = max(eligible_candidates, key=lambda item: item.total_score)
                self.stats["pending_count"] += 1
                if self.verifier.start(best):
                    selected = best
            elif rejection_reasons:
                reject_reason = rejection_reasons[0]

        output_bbox = current_bbox
        switched = False
        if selected is not None:
            output_bbox = self.switcher.switch(current_bbox, selected.bbox)
            self.verifier.reset()
            if self.shadow_only:
                switch_event = "shadow_confirmation_no_switch"
            else:
                switched = True
                switch_event = "confirmed_switch"
                self.scheduler.switched()
                self.motion.reset(output_bbox)
                self.stats["confirmed_switch_count"] += 1
            self.anchor_selection[selected.anchor_type] += 1
            self.scale_selection[str(selected.scale)] += 1
        if reject_reason:
            self.stats["rejected_count"] += 1
        self.stats["candidate_region_count"] += len(generated)
        self.stats["additional_forward_count"] += additional_forwards
        for candidate in generated:
            if candidate.total_score is not None:
                self.score_sum += candidate.total_score
                self.stats["scored_candidate_count"] += 1
            if candidate.remote_diagnostics is None:
                self.stats["missing_remote_diagnostics_count"] += 1

        diagnostic = {
            "frame_id": int(frame_id),
            "enabled": True,
            "shadow_only": self.shadow_only,
            "trigger_reason": action,
            "scheduler_state": "cooldown" if self.scheduler.cooldown_remaining else (
                "pending" if self.verifier.pending is not None else "periodic"),
            "motion_center": _list_or_none(self.motion.predicted_center),
            "current_center": _list_or_none(bbox_center(current_bbox)),
            "last_reliable_center": _list_or_none(self.motion.last_reliable_center),
            "generated_region_count": len(generated),
            "candidates": [item.diagnostic_dict() for item in generated],
            "current_reference_score": reference_score,
            "current_reference_components": current_reference.diagnostic_dict(),
            "selected_candidate": selected.diagnostic_dict() if selected is not None else None,
            "pending_candidate": self.verifier.pending.diagnostic_dict() if self.verifier.pending else None,
            "confirm_count": int(self.verifier.confirm_count),
            "reject_reason": reject_reason,
            "safety_rejections": safety_rejections,
            "switch_event": switch_event,
            "cooldown_remaining": int(self.scheduler.cooldown_remaining),
            "update_freeze_remaining": int(self.switcher.update_freeze_remaining),
            "additional_forward_count": additional_forwards,
            "remote_diagnostics_available": current_remote_diagnostics is not None,
        }
        return {"bbox": output_bbox, "switched": switched, "diagnostic": diagnostic,
                "candidate": selected, "updates_allowed": self.switcher.updates_allowed}

    def summary(self) -> Dict[str, Any]:
        scored = self.stats["scored_candidate_count"]
        return {
            "periodic_trigger_count": int(self.stats["periodic_trigger_count"]),
            "candidate_region_count": int(self.stats["candidate_region_count"]),
            "pending_count": int(self.stats["pending_count"]),
            "rejected_count": int(self.stats["rejected_count"]),
            "confirmed_switch_count": int(self.stats["confirmed_switch_count"]),
            "additional_forward_count": int(self.stats["additional_forward_count"]),
            "average_candidate_score": self.score_sum / scored if scored else None,
            "anchor_selection_distribution": dict(sorted(self.anchor_selection.items())),
            "scale_selection_distribution": dict(sorted(self.scale_selection.items())),
            "missing_remote_diagnostics_count": int(self.stats["missing_remote_diagnostics_count"]),
            "current_large_scale_geometry_reject_count": int(
                self.stats["current_large_scale_geometry_reject_count"]),
        }


def _list_or_none(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def mcr_diagnostics_file(results_dir: str, sequence_name: str) -> str:
    return os.path.join(results_dir, "mcr_diagnostics", "{}.jsonl".format(sequence_name))


def save_mcr_diagnostics(results_dir: str, sequence_name: str,
                         records: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    diagnostics_dir = os.path.join(results_dir, "mcr_diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = mcr_diagnostics_file(results_dir, sequence_name)
    with open(jsonl_path, "w") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False))
            handle.write("\n")
    summary = summarize_mcr_records(records)
    summary["sequence"] = str(sequence_name)
    summary_path = os.path.join(diagnostics_dir, "{}_summary.json".format(sequence_name))
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return jsonl_path, summary_path


def summarize_mcr_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    records = [item for item in records if isinstance(item, dict)]
    parameter_names = sorted({
        str(item["tracker_parameter_name"])
        for item in records
        if item.get("tracker_parameter_name")
    })
    candidates = [candidate for record in records for candidate in record.get("candidates", [])]
    selected = [record.get("selected_candidate") for record in records if record.get("selected_candidate")]
    scores = [_finite_float(item.get("total_score")) for item in candidates]
    scores = [item for item in scores if item is not None]
    return {
        "tracker_parameter_name": parameter_names[0] if len(parameter_names) == 1 else None,
        "frame_count": len(records),
        "periodic_trigger_count": sum(record.get("trigger_reason") == "periodic" for record in records),
        "candidate_region_count": len(candidates),
        "pending_count": sum(record.get("pending_candidate") is not None for record in records),
        "rejected_count": sum(record.get("reject_reason") is not None for record in records),
        "confirmed_switch_count": sum(record.get("switch_event") == "confirmed_switch" for record in records),
        "additional_forward_count": sum(int(record.get("additional_forward_count", 0)) for record in records),
        "average_candidate_score": float(np.mean(scores)) if scores else None,
        "anchor_selection_distribution": dict(Counter(item.get("anchor_type") for item in selected)),
        "scale_selection_distribution": dict(Counter(str(item.get("scale")) for item in selected)),
        "missing_remote_diagnostics_count": sum(
            item.get("remote_diagnostics") is None for item in candidates),
        "current_large_scale_geometry_reject_count": sum(
            candidate.get("rejection_reason") == "current_large_scale_low_geometry"
            for candidate in candidates),
    }
