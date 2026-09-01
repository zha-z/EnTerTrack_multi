"""Train-only D2-S1 frozen P50 source degradation.

The selected triplet and weak-view sampling deliberately matches E3-D1.  The
only experimental change is the target-box source: D1 fills P100, while this
module fills the D2-P2 frozen P50 edge-anchored block.  Block orientation is
derived from stable loader metadata and SHA256; RNG is never used for it.
"""

import hashlib
import math

import torch

from lib.train.target_prompt_asymmetric_degradation import (
    _normalized_box_to_pixels,
    asymmetric_degradation_enabled,
)


ORIENTATIONS = ("left", "right", "top", "bottom")
ORIENTATION_NAMESPACE = "D2-P2-orientation-v1"
CANDIDATE = "P50"
COVERAGE = 0.50
FILL_VALUE_NORMALIZED = 0.0
BLOCK_MECHANISM = "single_contiguous_edge_anchored_block"
VIEW_SUFFIX = {"A": "1", "B": "2", "C": "3"}


def source_degradation_enabled(cfg):
    train_cfg = getattr(getattr(
        cfg.TRAIN, "TARGET_PROMPT_COLLABORATION", None),
        "SOURCE_DEGRADATION", None)
    return bool(getattr(train_cfg, "ENABLED", False))


def _metadata_item(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("D2-S1 metadata item must be scalar")
        return value.detach().cpu().item()
    return value


def _nested_item(values, first, second, name):
    try:
        return _metadata_item(values[first][second])
    except (IndexError, KeyError, TypeError):
        raise ValueError("D2-S1 requires collated {} metadata".format(name))


def d2_p1_clean_sample_id(data, view_index, batch_index):
    """Reconstruct the exact D2-P1 clean identity from loader metadata."""
    targets = data.get("target_id", None)
    views = data.get("view_ids", None)
    frame_ids = data.get("search_frame_ids", None)
    if targets is None or views is None or frame_ids is None:
        raise ValueError(
            "D2-S1 requires target_id, view_ids, and search_frame_ids")
    try:
        target_id = str(_metadata_item(targets[batch_index]))
    except (IndexError, TypeError):
        raise ValueError("D2-S1 target_id batch metadata mismatch")
    view_id = str(_nested_item(
        views, view_index, batch_index, "view_ids")).upper()
    if view_id not in VIEW_SUFFIX:
        raise ValueError("D2-S1 requires canonical A/B/C view_ids")
    try:
        frame_slots = len(frame_ids)
    except TypeError:
        raise ValueError("D2-S1 search_frame_ids must be collated by slot")
    if frame_slots == 1:
        frame_id = _nested_item(
            frame_ids, 0, batch_index, "search_frame_ids")
    elif frame_slots == 3:
        frame_id = _nested_item(
            frame_ids, view_index, batch_index, "search_frame_ids")
    else:
        raise ValueError(
            "D2-S1 search_frame_ids requires one shared or three view slots")
    try:
        frame_id = int(frame_id)
    except (TypeError, ValueError):
        raise ValueError("D2-S1 search frame id must be an integer")
    if not target_id or frame_id < 0:
        raise ValueError("D2-S1 sample identity metadata is invalid")
    sequence_name = "{}-{}".format(target_id, VIEW_SUFFIX[view_id])
    return "d2p1-train-{}-{:06d}-clean".format(sequence_name, frame_id)


def orientation_for_sample(sample_id):
    """Frozen D2-P2 SHA256 orientation; no Python hash or RNG."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    payload = (ORIENTATION_NAMESPACE + "\0" + sample_id).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return ORIENTATIONS[value % len(ORIENTATIONS)]


def partial_pixel_box(normalized_box, height, width, coverage, orientation):
    """Frozen D2-P2 single contiguous edge-anchored pixel block."""
    if float(coverage) != COVERAGE:
        raise ValueError("D2-S1 freezes P50 coverage=0.50")
    if orientation not in ORIENTATIONS:
        raise ValueError("D2-S1 orientation must be left/right/top/bottom")
    (x0, y0, x1, y1), clipped = _normalized_box_to_pixels(
        normalized_box, int(height), int(width))
    box_width = x1 - x0
    box_height = y1 - y0
    if orientation in ("left", "right"):
        covered = max(1, min(
            box_width, int(math.ceil(box_width * COVERAGE))))
        block = (x0, y0, x0 + covered, y1) if orientation == "left" \
            else (x1 - covered, y0, x1, y1)
    else:
        covered = max(1, min(
            box_height, int(math.ceil(box_height * COVERAGE))))
        block = (x0, y0, x1, y0 + covered) if orientation == "top" \
            else (x0, y1 - covered, x1, y1)
    return block, clipped


def _audit(enabled, training, batch_size=0):
    return {
        "enabled": bool(enabled),
        "training": bool(training),
        "applied": False,
        "candidate": CANDIDATE,
        "requested_coverage": COVERAGE,
        "triplet_batch_size": int(batch_size),
        "selected_triplets": 0,
        "selected_ratio": 0.0,
        "weak_view_A": 0,
        "weak_view_B": 0,
        "weak_view_C": 0,
        "exactly_one_view_violations": 0,
        "clipped_bbox_count": 0,
        "invalid_bbox_count": 0,
        "occluded_pixels": 0,
        "realized_bbox_coverage_mean": 0.0,
        "orientation_left": 0,
        "orientation_right": 0,
        "orientation_top": 0,
        "orientation_bottom": 0,
        "sample_ids": [],
        "template_unchanged": True,
        "annotation_unchanged": True,
    }


def apply_e3_d2_s1_source_degradation(
        data, cfg, training, generator=None):
    """Clone only search images and apply the frozen P50 source transform."""
    enabled = source_degradation_enabled(cfg)
    images = data.get("search_images", None)
    batch_size = int(images.shape[1]) if torch.is_tensor(images) \
        and images.dim() == 5 else 0
    audit = _audit(enabled, training, batch_size=batch_size)
    if not enabled or not training:
        return data, audit
    if asymmetric_degradation_enabled(cfg):
        raise ValueError("D1 P100 and D2-S1 P50 cannot be enabled together")
    if not torch.is_tensor(images) or images.dim() != 5:
        raise ValueError("D2-S1 requires search_images [V,B,C,H,W]")
    if int(images.shape[0]) != 3:
        raise ValueError("D2-S1 requires exactly three canonical views")
    annotations = data.get("search_anno", None)
    if not torch.is_tensor(annotations) or annotations.shape[0] != 3:
        raise ValueError("D2-S1 requires tensor search_anno [V,B,...,4]")
    if int(annotations.shape[1]) != batch_size:
        raise ValueError("D2-S1 search image/annotation batch mismatch")
    if batch_size <= 0 or batch_size % 2 != 0:
        raise ValueError(
            "D2-S1 requires an even positive local batch for exact 50%")

    d2_cfg = cfg.TRAIN.TARGET_PROMPT_COLLABORATION.SOURCE_DEGRADATION
    frozen = {
        "CANDIDATE": (str(d2_cfg.CANDIDATE), CANDIDATE),
        "COVERAGE": (float(d2_cfg.COVERAGE), COVERAGE),
        "TRIPLET_RATIO": (float(d2_cfg.TRIPLET_RATIO), 0.50),
        "WEAK_VIEWS_PER_TRIPLET": (
            int(d2_cfg.WEAK_VIEWS_PER_TRIPLET), 1),
        "FILL_VALUE_NORMALIZED": (
            float(d2_cfg.FILL_VALUE_NORMALIZED), FILL_VALUE_NORMALIZED),
        "BLOCK_MECHANISM": (
            str(d2_cfg.BLOCK_MECHANISM), BLOCK_MECHANISM),
        "ORIENTATION_NAMESPACE": (
            str(d2_cfg.ORIENTATION_NAMESPACE), ORIENTATION_NAMESPACE),
    }
    for name, (actual, expected) in frozen.items():
        if actual != expected:
            raise ValueError(
                "D2-S1 freezes {}={!r}".format(name, expected))

    selected_count = batch_size // 2
    # These two RNG calls intentionally preserve D1 triplet/view selection.
    # Orientation below is exclusively stable SHA256 over sample identity.
    selected = torch.randperm(
        batch_size, device=images.device, generator=generator)[:selected_count]
    weak_views = torch.randint(
        low=0, high=3, size=(selected_count,), device=images.device,
        generator=generator)
    degraded_images = images.clone()
    height, width = int(images.shape[-2]), int(images.shape[-1])
    occluded_pixels = []
    realized_coverages = []
    for position in range(selected_count):
        batch_index = int(selected[position].item())
        view_index = int(weak_views[position].item())
        sample_id = d2_p1_clean_sample_id(data, view_index, batch_index)
        orientation = orientation_for_sample(sample_id)
        try:
            pixel_box, clipped = partial_pixel_box(
                annotations[view_index, batch_index], height, width,
                COVERAGE, orientation)
            full_box, _ = _normalized_box_to_pixels(
                annotations[view_index, batch_index], height, width)
        except ValueError:
            audit["invalid_bbox_count"] += 1
            raise
        x0, y0, x1, y1 = pixel_box
        degraded_images[view_index, batch_index, :, y0:y1, x0:x1] = \
            FILL_VALUE_NORMALIZED
        pixels = (x1 - x0) * (y1 - y0)
        full_pixels = ((full_box[2] - full_box[0])
                       * (full_box[3] - full_box[1]))
        occluded_pixels.append(pixels)
        realized_coverages.append(pixels / float(full_pixels))
        audit["clipped_bbox_count"] += int(clipped)
        audit[("weak_view_A", "weak_view_B", "weak_view_C")[view_index]] += 1
        audit["orientation_{}".format(orientation)] += 1
        audit["sample_ids"].append(sample_id)

    if sum(audit[name] for name in (
            "weak_view_A", "weak_view_B", "weak_view_C")) != selected_count:
        audit["exactly_one_view_violations"] += 1
        raise RuntimeError("D2-S1 exactly-one-view invariant failed")
    degraded = data.copy()
    degraded["search_images"] = degraded_images
    audit.update({
        "applied": True,
        "selected_triplets": selected_count,
        "selected_ratio": selected_count / float(batch_size),
        "occluded_pixels": int(sum(occluded_pixels)),
        "realized_bbox_coverage_mean": float(
            sum(realized_coverages) / len(realized_coverages)),
        "template_unchanged": degraded.get("template_images") is data.get(
            "template_images"),
        "annotation_unchanged": degraded.get("search_anno") is data.get(
            "search_anno"),
    })
    return degraded, audit


def degradation_status(audit):
    """Convert the D2-S1 audit to scalar actor logger fields."""
    fields = (
        "enabled", "training", "applied", "requested_coverage",
        "triplet_batch_size", "selected_triplets", "selected_ratio",
        "weak_view_A", "weak_view_B", "weak_view_C",
        "exactly_one_view_violations", "clipped_bbox_count",
        "invalid_bbox_count", "occluded_pixels",
        "realized_bbox_coverage_mean", "orientation_left",
        "orientation_right", "orientation_top", "orientation_bottom",
        "template_unchanged", "annotation_unchanged",
    )
    return {"D2S1/{}".format(name): float(audit[name]) for name in fields}
