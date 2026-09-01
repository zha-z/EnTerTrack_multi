#!/usr/bin/env python3
"""Representation-only deterministic partial occlusion for D2-P2."""

import hashlib
import math

import torch

from lib.train.target_prompt_asymmetric_degradation import (
    _normalized_box_to_pixels,
)


ORIENTATIONS = ("left", "right", "top", "bottom")
CANDIDATE_COVERAGE = {
    "P25": 0.25,
    "P50": 0.50,
    "P75": 0.75,
    "P100": 1.00,
}
ORIENTATION_NAMESPACE = "D2-P2-orientation-v1"
FILL_VALUE_NORMALIZED = 0.0


def orientation_for_sample(sample_id):
    """Return the frozen orientation derived only from the clean sample id."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    payload = (ORIENTATION_NAMESPACE + "\0" + sample_id).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return ORIENTATIONS[value % len(ORIENTATIONS)]


def partial_pixel_box(normalized_box, height, width, coverage, orientation):
    """Return one contiguous edge-anchored block inside the D1 pixel bbox."""
    coverage = float(coverage)
    if coverage not in CANDIDATE_COVERAGE.values():
        raise ValueError("coverage must be one of 0.25, 0.50, 0.75, 1.00")
    if orientation not in ORIENTATIONS:
        raise ValueError("unknown partial-occlusion orientation")
    (x0, y0, x1, y1), clipped = _normalized_box_to_pixels(
        normalized_box, int(height), int(width))
    if coverage == 1.0:
        return (x0, y0, x1, y1), clipped
    box_width = x1 - x0
    box_height = y1 - y0
    if orientation in ("left", "right"):
        covered = max(1, min(box_width, int(math.ceil(box_width * coverage))))
        if orientation == "left":
            block = (x0, y0, x0 + covered, y1)
        else:
            block = (x1 - covered, y0, x1, y1)
    else:
        covered = max(1, min(box_height, int(math.ceil(box_height * coverage))))
        if orientation == "top":
            block = (x0, y0, x1, y0 + covered)
        else:
            block = (x0, y1 - covered, x1, y1)
    return block, clipped


def apply_partial_occlusion(search, normalized_box, candidate, sample_id):
    """Clone ``search`` and fill only the frozen candidate block with zero."""
    if candidate not in CANDIDATE_COVERAGE:
        raise ValueError("candidate must be P25/P50/P75/P100")
    if not torch.is_tensor(search) or search.dim() != 3:
        raise ValueError("search must have shape [C,H,W]")
    coverage = CANDIDATE_COVERAGE[candidate]
    orientation = orientation_for_sample(sample_id)
    height, width = int(search.shape[-2]), int(search.shape[-1])
    block, clipped = partial_pixel_box(
        normalized_box, height, width, coverage, orientation)
    full_box, _ = _normalized_box_to_pixels(
        normalized_box, height, width)
    x0, y0, x1, y1 = block
    degraded = search.clone()
    degraded[:, y0:y1, x0:x1] = FILL_VALUE_NORMALIZED
    block_pixels = (x1 - x0) * (y1 - y0)
    full_pixels = ((full_box[2] - full_box[0])
                   * (full_box[3] - full_box[1]))
    return degraded, {
        "candidate": candidate,
        "requested_coverage": coverage,
        "orientation": orientation,
        "fill_value_normalized": FILL_VALUE_NORMALIZED,
        "pixel_box": list(block),
        "full_pixel_box": list(full_box),
        "block_pixels": int(block_pixels),
        "full_pixels": int(full_pixels),
        "realized_coverage": block_pixels / float(full_pixels),
        "bbox_clipped": bool(clipped),
    }
