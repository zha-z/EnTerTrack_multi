import torch


REMOTE_STATE_SOURCES = ("tracker", "none", "gt_legacy")


def validate_remote_state_source(source):
    source = str(source).lower()
    if source not in REMOTE_STATE_SOURCES:
        raise ValueError(
            "Unsupported PCUM remote state source: {}".format(source)
        )
    return source


def uses_gt_visibility(source, use_remote_visible_mask=False):
    source = validate_remote_state_source(source)
    return source == "gt_legacy" or bool(use_remote_visible_mask)


def read_gt_visibility(source, use_remote_visible_mask, sequences, frame_id):
    """Read visibility only for explicit legacy state or oracle filtering."""
    if not uses_gt_visibility(source, use_remote_visible_mask):
        return None

    visibility = []
    for sequence in sequences:
        target_visible = getattr(sequence, "target_visible", None)
        visibility.append(
            True if target_visible is None else bool(target_visible[frame_id])
        )
    return visibility


def _clamp_unit(value):
    if torch.is_tensor(value):
        value = value.detach().item()
    return max(0.0, min(1.0, float(value)))


def build_remote_state(
    scores,
    motion_reliabilities,
    source,
    device,
    use_motion_confidence=False,
    gt_visibility=None,
    apces=None,
    bbox_scores=None,
    valid=None,
    uav_indices=None,
):
    """Build test-time PCUM state from predictions or explicit legacy GT."""
    source = validate_remote_state_source(source)
    if source == "none":
        return None

    scores = [_clamp_unit(score) for score in scores]
    raw_motion = list(motion_reliabilities)
    motion = [0.0 if value is None else _clamp_unit(value) for value in raw_motion]
    if len(scores) == 0 or len(scores) != len(motion):
        raise ValueError("Remote score and motion lists must be non-empty and aligned")

    if valid is None:
        valid = [True] * len(scores)
    if len(valid) != len(scores):
        raise ValueError("Remote valid flags must align with scores")
    valid = [bool(value) for value in valid]

    visible = None
    if source == "gt_legacy":
        if gt_visibility is None or len(gt_visibility) != len(scores):
            raise ValueError("gt_legacy requires aligned GT visibility values")
        visible = [1.0 if bool(value) else 0.0 for value in gt_visibility]
        valid = [valid[i] and bool(visible[i]) for i in range(len(valid))]
        confidence = [scores[i] * visible[i] for i in range(len(scores))]
        motion = [motion[i] * visible[i] for i in range(len(motion))]
    else:
        confidence = list(scores)

    if use_motion_confidence:
        confidence = [
            confidence[i] * (0.5 + 0.5 * motion[i])
            for i in range(len(confidence))
        ]

    def mean_tensor(values):
        return torch.tensor(
            [sum(values) / len(values)],
            device=device,
            dtype=torch.float32,
        )

    state = {
        "score": mean_tensor(scores),
        "confidence": mean_tensor(confidence),
        "motion_reliability": mean_tensor(motion),
        "per_remote_score": torch.tensor(
            [scores], device=device, dtype=torch.float32
        ),
        "per_remote_motion_reliability": torch.tensor(
            [[
                float("nan") if value is None else _clamp_unit(value)
                for value in raw_motion
            ]],
            device=device,
            dtype=torch.float32,
        ),
        "per_remote_valid": torch.tensor(
            [valid], device=device, dtype=torch.bool
        ),
    }

    def optional_metric(values, name):
        if values is None:
            return None
        if len(values) != len(scores):
            raise ValueError("Remote {} values must align with scores".format(name))
        normalized = [
            float("nan") if value is None else _clamp_unit(value)
            for value in values
        ]
        return torch.tensor([normalized], device=device, dtype=torch.float32)

    per_remote_apce = optional_metric(apces, "APCE")
    if per_remote_apce is not None:
        state["per_remote_apce"] = per_remote_apce
    per_remote_bbox_score = optional_metric(bbox_scores, "bbox score")
    if per_remote_bbox_score is not None:
        state["per_remote_bbox_score"] = per_remote_bbox_score
    if uav_indices is not None:
        if len(uav_indices) != len(scores):
            raise ValueError("Remote UAV indices must align with scores")
        state["per_remote_uav_indices"] = torch.tensor(
            [uav_indices], device=device, dtype=torch.long
        )
    if visible is not None:
        state["visible"] = mean_tensor(visible)
    return state
