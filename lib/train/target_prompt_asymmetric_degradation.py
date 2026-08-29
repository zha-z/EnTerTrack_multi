"""Train-only E3-D1 exactly-one-view target occlusion."""

import math

import torch


def asymmetric_degradation_enabled(cfg):
    train_cfg = getattr(getattr(
        cfg.TRAIN, "TARGET_PROMPT_COLLABORATION", None),
        "ASYMMETRIC_DEGRADATION", None)
    return bool(getattr(train_cfg, "ENABLED", False))


def _audit(enabled, training, batch_size=0):
    return {
        "enabled": bool(enabled),
        "training": bool(training),
        "applied": False,
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
        "occluded_fraction_mean": 0.0,
        "template_unchanged": True,
        "annotation_unchanged": True,
    }


def _normalized_box_to_pixels(box, height, width):
    values = box.detach().float().reshape(-1)
    if values.numel() < 4 or not bool(torch.isfinite(values[:4]).all().item()):
        raise ValueError("E3-D1 target bbox must contain four finite values")
    x, y, w, h = [float(value.item()) for value in values[:4]]
    if w <= 0.0 or h <= 0.0:
        raise ValueError("E3-D1 target bbox must have positive size")
    raw = (
        math.floor(x * width), math.floor(y * height),
        math.ceil((x + w) * width), math.ceil((y + h) * height))
    clipped = (
        max(0, min(width, raw[0])), max(0, min(height, raw[1])),
        max(0, min(width, raw[2])), max(0, min(height, raw[3])))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("E3-D1 target bbox is empty after clipping")
    return clipped, raw != clipped


def apply_e3_d1_asymmetric_degradation(
        data, cfg, training, generator=None):
    """Return a shallow data copy with only selected search pixels cloned.

    Formal E3-D1 requires an even local triplet batch.  Exactly half of its
    synchronized triplets are sampled without replacement and exactly one
    uniformly sampled view is occluded in each selected triplet.
    """
    enabled = asymmetric_degradation_enabled(cfg)
    images = data.get("search_images", None)
    batch_size = int(images.shape[1]) if torch.is_tensor(images) \
        and images.dim() == 5 else 0
    audit = _audit(enabled, training, batch_size=batch_size)
    if not enabled or not training:
        return data, audit
    if not torch.is_tensor(images) or images.dim() != 5:
        raise ValueError("E3-D1 requires search_images [V,B,C,H,W]")
    if int(images.shape[0]) != 3:
        raise ValueError("E3-D1 requires exactly three canonical views")
    annotations = data.get("search_anno", None)
    if not torch.is_tensor(annotations) or annotations.shape[0] != 3:
        raise ValueError("E3-D1 requires tensor search_anno [V,B,...,4]")
    if int(annotations.shape[1]) != batch_size:
        raise ValueError("E3-D1 search image/annotation batch mismatch")
    if batch_size <= 0 or batch_size % 2 != 0:
        raise ValueError(
            "E3-D1 requires an even positive local batch for exact 50%")

    d1_cfg = cfg.TRAIN.TARGET_PROMPT_COLLABORATION.ASYMMETRIC_DEGRADATION
    ratio = float(d1_cfg.TRIPLET_RATIO)
    weak_views_per_triplet = int(d1_cfg.WEAK_VIEWS_PER_TRIPLET)
    box_scale = float(d1_cfg.OCCLUSION_BOX_SCALE)
    fill_value = float(d1_cfg.FILL_VALUE_NORMALIZED)
    if ratio != 0.5:
        raise ValueError("E3-D1 freezes TRIPLET_RATIO=0.50")
    if weak_views_per_triplet != 1:
        raise ValueError("E3-D1 freezes exactly one weak view per triplet")
    if box_scale != 1.0:
        raise ValueError("E3-D1 freezes OCCLUSION_BOX_SCALE=1.00")
    if fill_value != 0.0:
        raise ValueError("E3-D1 freezes normalized fill value=0.0")

    selected_count = batch_size // 2
    selected = torch.randperm(
        batch_size, device=images.device, generator=generator)[:selected_count]
    weak_views = torch.randint(
        low=0, high=3, size=(selected_count,), device=images.device,
        generator=generator)
    degraded_images = images.clone()
    height, width = int(images.shape[-2]), int(images.shape[-1])
    occluded_pixels = []
    for position in range(selected_count):
        batch_index = int(selected[position].item())
        view_index = int(weak_views[position].item())
        try:
            pixel_box, clipped = _normalized_box_to_pixels(
                annotations[view_index, batch_index], height, width)
        except ValueError:
            audit["invalid_bbox_count"] += 1
            raise
        x0, y0, x1, y1 = pixel_box
        degraded_images[view_index, batch_index, :, y0:y1, x0:x1] = fill_value
        pixels = (x1 - x0) * (y1 - y0)
        occluded_pixels.append(pixels)
        audit["clipped_bbox_count"] += int(clipped)
        audit[("weak_view_A", "weak_view_B", "weak_view_C")[view_index]] += 1

    if sum(audit[name] for name in (
            "weak_view_A", "weak_view_B", "weak_view_C")) != selected_count:
        audit["exactly_one_view_violations"] += 1
        raise RuntimeError("E3-D1 exactly-one-view invariant failed")
    degraded = data.copy()
    degraded["search_images"] = degraded_images
    audit.update({
        "applied": True,
        "selected_triplets": selected_count,
        "selected_ratio": selected_count / float(batch_size),
        "occluded_pixels": int(sum(occluded_pixels)),
        "occluded_fraction_mean": float(
            sum(occluded_pixels) / (selected_count * height * width)),
    })
    return degraded, audit


def degradation_status(audit):
    """Convert the D1 audit to scalar actor logger fields."""
    return {
        "E3D1/enabled": float(audit["enabled"]),
        "E3D1/training": float(audit["training"]),
        "E3D1/applied": float(audit["applied"]),
        "E3D1/triplet_batch_size": float(audit["triplet_batch_size"]),
        "E3D1/selected_triplets": float(audit["selected_triplets"]),
        "E3D1/selected_ratio": float(audit["selected_ratio"]),
        "E3D1/weak_view_A": float(audit["weak_view_A"]),
        "E3D1/weak_view_B": float(audit["weak_view_B"]),
        "E3D1/weak_view_C": float(audit["weak_view_C"]),
        "E3D1/exactly_one_view_violations": float(
            audit["exactly_one_view_violations"]),
        "E3D1/clipped_bbox_count": float(audit["clipped_bbox_count"]),
        "E3D1/invalid_bbox_count": float(audit["invalid_bbox_count"]),
        "E3D1/occluded_pixels": float(audit["occluded_pixels"]),
        "E3D1/occluded_fraction_mean": float(
            audit["occluded_fraction_mean"]),
        "E3D1/template_unchanged": float(audit["template_unchanged"]),
        "E3D1/annotation_unchanged": float(audit["annotation_unchanged"]),
    }
