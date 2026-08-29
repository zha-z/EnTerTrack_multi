#!/usr/bin/env python3
"""One-batch train/validation data-flow audit for preregistered E3-D1."""

import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.train.admin.settings import Settings  # noqa: E402
from lib.train.base_functions import (  # noqa: E402
    build_dataloaders_threemdot,
    update_settings,
)
from lib.train.data.sampler_threemdot import (  # noqa: E402
    TrackingSamplerThreeMDOT,
)
from lib.train.target_prompt_asymmetric_degradation import (  # noqa: E402
    apply_e3_d1_asymmetric_degradation,
)
from lib.train.train_script import use_grouped_multiview_loader  # noqa: E402


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_audit_config():
    resolved = copy.deepcopy(cfg)
    update_config_from_file(os.path.join(
        ROOT, "experiments", "entertrack",
        "target_prompt_collaboration_e3_d1.yaml"), base_cfg=resolved)
    # One-batch, in-process audit overrides only; the formal YAML is unchanged.
    resolved.TRAIN.NUM_WORKER = 0
    resolved.TRAIN.BATCH_SIZE = 2
    resolved.DATA.TRAIN.SAMPLE_PER_EPOCH = 2
    resolved.DATA.VAL.SAMPLE_PER_EPOCH = 2
    return resolved


def nested_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [nested_list(item) for item in value]
    return value


def metadata(batch):
    return {
        "template_shape": list(batch["template_images"].shape),
        "search_shape": list(batch["search_images"].shape),
        "search_anno_shape": list(batch["search_anno"].shape),
        "target_ids": [str(value) for value in batch["target_id"]],
        "view_ids": [
            [str(value) for value in values] for values in batch["view_ids"]],
        "template_frame_ids": nested_list(batch["template_frame_ids"]),
        "search_frame_ids": nested_list(batch["search_frame_ids"]),
        "template_all_views_visible": bool(torch.as_tensor(
            batch["template_view_valid"]).all().item()),
        "search_all_views_visible": bool(torch.as_tensor(
            batch["search_view_valid"]).all().item()),
    }


def validate_batch(batch):
    if tuple(batch["search_images"].shape[:2]) != (3, 2):
        raise RuntimeError("Expected [V=3,B=2,...] search images")
    expected = (("A", "A"), ("B", "B"), ("C", "C"))
    actual = tuple(tuple(str(value).upper() for value in row)
                   for row in batch["view_ids"])
    if actual != expected:
        raise RuntimeError("Non-canonical ABC batch: {}".format(actual))
    if len(batch["target_id"]) != 2:
        raise RuntimeError("Expected two synchronized target triplets")
    if not bool(torch.as_tensor(batch["template_view_valid"]).all().item()):
        raise RuntimeError("Template violates common-visible contract")
    if not bool(torch.as_tensor(batch["search_view_valid"]).all().item()):
        raise RuntimeError("Search violates common-visible contract")


def run(seed):
    resolved = load_audit_config()
    if not use_grouped_multiview_loader(resolved):
        raise RuntimeError("D1 does not resolve to grouped multiview loader")
    settings = Settings()
    settings.local_rank = -1
    settings.use_lmdb = False
    update_settings(settings, resolved)
    train_loader, val_loader = build_dataloaders_threemdot(resolved, settings)
    if not isinstance(train_loader.dataset, TrackingSamplerThreeMDOT):
        raise RuntimeError("Train loader is not TrackingSamplerThreeMDOT")
    if not isinstance(val_loader.dataset, TrackingSamplerThreeMDOT):
        raise RuntimeError("Validation loader is not TrackingSamplerThreeMDOT")

    reset_seed(seed)
    train_batch = next(iter(train_loader))
    reset_seed(seed + 1)
    val_batch = next(iter(val_loader))
    validate_batch(train_batch)
    validate_batch(val_batch)

    train_source = train_batch["search_images"].clone()
    template_source = train_batch["template_images"]
    annotation_source = train_batch["search_anno"]
    degraded, train_audit = apply_e3_d1_asymmetric_degradation(
        train_batch, resolved, training=True,
        generator=torch.Generator().manual_seed(seed + 2))
    changed = (degraded["search_images"] != train_source).flatten(2).any(2)
    if int(changed.sum().item()) != 1:
        raise RuntimeError("Real batch did not change exactly one view")
    if int(changed.any(0).sum().item()) != 1:
        raise RuntimeError("Real batch did not select exactly one triplet")
    if degraded["template_images"] is not template_source:
        raise RuntimeError("D1 replaced the template tensor")
    if degraded["search_anno"] is not annotation_source:
        raise RuntimeError("D1 replaced search supervision")

    val_effective, val_audit = apply_e3_d1_asymmetric_degradation(
        val_batch, resolved, training=False)
    if val_effective is not val_batch:
        raise RuntimeError("Validation is not an exact data bypass")
    if val_effective["search_images"] is not val_batch["search_images"]:
        raise RuntimeError("Validation search tensor identity changed")

    return {
        "status": "PASS",
        "seed": int(seed),
        "official_test_accessed": False,
        "formal_yaml_mutated": False,
        "loader_class": type(train_loader.dataset).__name__,
        "grouped_multiview": True,
        "common_visible": True,
        "canonical_abc": True,
        "independent_view_sampling": False,
        "train": metadata(train_batch),
        "validation": metadata(val_batch),
        "d1": {
            "selected_triplets": train_audit["selected_triplets"],
            "selected_ratio": train_audit["selected_ratio"],
            "weak_view_counts": [
                train_audit["weak_view_A"], train_audit["weak_view_B"],
                train_audit["weak_view_C"]],
            "changed_view_triplet_indices": changed.nonzero().tolist(),
            "template_identity": degraded["template_images"] is template_source,
            "annotation_identity": degraded["search_anno"] is annotation_source,
            "validation_exact_bypass": val_effective is val_batch,
            "validation_applied": val_audit["applied"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    print(json.dumps(run(args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
