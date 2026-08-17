#!/usr/bin/env python3
"""Audit and smoke-test the frozen controlled B0 configuration."""

import argparse
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import get_default_config, update_config_from_file
from lib.models.entertrack import build_entertrack


B0_CONFIG = "ostrack_deit_tiny_b0_ep25"
B0_ROLE = "controlled_b0"
B0_BACKBONE = "vit_tiny_patch16_224_half"
B0_BACKBONE_SPEC = {
    "embed_dim": 192,
    "depth": 6,
    "heads": 3,
    "patch_size": 16,
}


def config_path(config_name):
    if os.path.isfile(config_name):
        return os.path.abspath(config_name)
    return os.path.join(ROOT, "experiments", "entertrack", config_name + ".yaml")


def load_config(config_name):
    path = config_path(config_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    resolved = get_default_config()
    update_config_from_file(path, base_cfg=resolved)
    return resolved, path


def _bool(config, path, default=False):
    value = config
    for key in path.split("."):
        value = getattr(value, key, None)
        if value is None:
            return bool(default)
    return bool(value)


def resolved_values(config_name, config):
    backbone = config.MODEL.BACKBONE
    spec = B0_BACKBONE_SPEC if backbone.TYPE == B0_BACKBONE else {
        "embed_dim": "UNKNOWN",
        "depth": "UNKNOWN",
        "heads": "UNKNOWN",
        "patch_size": backbone.STRIDE,
    }
    pruning = (
        _bool(config, "MODEL.BACKBONE.PRUNING_ENABLED")
        or bool(backbone.CE_LOC)
        or str(backbone.TYPE).endswith("_arp")
    )
    compensation = _bool(config, "MODEL.BACKBONE.TOKEN_COMPENSATION_ENABLED")
    dynamic_threshold = _bool(config, "MODEL.BACKBONE.DYNAMIC_THRESHOLD_ENABLED")
    pcum = _bool(config, "MODEL.PCUM.ENABLED")
    mcr = _bool(config, "TEST.MCR.ENABLED")
    remote = (
        _bool(config, "TEST.PCUM.USE_REMOTE")
        or _bool(config, "TEST.COOP.ENABLED")
        or _bool(config, "MODEL.USE_SEARCH_PROMPT")
        or _bool(config, "TEST.USE_SEARCH_PROMPT")
    )
    visible_mask = _bool(config, "TEST.PCUM.USE_REMOTE_VISIBLE_MASK")
    source = str(config.TEST.PCUM.REMOTE_STATE_SOURCE).lower()
    no_gt = not visible_mask and not remote and source == "none"
    return {
        "config_name": config_name,
        "model_role": str(config.MODEL_ROLE),
        "backbone": str(backbone.TYPE),
        "embed_dim": spec["embed_dim"],
        "depth": spec["depth"],
        "heads": spec["heads"],
        "patch_size": spec["patch_size"],
        "template_size": int(config.DATA.TEMPLATE.SIZE),
        "search_size": int(config.DATA.SEARCH.SIZE),
        "test_template_size": int(config.TEST.TEMPLATE_SIZE),
        "test_search_size": int(config.TEST.SEARCH_SIZE),
        "training_datasets": list(config.DATA.TRAIN.DATASETS_NAME),
        "pretrained_checkpoint": str(config.MODEL.PRETRAIN_FILE),
        "total_epochs": int(config.TRAIN.TOTAL_EPOCH),
        "train_epochs": int(config.TRAIN.EPOCH),
        "optimizer": str(config.TRAIN.OPTIMIZER),
        "learning_rate": float(config.TRAIN.LR),
        "scheduler": str(config.TRAIN.SCHEDULER.TYPE),
        "lr_drop_epoch": int(config.TRAIN.LR_DROP_EPOCH),
        "pruning_enabled": pruning,
        "dynamic_threshold_enabled": dynamic_threshold,
        "compensation_enabled": compensation,
        "pcum_enabled": pcum,
        "mcr_enabled": mcr,
        "remote_input_enabled": remote,
        "remote_state_source": source,
        "use_remote_visible_mask": visible_mask,
        "no_gt_inference": no_gt,
    }


def validate_config(values, expect_role=B0_ROLE):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(values["model_role"] == expect_role,
            "model role must be {}".format(expect_role))
    require(values["backbone"] == B0_BACKBONE,
            "backbone must be {}".format(B0_BACKBONE))
    for key, expected in B0_BACKBONE_SPEC.items():
        require(values[key] == expected,
                "{} must be {}".format(key, expected))
    require(values["template_size"] == 128, "DATA template size must be 128")
    require(values["search_size"] == 256, "DATA search size must be 256")
    require(values["test_template_size"] == 128, "TEST template size must be 128")
    require(values["test_search_size"] == 256, "TEST search size must be 256")
    require(values["training_datasets"] == ["THREEMDOT"],
            "training datasets must be [THREEMDOT]")
    require(values["total_epochs"] == 25, "TOTAL_EPOCH must be 25")
    require(values["train_epochs"] == 25, "TRAIN.EPOCH must be 25")
    require(not values["pruning_enabled"], "token pruning must be disabled")
    require(not values["dynamic_threshold_enabled"],
            "dynamic threshold must be disabled")
    require(not values["compensation_enabled"],
            "token compensation must be disabled")
    require(not values["pcum_enabled"], "PCUM must be disabled")
    require(not values["mcr_enabled"], "MCR must be disabled")
    require(not values["remote_input_enabled"], "remote input must be disabled")
    require(values["no_gt_inference"], "no-GT inference audit failed")
    return errors


def checkpoint_path(config):
    value = str(config.MODEL.PRETRAIN_FILE)
    if os.path.isabs(value):
        return value
    return os.path.join(ROOT, "pretrained_models", value)


def run_checkpoint_forward_smoke(config_name=B0_CONFIG):
    config, _ = load_config(config_name)
    values = resolved_values(config_name, config)
    errors = validate_config(values)
    if errors:
        raise RuntimeError("B0 config audit failed before smoke: {}".format(errors))

    source = checkpoint_path(config)
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    model = build_entertrack(config, training=True)
    init_report = getattr(model, "initialization_audit", None)
    if init_report is None or not init_report.get("strict_core_load", False):
        raise RuntimeError("Controlled B0 strict initialization audit is missing")
    if any(".atp." not in key for key in init_report["excluded_source_keys"]):
        raise RuntimeError("Non-ATP source key was excluded")
    if any(".atp." in key for key in model.state_dict()):
        raise RuntimeError("B0 model still contains ATP parameters")
    if model.pcum is not None:
        raise RuntimeError("B0 model unexpectedly contains PCUM")

    model.eval()
    template = torch.randn(1, 3, config.DATA.TEMPLATE.SIZE, config.DATA.TEMPLATE.SIZE)
    search = torch.randn(1, 3, config.DATA.SEARCH.SIZE, config.DATA.SEARCH.SIZE)
    with torch.inference_mode():
        output = model(
            template=template,
            search=search,
            return_atp=False,
            training=False,
        )
    pred_boxes = output["pred_boxes"]
    score_map = output["score_map"]
    if tuple(pred_boxes.shape) != (1, 1, 4):
        raise RuntimeError("Unexpected pred_boxes shape: {}".format(
            tuple(pred_boxes.shape)))
    if tuple(score_map.shape) != (1, 1, 16, 16):
        raise RuntimeError("Unexpected score_map shape: {}".format(
            tuple(score_map.shape)))
    if not torch.isfinite(pred_boxes).all() or not torch.isfinite(score_map).all():
        raise RuntimeError("B0 forward produced NaN or Inf")
    return {
        "checkpoint": source,
        "checkpoint_epoch": init_report["checkpoint_epoch"],
        "loaded_key_count": init_report["loaded_key_count"],
        "excluded_source_keys": init_report["excluded_source_keys"],
        "missing_keys": init_report["missing_keys"],
        "unexpected_keys": init_report["unexpected_keys"],
        "shape_mismatches": init_report["shape_mismatches"],
        "strict_core_load": init_report["strict_core_load"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "pred_boxes_shape": tuple(pred_boxes.shape),
        "score_map_shape": tuple(score_map.shape),
        "finite": True,
    }


def print_values(values):
    ordered = [
        "config_name", "model_role", "backbone", "embed_dim", "depth",
        "heads", "patch_size", "template_size", "search_size",
        "test_template_size", "test_search_size", "training_datasets",
        "pretrained_checkpoint", "total_epochs", "train_epochs", "optimizer",
        "learning_rate", "scheduler", "lr_drop_epoch", "pruning_enabled",
        "dynamic_threshold_enabled", "compensation_enabled", "pcum_enabled",
        "mcr_enabled", "remote_input_enabled", "remote_state_source",
        "use_remote_visible_mask", "no_gt_inference",
    ]
    for key in ordered:
        print("{}={}".format(key, values[key]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=B0_CONFIG)
    parser.add_argument("--expect-role", default=B0_ROLE)
    parser.add_argument("--checkpoint-forward-smoke", action="store_true")
    args = parser.parse_args()

    config, path = load_config(args.config)
    values = resolved_values(args.config, config)
    print("config_path={}".format(path))
    print_values(values)
    errors = validate_config(values, args.expect_role)
    if errors:
        for error in errors:
            print("ERROR: {}".format(error), file=sys.stderr)
        print("audit=FAIL")
        return 1
    print("audit=PASS")

    if args.checkpoint_forward_smoke:
        report = run_checkpoint_forward_smoke(args.config)
        for key, value in report.items():
            print("smoke_{}={}".format(key, value))
        print("checkpoint_forward_smoke=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
