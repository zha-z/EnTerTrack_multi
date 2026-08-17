"""Strict base-YAML loader for the fixed FCVC training contract."""

import copy
from pathlib import Path

import yaml


def _deep_update(base, update):
    output = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_update(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def load_resolved_config(path, seen=None, validate=True):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError("cyclic BASE_CONFIG reference: {}".format(path))
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base_name = raw.pop("BASE_CONFIG", None)
    base = {}
    if base_name:
        base = load_resolved_config(
            path.parent / base_name, seen=seen, validate=False)
    resolved = _deep_update(base, raw)
    if validate:
        validate_fcvc_config(resolved, require_fcvc=False)
    return resolved


def _required(config, path):
    node = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ValueError("missing explicit config key {}".format(path))
        node = node[key]
    return node


def validate_fcvc_config(config, require_fcvc=True):
    for section in ("MODEL", "TRAIN", "DATA", "TEST"):
        _required(config, section)
    for section in (
        "BACKBONE", "HEAD", "COLLABORATION", "FCVC", "TEACHER", "SAFE_COMMIT"
    ):
        _required(config, "MODEL." + section)
    for section in (
        "SEED", "EPOCH", "BATCH_SIZE", "ACCUMULATION", "OPTIMIZER", "LR",
        "SCHEDULER", "LOSS", "CHECKPOINT",
    ):
        _required(config, "TRAIN." + section)
    for section in ("TRAIN", "TEST", "SAMPLER", "PROCESSING"):
        _required(config, "DATA." + section)
    collaboration_type = _required(config, "MODEL.COLLABORATION.TYPE")
    if collaboration_type not in ("none", "c3r", "fcvc"):
        raise ValueError("MODEL.COLLABORATION.TYPE must be none, c3r, or fcvc")
    if require_fcvc and collaboration_type != "fcvc":
        raise ValueError("FCVC training requires MODEL.COLLABORATION.TYPE=fcvc")
    fixed = {
        "TRAIN.SEED": 42,
        "TRAIN.EPOCH": 30,
        "TRAIN.WORLD_SIZE": 6,
        "TRAIN.MICRO_BATCH_SIZE_PER_GPU": 1,
        "TRAIN.ACCUMULATION": 3,
        "TRAIN.ACCUMULATION_STEPS": 3,
        "TRAIN.GLOBAL_BATCH_SIZE": 18,
        "TRAIN.STEPS_PER_EPOCH": 556,
        "TRAIN.TOTAL_STEPS": 16680,
        "DATA.MAX_SAMPLE_INTERVAL": 200,
        "DATA.TRAIN.SYNC_GROUPS_PER_EPOCH": 3336,
        "DATA.TRAIN.RECEIVER_ASSIGNMENTS_PER_GROUP": 3,
        "DATA.TRAIN.SAMPLE_PER_EPOCH": 10008,
        "DATA.TRAIN.RECEIVER_CASES": 10008,
        "DATA.SAMPLER.STEPS_PER_EPOCH": 556,
        "DATA.SAMPLER.TOTAL_STEPS": 16680,
        "DATA.SPLIT.SEED": 42,
        "DATA.SPLIT.TRAIN_TARGETS": 18,
        "DATA.SPLIT.VAL_TARGETS": 4,
        "DATA.VAL.SYNC_GROUPS": 504,
        "DATA.VAL.RECEIVER_CASES": 1512,
        "DATA.VAL.MANIFEST_SEED": 4242,
        "TRAIN.VAL_EPOCH_INTERVAL": 1,
        "VALIDATION.PAIR_INTERVAL": 1,
        "VALIDATION.ONLINE_INTERVAL": 5,
        "VALIDATION.EPSILON": 1.0e-6,
    }
    for path, expected in fixed.items():
        actual = _required(config, path)
        if actual != expected:
            raise ValueError("{} must be {!r}, got {!r}".format(path, expected, actual))
    expected_losses = {
        "L_cls": 1.0, "L_giou": 2.0, "L_l1": 5.0, "L_align": 0.5,
        "L_recon": 1.0, "L_safe": 0.5, "L_cycle": 0.1,
        "L_teacher_track": 0.25,
    }
    if _required(config, "TRAIN.LOSS") != expected_losses:
        raise ValueError("TRAIN.LOSS differs from the frozen FCVC loss contract")
    if _required(config, "MODEL.COLLABORATION.SAFE_COMMIT") is not True:
        raise ValueError("Safe Commit must remain enabled")
    return config


def legacy_training_contract(config):
    """Map resolved YAML to the frozen legacy runner schema without defaults."""
    validate_fcvc_config(config, require_fcvc=True)
    train = config["TRAIN"]
    return {
        "seed": train["SEED"],
        "seed_contract": {
            "python_random": 42,
            "numpy": 42,
            "torch_manual_seed": 42,
            "torch_cuda_manual_seed": 42,
            "torch_cuda_manual_seed_all": 42,
            "dataloader_generator": 42,
            "worker_init_fn": "base_seed + worker_id",
            "sampler_shuffle": "deterministic_epoch_seed",
            "epoch_seed_rule": "epoch_seed = seed + epoch_index_zero_based",
            "student_teacher_initialization": 42,
            "dropout_and_random_modules": 42,
            "system_time_allowed": False,
        },
        "optimizer": train["OPTIMIZER"]["TYPE"],
        "student_lr": train["LR"]["STUDENT"],
        "teacher_lr": train["LR"]["TEACHER"],
        "student_weight_decay": train["OPTIMIZER"]["WEIGHT_DECAY"],
        "teacher_weight_decay": train["OPTIMIZER"]["WEIGHT_DECAY"],
        "betas": train["OPTIMIZER"]["BETAS"],
        "eps": train["OPTIMIZER"]["EPS"],
        "max_epochs": train["EPOCH"],
        "world_size": train["WORLD_SIZE"],
        "global_batch_size": train["GLOBAL_BATCH_SIZE"],
        "sample_per_epoch": config["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"],
        "steps_per_epoch": train["STEPS_PER_EPOCH"],
        "total_steps": train["TOTAL_STEPS"],
        "logical_batch_size": train["ACCUMULATION_STEPS"],
        "microbatch_size": train["MICRO_BATCH_SIZE_PER_GPU"],
        "gradient_accumulation_steps": train["ACCUMULATION_STEPS"],
        "precision": config["DATA"]["PROCESSING"]["PRECISION"],
        "gradient_clip_max_norm": train["GRADIENT_CLIP_MAX_NORM"],
        "scheduler": {
            "type": train["SCHEDULER"]["TYPE"],
            "warmup_epochs": train["SCHEDULER"]["WARMUP_EPOCHS"],
            "min_lr": train["SCHEDULER"]["MIN_LR"],
        },
        "loss_weights": dict(train["LOSS"]),
        "checkpoint": {
            "save_epoch_checkpoints": train["CHECKPOINT"]["SAVE_EPOCH_CHECKPOINTS"],
            "epoch_pattern": train["CHECKPOINT"]["EPOCH_PATTERN"],
            "interrupt_interval_optimizer_steps": train["CHECKPOINT"]["INTERRUPT_INTERVAL_OPTIMIZER_STEPS"],
            "formal_test_checkpoint": train["CHECKPOINT"]["FORMAL_TEST_CHECKPOINT"],
            "student_export": train["CHECKPOINT"]["STUDENT_EXPORT"],
        },
        "data_contract": {
            "source_manifest": Path(config["DATA"]["TRAIN"]["MANIFEST"]).name,
            "random_sync_sampling": config["DATA"]["TRAIN"]["RANDOM_SYNC_SAMPLING"],
            "multi_view_sync": config["DATA"]["TRAIN"]["MULTI_VIEW_SYNC"],
            "sync_groups_per_epoch": config["DATA"]["TRAIN"]["SYNC_GROUPS_PER_EPOCH"],
            "receiver_assignments_per_group": config["DATA"]["TRAIN"]["RECEIVER_ASSIGNMENTS_PER_GROUP"],
            "sample_per_epoch": config["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"],
            "max_sample_interval": config["DATA"]["MAX_SAMPLE_INTERVAL"],
            "target_balanced": config["DATA"]["TRAIN"]["TARGET_BALANCED"],
            "replacement_sampling": config["DATA"]["SAMPLER"]["REPLACEMENT"],
            "hard_mining": config["DATA"]["SAMPLER"]["HARD_MINING"],
            "old_failure_target_weighting": config["DATA"]["SAMPLER"]["OLD_FAILURE_TARGET_WEIGHTING"],
            "one_visit_per_case_per_epoch": config["DATA"]["SAMPLER"]["ONE_VISIT_PER_CASE_PER_EPOCH"],
            "sync_abc_group_bound": config["DATA"]["SAMPLER"]["SYNC_ABC_GROUP_BOUND"],
        },
        "validation_contract": {
            "split_mode": config["DATA"]["SPLIT"]["MODE"],
            "split_seed": config["DATA"]["SPLIT"]["SEED"],
            "train_targets": config["DATA"]["SPLIT"]["TRAIN_TARGETS"],
            "val_targets": config["DATA"]["SPLIT"]["VAL_TARGETS"],
            "bind_abc_views": config["DATA"]["SPLIT"]["BIND_ABC_VIEWS"],
            "pair_groups": config["DATA"]["VAL"]["SYNC_GROUPS"],
            "pair_cases": config["DATA"]["VAL"]["RECEIVER_CASES"],
            "pair_seed": config["DATA"]["VAL"]["MANIFEST_SEED"],
            "random_augmentation": config["DATA"]["VAL"]["RANDOM_AUGMENTATION"],
            "center_jitter": config["DATA"]["VAL"]["CENTER_JITTER"],
            "scale_jitter": config["DATA"]["VAL"]["SCALE_JITTER"],
            "pair_interval": config["VALIDATION"]["PAIR_INTERVAL"],
            "online_interval": config["VALIDATION"]["ONLINE_INTERVAL"],
            "epsilon": config["VALIDATION"]["EPSILON"],
            "best_metric": config["VALIDATION"]["BEST_METRIC"],
            "tie_breaker": list(config["VALIDATION"]["TIE_BREAKER"]),
        },
    }
