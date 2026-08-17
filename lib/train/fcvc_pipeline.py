"""OSTrack-style six-GPU orchestration for FCVC training."""

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from lib.train.fcvc_config import legacy_training_contract, load_resolved_config


ROOT = Path(__file__).resolve().parents[2]


def _distributed_context(settings):
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size()), dist
    return 0, 1, None


def _write_startup_contract(run_dir, resolved, mapped, sampler_contract,
                            target_split_sha256, pair_manifest_sha256):
    from lib.train.fcvc_checkpoint import config_digest

    payload = {
        "decision": "FCVC_OSTRACK_DDP_READY",
        "formal_training_executed": False,
        "steps_per_epoch": 556,
        "total_optimizer_steps": 16680,
        "warmup_steps": 556,
        "world_size": 6,
        "micro_batch_size_per_gpu": 1,
        "accumulation_steps": 3,
        "global_batch_size": 18,
        "sample_per_epoch": 10008,
        "training_config_sha256": config_digest(mapped),
        "epoch_1_manifest_contract": sampler_contract,
        "validation_target_split_sha256": target_split_sha256,
        "validation_pair_manifest_sha256": pair_manifest_sha256,
    }
    (run_dir / "training_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (run_dir / "train.log").touch(exist_ok=True)
    return payload


def run(settings):
    config = load_resolved_config(settings.cfg_file)
    mapped = legacy_training_contract(config)
    rank, world_size, dist_module = _distributed_context(settings)
    run_dir = Path(settings.save_dir).resolve() / settings.script_name / settings.config_name

    from lib.train.data.fcvc_sampler import FCVCSampler
    from lib.train.admin import env_settings
    from lib.train.fcvc_validation_sampler import ensure_validation_manifest
    from lib.train.fcvc_validation_split import ensure_target_split

    manifest = (ROOT / config["DATA"]["TRAIN"]["MANIFEST"]).resolve()
    validation_dir = (ROOT / config["VALIDATION"]["OUTPUT_DIR"]).resolve()
    split_dir = validation_dir.parent / "splits"
    if rank == 0:
        split_payload, split_sha = ensure_target_split(
            split_dir, manifest, env_settings().threemdot_dir,
            seed=config["DATA"]["SPLIT"]["SEED"],
            train_count=config["DATA"]["SPLIT"]["TRAIN_TARGETS"],
            val_count=config["DATA"]["SPLIT"]["VAL_TARGETS"])
        val_manifest, val_manifest_sha, _ = ensure_validation_manifest(
            validation_dir, manifest, split_payload["validation_targets"])
        validation_contract = {
            "split": split_payload,
            "split_sha256": split_sha,
            "pair_manifest": str(val_manifest),
            "pair_manifest_sha256": val_manifest_sha,
        }
    else:
        validation_contract = None
    if dist_module is not None:
        payload = [validation_contract]
        dist_module.broadcast_object_list(payload, src=0)
        validation_contract = payload[0]
    sampler = FCVCSampler(
        manifest, seed=mapped["seed"],
        sync_groups=config["DATA"]["TRAIN"]["SYNC_GROUPS_PER_EPOCH"],
        max_sample_interval=config["DATA"]["MAX_SAMPLE_INTERVAL"],
        world_size=mapped["world_size"],
        allowed_targets=validation_contract["split"]["train_targets"])
    if rank == 0:
        _, epoch_one_contract = sampler.generate_epoch(1)
        run_dir.mkdir(parents=True, exist_ok=True)
        report = _write_startup_contract(
            run_dir, config, mapped, epoch_one_contract,
            validation_contract["split_sha256"],
            validation_contract["pair_manifest_sha256"])
    else:
        report = None
    if dist_module is not None:
        dist_module.barrier()
        payload = [report]
        dist_module.broadcast_object_list(payload, src=0)
        report = payload[0]

    args = SimpleNamespace(
        manifest=manifest,
        device=("cuda:{}".format(settings.local_rank)
                if getattr(settings, "local_rank", -1) >= 0 else
                getattr(settings, "device_name", "")),
        interrupt_checkpoint_interval=config["TRAIN"]["CHECKPOINT"][
            "INTERRUPT_INTERVAL_OPTIMIZER_STEPS"],
        resume=(Path(settings.resume).resolve()
                if getattr(settings, "resume", None) else None),
    )
    if getattr(settings, "dry_run", False):
        if rank == 0:
            print(json.dumps(report, indent=2, sort_keys=True))
        return report
    if world_size != mapped["world_size"]:
        raise RuntimeError("formal FCVC training requires torchrun world_size=6")
    return _run_training(
        settings, args, config, mapped, report, sampler, rank, world_size,
        dist_module, run_dir, validation_contract, validation_dir)


def _run_training(settings, args, resolved, config, report, sampler, rank,
                  world_size, dist_module, run_dir, validation_contract,
                  validation_dir):
    import random

    import numpy as np
    import torch
    from torch.nn.parallel import DistributedDataParallel

    from lib.config.entertrack.config import cfg as tracker_cfg, update_config_from_file
    from lib.models.entertrack.entertrack import build_entertrack
    from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel
    from lib.train.actors.fcvc_actor import FCVCActor
    from lib.train.data.fcvc_processing import FCVCProcessing
    from lib.train.dataset.threemdot import ThreeMDOT
    from lib.train.admin import env_settings
    from lib.train.fcvc_checkpoint import load_checkpoint
    from lib.train.fcvc_training_graph import FCVCTrainingGraph
    from lib.train.fcvc_pair_validation import PairValidator
    from lib.train.fcvc_online_validation import OnlineValidator
    from lib.train.fcvc_validation_sampler import FixedPairValidationSampler
    from lib.train.trainers import FCVCLTRTrainer
    from tracking import train_fcvc_full as legacy

    # Override the generic rank-offset initialization: FCVC has one frozen base seed.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    legacy.OUT = run_dir
    legacy.RUN_DIR = run_dir / "checkpoints"
    update_config_from_file(str(
        ROOT / "experiments/entertrack/entertrack_threemdot_lasot_ft_cons.yaml"))
    frozen_tracker = build_entertrack(tracker_cfg, training=True).to(device).eval()
    for parameter in frozen_tracker.parameters():
        parameter.requires_grad_(False)
    model_config = {
        key.lower(): value for key, value in resolved["MODEL"]["FCVC"].items()
    }
    fcvc = FCVCModel(FCVCConfig(**model_config)).to(device).train()
    student = [
        parameter for name, parameter in fcvc.named_parameters()
        if parameter.requires_grad and not name.startswith("teacher.")]
    teacher = [
        parameter for name, parameter in fcvc.named_parameters()
        if parameter.requires_grad and name.startswith("teacher.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": student, "lr": config["student_lr"],
             "weight_decay": config["student_weight_decay"]},
            {"params": teacher, "lr": config["teacher_lr"],
             "weight_decay": config["teacher_weight_decay"]},
        ], betas=tuple(config["betas"]), eps=config["eps"])
    resume_state = load_checkpoint(
        args.resume, fcvc, optimizer, config, rank=rank, world_size=world_size)
    model = DistributedDataParallel(
        FCVCTrainingGraph(fcvc), device_ids=[settings.local_rank],
        output_device=settings.local_rank,
        broadcast_buffers=False, find_unused_parameters=False)
    dataset = ThreeMDOT(split="train")
    processing = FCVCProcessing(dataset, device, frozen_tracker, model)
    actor = FCVCActor(model, frozen_tracker)
    pair_sampler = FixedPairValidationSampler(
        validation_contract["pair_manifest"],
        validation_contract["split"]["validation_targets"],
        world_size=world_size)
    pair_validator = PairValidator(
        model, frozen_tracker, processing, pair_sampler, optimizer, sampler,
        validation_dir, device, rank=rank, world_size=world_size,
        dist_module=dist_module, epsilon=resolved["VALIDATION"]["EPSILON"])
    online_validator = OnlineValidator(
        model, frozen_tracker, env_settings().threemdot_dir,
        validation_contract["split"]["validation_targets"], device,
        rank=rank, world_size=world_size, dist_module=dist_module,
        epsilon=resolved["VALIDATION"]["EPSILON"],
        output_dir=validation_dir / "online")
    trainer_report = {
        "steps_per_epoch": report["steps_per_epoch"],
        "total_optimizer_steps": report["total_optimizer_steps"],
        "warmup_steps": report["warmup_steps"],
    }
    return FCVCLTRTrainer(
        legacy, args, config, trainer_report, sampler, processing, actor, model,
        optimizer, device, resume_state, rank=rank, world_size=world_size,
        dist_module=dist_module, pair_validator=pair_validator,
        online_validator=online_validator,
        validation_contract=validation_contract,
        validation_output_dir=validation_dir,
        online_interval=resolved["VALIDATION"]["ONLINE_INTERVAL"]).train()
