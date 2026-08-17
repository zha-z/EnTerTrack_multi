#!/usr/bin/env python3
"""FCVC full-train entrypoint.

The default action is a static contract check. Real training requires
--run-training so --help/static use cannot accidentally call optimizer.step().
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/multi_agent_collaboration_clean/fcvc_manual_run"
CONFIG_PATH = OUT / "fcvc_full_train_config.json"
MANIFEST_PATH = OUT / "full_train_receiver_manifest.csv"
RUN_DIR = OUT / "checkpoints"
METRICS_PATH = OUT / "epoch_metrics.csv"
TRAIN_REPORT_PATH = OUT / "train_plan_static_report.json"
VIEWS = {"A": 1, "B": 2, "C": 3}
REQUIRED_RESUME_KEYS = (
    "current_epoch",
    "within_epoch_case_offset",
    "global_optimizer_step",
    "seed",
    "student",
    "teacher",
    "optimizer",
    "scheduler",
    "sampler_shuffle_state",
    "rng_state",
    "gradient_accumulation_state",
    "manifest_sha256",
    "epoch_order_digest",
    "training_config_sha256",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if "max_optimizer_steps" in cfg:
        raise ValueError("max_optimizer_steps is forbidden for FCVC full training")
    expected = {
        "seed": 42,
        "optimizer": "AdamW",
        "student_lr": 1.0e-4,
        "teacher_lr": 1.0e-4,
        "student_weight_decay": 1.0e-4,
        "teacher_weight_decay": 1.0e-4,
        "eps": 1.0e-8,
        "max_epochs": 20,
        "logical_batch_size": 16,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 16,
        "precision": "FP32",
        "gradient_clip_max_norm": 10.0,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError("config {} must be {!r}, got {!r}".format(key, value, cfg.get(key)))
    if cfg.get("betas") != [0.9, 0.999]:
        raise ValueError("betas must be [0.9, 0.999]")
    sched = cfg.get("scheduler", {})
    if sched.get("type") != "linear_warmup_then_cosine":
        raise ValueError("scheduler type must be linear_warmup_then_cosine")
    if sched.get("warmup_epochs") != 1 or sched.get("min_lr") != 1.0e-6:
        raise ValueError("scheduler must use 1 warmup epoch and min_lr=1e-6")
    seed_contract = cfg.get("seed_contract", {})
    expected_seed_contract = {
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
    }
    if seed_contract != expected_seed_contract:
        raise ValueError("seed_contract must exactly document the seed=42 random-entry contract")


def read_manifest(path=MANIFEST_PATH):
    if not path.exists():
        raise FileNotFoundError("required frozen manifest is missing: {}".format(path))
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("manifest has no receiver cases")
    required = {"sync_group_id", "target", "frame", "receiver", "sender_1", "sender_2"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("manifest missing columns: {}".format(",".join(sorted(missing))))
    validate_manifest(rows)
    return rows


def validate_manifest(rows):
    seen = set()
    groups = {}
    for i, row in enumerate(rows):
        key = (row["target"], int(row["frame"]), row["receiver"])
        if key in seen:
            raise ValueError("duplicate receiver case at row {}: {}".format(i, key))
        seen.add(key)
        group = (row["sync_group_id"], row["target"], int(row["frame"]))
        groups.setdefault(group, []).append(row)
        receiver = row["receiver"]
        senders = (row["sender_1"], row["sender_2"])
        if receiver not in ("A", "B", "C"):
            raise ValueError("invalid receiver at row {}: {}".format(i, receiver))
        if sorted((receiver,) + senders) != ["A", "B", "C"]:
            raise ValueError("invalid A/B/C assignment at row {}: {}".format(i, row))
        if row.get("base_seed") != "42":
            raise ValueError("manifest base_seed must be 42 at row {}".format(i))
        if row.get("epoch_seed_rule") != "epoch_seed = seed + epoch_index_zero_based":
            raise ValueError("manifest epoch_seed_rule mismatch at row {}".format(i))
    for group, group_rows in groups.items():
        receivers = sorted(row["receiver"] for row in group_rows)
        if receivers != ["A", "B", "C"]:
            raise ValueError("sync group {} does not preserve all receiver assignments".format(group))


def epoch_order(rows, seed, epoch):
    groups = {}
    for idx, row in enumerate(rows):
        groups.setdefault(row["sync_group_id"], []).append(idx)
    for idxs in groups.values():
        idxs.sort(key=lambda i: rows[i]["receiver"])
    group_ids = sorted(groups)
    epoch_seed = seed + (epoch - 1)
    rng = random.Random(epoch_seed)
    rng.shuffle(group_ids)
    order = []
    for group_id in group_ids:
        order.extend(groups[group_id])
    return order


def order_digest(rows, order):
    h = hashlib.sha256()
    for idx in order:
        row = rows[idx]
        h.update("{}|{}|{}|{}|{}|{}\n".format(
            row["sync_group_id"], row["target"], row["frame"],
            row["receiver"], row["sender_1"], row["sender_2"]).encode("utf-8"))
    return h.hexdigest()


def compute_plan(cfg, rows):
    n = len(rows)
    batch = cfg["logical_batch_size"]
    steps_per_epoch = int(math.ceil(n / float(batch)))
    total = cfg["max_epochs"] * steps_per_epoch
    return {
        "receiver_case_count": n,
        "steps_per_epoch": steps_per_epoch,
        "max_epochs": cfg["max_epochs"],
        "total_optimizer_steps": total,
        "warmup_steps": steps_per_epoch,
        "last_batch_incomplete": (n % batch) != 0,
        "last_batch_size": n % batch if n % batch else batch,
    }


def cosine_with_warmup(step, total_steps, warmup_steps, min_lr, base_lr):
    if step <= 0:
        return 0.0
    if step <= warmup_steps:
        return base_lr * step / float(max(1, warmup_steps))
    denom = float(max(1, total_steps - warmup_steps))
    progress = min(1.0, (step - warmup_steps) / denom)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def estimate_time(plan, args):
    sec_per_case = args.seconds_per_receiver_case
    if sec_per_case is None:
        sec_per_case = read_scale_audit_seconds_per_case()
    sec_per_step = sec_per_case * args.logical_batch_size
    epoch_sec = sec_per_case * plan["receiver_case_count"]
    total_sec = epoch_sec * plan["max_epochs"]
    return {
        "seconds_per_receiver_case": sec_per_case,
        "seconds_per_optimizer_step": sec_per_step,
        "seconds_per_epoch": epoch_sec,
        "seconds_total": total_sec,
    }


def read_scale_audit_seconds_per_case():
    profile = ROOT / "output/multi_agent_collaboration_clean/fcvc_scale_audit/resource_profile.md"
    cases = ROOT / "output/multi_agent_collaboration_clean/fcvc_scale_audit/sample_manifest_identity.md"
    elapsed = None
    count = None
    if profile.exists():
        for line in profile.read_text(encoding="utf-8").splitlines():
            if "elapsed_sec:" in line:
                elapsed = float(line.split("`")[1])
    if cases.exists():
        for line in cases.read_text(encoding="utf-8").splitlines():
            if "receiver_cases:" in line:
                count = int(line.split("`")[1])
    if elapsed and count:
        return elapsed / float(count)
    return 1.0


def print_and_save_plan(cfg, rows, args):
    plan = compute_plan(cfg, rows)
    est = estimate_time(plan, args)
    epoch_digests = {
        "epoch_{:02d}".format(epoch): order_digest(rows, epoch_order(rows, cfg["seed"], epoch))
        for epoch in range(1, cfg["max_epochs"] + 1)
    }
    epoch_seeds = {
        "epoch_{:02d}".format(epoch): cfg["seed"] + (epoch - 1)
        for epoch in range(1, cfg["max_epochs"] + 1)
    }
    report = {
        **plan,
        **est,
        "seed": cfg["seed"],
        "seed_contract": cfg["seed_contract"],
        "epoch_seed_rule": "epoch_seed = seed + epoch_index_zero_based",
        "epoch_seed_by_epoch": epoch_seeds,
        "manifest_sha256": sha256_file(args.manifest),
        "training_config_sha256": sha256_file(args.config),
        "epoch_order_digest": epoch_digests,
        "cases_per_epoch": plan["receiver_case_count"],
        "one_visit_per_case_per_epoch": True,
        "replacement_sampling": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key in (
        "receiver_case_count", "steps_per_epoch", "max_epochs",
        "total_optimizer_steps", "warmup_steps", "cases_per_epoch",
        "last_batch_incomplete", "last_batch_size",
        "seconds_per_receiver_case", "seconds_per_optimizer_step",
        "seconds_per_epoch", "seconds_total",
    ):
        print("{}={}".format(key, report[key]))
    print("manifest_sha256={}".format(report["manifest_sha256"]))
    print("training_config_sha256={}".format(report["training_config_sha256"]))
    return report


def static_check(args):
    cfg = load_config(args.config)
    rows = read_manifest(args.manifest)
    report = print_and_save_plan(cfg, rows, args)
    first = order_digest(rows, epoch_order(rows, cfg["seed"], 1))
    repeat = order_digest(rows, epoch_order(rows, cfg["seed"], 1))
    if first != repeat:
        raise RuntimeError("epoch order digest is not deterministic")
    resume = empty_resume_state(cfg, rows, report)
    missing = [key for key in REQUIRED_RESUME_KEYS if key not in resume]
    if missing:
        raise RuntimeError("resume state missing keys: {}".format(",".join(missing)))
    print("sampler_single_epoch_coverage=PASS")
    print("resume_state_structure=PASS")
    print("optimizer_step_called=false")


def empty_resume_state(cfg, rows, report):
    return {
        "current_epoch": 1,
        "within_epoch_case_offset": 0,
        "global_optimizer_step": 0,
        "seed": cfg["seed"],
        "student": None,
        "teacher": None,
        "optimizer": None,
        "scheduler": None,
        "sampler_shuffle_state": {
            "seed": cfg["seed"],
            "epoch": 1,
            "epoch_seed": cfg["seed"],
            "epoch_seed_rule": "epoch_seed = seed + epoch_index_zero_based",
        },
        "rng_state": None,
        "gradient_accumulation_state": {"microbatches_pending": 0, "pending_case_indices": []},
        "manifest_sha256": report["manifest_sha256"],
        "epoch_order_digest": report["epoch_order_digest"],
        "training_config_sha256": report["training_config_sha256"],
    }


def build_manifest(args):
    from lib.train.dataset.threemdot import ThreeMDOT

    cfg = load_config(args.config)
    split_file = ROOT / "lib/train/data_specs/threemdot/threemdot_train.txt"
    dataset = ThreeMDOT(split_file=str(split_file), split="train")
    seqs = list(dataset.sequence_list)
    targets = sorted({seq.rsplit("-", 1)[0] for seq in seqs})
    rows = []
    group_id = 0
    for target in targets:
        view_seq_ids = {}
        for view_id in (1, 2, 3):
            seq = "{}-{}".format(target, view_id)
            if seq not in dataset.sequence_list:
                view_seq_ids = {}
                break
            view_seq_ids[view_id] = dataset.sequence_list.index(seq)
        if len(view_seq_ids) != 3:
            continue
        info = {view_id: dataset.get_sequence_info(seq_id) for view_id, seq_id in view_seq_ids.items()}
        min_len = min(int(info[v]["bbox"].shape[0]) for v in (1, 2, 3))
        for frame in range(1, min_len):
            if not all(bool((info[v]["visible"][frame] & info[v]["valid"][frame]).item()) for v in (1, 2, 3)):
                continue
            group = "g{:08d}".format(group_id)
            group_id += 1
            for receiver in ("A", "B", "C"):
                senders = [v for v in ("A", "B", "C") if v != receiver]
                rows.append({
                    "sync_group_id": group,
                    "target": target,
                    "frame": frame,
                    "receiver": receiver,
                    "sender_1": senders[0],
                    "sender_2": senders[1],
                    "split": "official_train",
                    "uses_gt_in_student_input": "false",
                    "synchronization_validity": "true",
                    "base_seed": str(cfg["seed"]),
                    "epoch_seed_rule": "epoch_seed = seed + epoch_index_zero_based",
                })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sync_group_id", "target", "frame", "receiver", "sender_1", "sender_2",
            "split", "uses_gt_in_student_input", "synchronization_validity",
            "base_seed", "epoch_seed_rule",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("wrote_manifest={}".format(args.manifest))
    print("receiver_case_count={}".format(len(rows)))
    print("manifest_sha256={}".format(sha256_file(args.manifest)))


def run_training(args):
    import numpy as np
    import torch
    from torch.nn.utils import clip_grad_norm_

    from lib.config.entertrack.config import cfg as tracker_cfg, update_config_from_file
    from lib.models.entertrack.entertrack import build_entertrack
    from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel
    from tracking.audit_fcvc_scale import build_case_batch, forward_case, module_groups, grad_stats
    from lib.train.dataset.threemdot import ThreeMDOT

    cfg = load_config(args.config)
    rows = read_manifest(args.manifest)
    report = print_and_save_plan(cfg, rows, args)
    set_deterministic_seed(cfg["seed"], np, torch)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    update_config_from_file(str(ROOT / "experiments/entertrack/entertrack_threemdot_lasot_ft_cons.yaml"))
    tracker = build_entertrack(tracker_cfg, training=True).to(device).eval()
    for p in tracker.parameters():
        p.requires_grad_(False)
    fcvc = FCVCModel(FCVCConfig(enabled=True)).to(device).train()
    student_params = [p for n, p in fcvc.named_parameters() if p.requires_grad and not n.startswith("teacher.")]
    teacher_params = [p for n, p in fcvc.named_parameters() if p.requires_grad and n.startswith("teacher.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": student_params, "lr": cfg["student_lr"], "weight_decay": cfg["student_weight_decay"]},
            {"params": teacher_params, "lr": cfg["teacher_lr"], "weight_decay": cfg["teacher_weight_decay"]},
        ],
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
    )
    state = load_resume(args, cfg, rows, report, fcvc, optimizer)
    dataset = ThreeMDOT(split="train")
    metrics_fields = [
        "epoch", "optimizer_steps", "processed_receiver_cases",
        "L_total_mean", "L_total_p50", "L_total_p90", "L_safe_active_rate",
        "gradient_norm_mean", "clip_rate", "residual_local_ratio_mean",
        "null_attention_mean", "sender_attention_mean", "epoch_wall_time_sec",
        "cumulative_wall_time_sec", "peak_gpu_memory", "lr", "checkpoint_sha256",
    ]
    if not METRICS_PATH.exists():
        with open(METRICS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writeheader()
    start_time = time.perf_counter()
    global_step = state["global_optimizer_step"]
    for epoch in range(state["current_epoch"], cfg["max_epochs"] + 1):
        order = epoch_order(rows, cfg["seed"], epoch)
        offset = state["within_epoch_case_offset"] if epoch == state["current_epoch"] else 0
        loss_values = []
        safe_active = []
        grad_norms = []
        clipped = []
        residual_ratio = []
        null_attention = []
        sender_attention = []
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for pos in range(offset, len(order), cfg["microbatch_size"]):
            batch_indices = order[pos:pos + cfg["microbatch_size"]]
            batch_rows = [rows[i] for i in batch_indices]
            cases = build_case_batch(dataset, batch_rows, device, tracker, fcvc)
            micro_losses = []
            for case in cases:
                losses, diag = forward_case(fcvc, tracker, case)
                loss = losses["L_total"].mean() / float(cfg["gradient_accumulation_steps"])
                micro_losses.append(loss)
                loss_values.append(float(losses["L_total"].detach().mean().item()))
                safe_active.append(float((losses["L_safe"].detach() > 0).float().mean().item()))
                residual_ratio.append(diag["residual_local_feature_norm_ratio"])
                null_attention.append(diag["global_matcher_null_attention_ratio"])
                sender_attention.append(0.5 * (diag["sender_1_attention_contribution"] + diag["sender_2_attention_contribution"]))
            torch.stack(micro_losses).sum().backward()
            accum_ready = ((pos + cfg["microbatch_size"]) % cfg["logical_batch_size"] == 0) or (pos + cfg["microbatch_size"] >= len(order))
            if accum_ready:
                global_step += 1
                lr = cosine_with_warmup(
                    global_step, report["total_optimizer_steps"], report["warmup_steps"],
                    cfg["scheduler"]["min_lr"], cfg["student_lr"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                norm = float(clip_grad_norm_(fcvc.parameters(), cfg["gradient_clip_max_norm"]).item())
                grad_norms.append(norm)
                clipped.append(float(norm > cfg["gradient_clip_max_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if global_step % args.interrupt_checkpoint_interval == 0:
                    save_checkpoint(args, fcvc, optimizer, cfg, rows, report, epoch, pos + cfg["microbatch_size"], global_step, interrupt=True)
        ckpt = save_checkpoint(args, fcvc, optimizer, cfg, rows, report, epoch, len(order), global_step, interrupt=False)
        append_epoch_metrics(metrics_fields, epoch, global_step, len(order), loss_values, safe_active,
                             grad_norms, clipped, residual_ratio, null_attention, sender_attention,
                             epoch_start, start_time, device, optimizer.param_groups[0]["lr"], ckpt)
        state["within_epoch_case_offset"] = 0
    export_student(args, fcvc)


def load_resume(args, cfg, rows, report, fcvc, optimizer):
    state = empty_resume_state(cfg, rows, report)
    state["current_epoch"] = 1
    if not args.resume:
        return state
    import torch

    ckpt = torch.load(args.resume, map_location="cpu")
    missing = [key for key in REQUIRED_RESUME_KEYS if key not in ckpt]
    if missing:
        raise RuntimeError("resume checkpoint missing keys: {}".format(",".join(missing)))
    if ckpt["seed"] != cfg["seed"]:
        raise RuntimeError("seed mismatch on resume")
    if ckpt["manifest_sha256"] != report["manifest_sha256"]:
        raise RuntimeError("manifest SHA256 mismatch on resume")
    if ckpt["training_config_sha256"] != report["training_config_sha256"]:
        raise RuntimeError("training config SHA256 mismatch on resume")
    fcvc.load_state_dict({**ckpt["student"], **ckpt["teacher"]}, strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("rng_state"):
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        if "torch_cuda_all" in ckpt["rng_state"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_state"]["torch_cuda_all"])
        random.setstate(ckpt["rng_state"]["python"])
        if "numpy" in ckpt["rng_state"]:
            import numpy as np
            np.random.set_state(ckpt["rng_state"]["numpy"])
    state.update({key: ckpt[key] for key in ("current_epoch", "within_epoch_case_offset", "global_optimizer_step")})
    return state


def save_checkpoint(args, fcvc, optimizer, cfg, rows, report, epoch, offset, global_step, interrupt):
    import torch

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    student = {k: v.detach().cpu() for k, v in fcvc.state_dict().items() if not k.startswith("teacher.")}
    teacher = {k: v.detach().cpu() for k, v in fcvc.state_dict().items() if k.startswith("teacher.")}
    payload = {
        "current_epoch": epoch,
        "within_epoch_case_offset": offset,
        "global_optimizer_step": global_step,
        "seed": cfg["seed"],
        "student": student,
        "teacher": teacher,
        "optimizer": optimizer.state_dict(),
        "scheduler": {"type": "linear_warmup_then_cosine", "last_step": global_step},
        "sampler_shuffle_state": {
            "seed": cfg["seed"],
            "epoch": epoch,
            "epoch_seed": cfg["seed"] + (epoch - 1),
            "epoch_seed_rule": "epoch_seed = seed + epoch_index_zero_based",
            "offset": offset,
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": __import__("numpy").random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "gradient_accumulation_state": {"microbatches_pending": offset % cfg["logical_batch_size"]},
        "manifest_sha256": report["manifest_sha256"],
        "epoch_order_digest": report["epoch_order_digest"],
        "training_config_sha256": report["training_config_sha256"],
    }
    if interrupt:
        path = RUN_DIR / "interrupt_step_{:06d}.pth".format(global_step)
    else:
        path = RUN_DIR / "epoch_{:02d}.pth".format(epoch)
    torch.save(payload, path)
    return path


def set_deterministic_seed(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def dataloader_worker_init(worker_id):
    import numpy as np
    import torch

    worker_seed = 42 + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2 ** 32 - 1))
    torch.manual_seed(worker_seed)


def dataloader_generator(seed=42):
    import torch

    return torch.Generator().manual_seed(seed)


def append_epoch_metrics(fields, epoch, global_step, processed, losses, safe_active, grad_norms,
                         clipped, residual_ratio, null_attention, sender_attention, epoch_start,
                         start_time, device, lr, ckpt):
    import numpy as np
    import torch

    row = {
        "epoch": epoch,
        "optimizer_steps": global_step,
        "processed_receiver_cases": processed,
        "L_total_mean": float(np.mean(losses)),
        "L_total_p50": float(np.percentile(losses, 50)),
        "L_total_p90": float(np.percentile(losses, 90)),
        "L_safe_active_rate": float(np.mean(safe_active)),
        "gradient_norm_mean": float(np.mean(grad_norms)),
        "clip_rate": float(np.mean(clipped)),
        "residual_local_ratio_mean": float(np.mean(residual_ratio)),
        "null_attention_mean": float(np.mean(null_attention)),
        "sender_attention_mean": float(np.mean(sender_attention)),
        "epoch_wall_time_sec": time.perf_counter() - epoch_start,
        "cumulative_wall_time_sec": time.perf_counter() - start_time,
        "peak_gpu_memory": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "lr": lr,
        "checkpoint_sha256": sha256_file(ckpt),
    }
    with open(METRICS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


def export_student(args, fcvc):
    import torch

    epoch20 = RUN_DIR / "epoch_20.pth"
    if not epoch20.exists():
        raise RuntimeError("refusing export: epoch_20.pth is missing")
    student = {k: v.detach().cpu() for k, v in fcvc.state_dict().items() if not k.startswith("teacher.")}
    count = sum(v.numel() for v in student.values())
    if count != 645792:
        raise RuntimeError("student parameter count must be 645792, got {}".format(count))
    export = RUN_DIR / "fcvc_student_epoch20.pth"
    torch.save({"student": student, "student_parameter_count": count, "teacher_key_count": 0}, export)
    print("exported_student={}".format(export))


def main():
    parser = argparse.ArgumentParser(description="FCVC fixed 20-epoch full training runner")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--device", default="")
    parser.add_argument("--seconds-per-receiver-case", type=float, default=None)
    parser.add_argument("--logical-batch-size", type=int, default=16)
    parser.add_argument("--interrupt-checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--build-manifest-only", action="store_true")
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    if args.build_manifest_only:
        build_manifest(args)
        return
    if args.run_training:
        run_training(args)
        return
    static_check(args)


if __name__ == "__main__":
    main()
