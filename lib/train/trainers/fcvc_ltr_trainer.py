"""FCVC six-rank DDP epoch, accumulation, logging and checkpoint loop."""

from contextlib import nullcontext
from pathlib import Path
import shutil

import torch
from torch.nn.utils import clip_grad_norm_

from lib.train.fcvc_checkpoint import export_student, save_checkpoint
from lib.train.fcvc_logging import (
    FCVCStepLogger,
    StepTimer,
    WeightedMetrics,
    distributed_case_weighted_means,
    extract_case_metrics,
    gradient_norm,
)
from lib.train.fcvc_validation_reporting import (
    ValidationReporter, online_epoch_due,
)


class FCVCLTRTrainer:
    def __init__(self, legacy_module, args, config, report, sampler, processing,
                 actor, model, optimizer, device, resume_state, rank=0,
                 world_size=1, dist_module=None, pair_validator=None,
                 online_validator=None, validation_contract=None,
                 validation_output_dir=None, online_interval=5):
        self.legacy_module = legacy_module
        self.args = args
        self.config = config
        self.report = report
        self.sampler = sampler
        self.processing = processing
        self.actor = actor
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.resume_state = resume_state
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dist = dist_module
        self.pair_validator = pair_validator
        self.online_validator = online_validator
        self.validation_contract = validation_contract or {}
        self.validation_output_dir = validation_output_dir
        self.online_interval = int(online_interval)

    def _checkpoint(self, epoch, offset, global_step, interrupt,
                    validation_metadata=None):
        name = (
            "interrupt_step_{:06d}.pth".format(global_step) if interrupt
            else "epoch_{:02d}.pth".format(epoch)
        )
        path = save_checkpoint(
            Path(self.legacy_module.RUN_DIR) / name, self.model, self.optimizer,
            self.config, self.sampler, epoch, offset, global_step,
            rank=self.rank, world_size=self.world_size,
            dist_module=self.dist,
            validation_metadata=validation_metadata,
        )
        if self.dist is not None:
            self.dist.barrier()
        return path

    def train(self):
        cfg = self.config
        legacy = self.legacy_module
        state = self.resume_state
        logger = None
        if self.rank == 0:
            logger = FCVCStepLogger(
                legacy.OUT, self.report["steps_per_epoch"], print_interval=50)
            if self.pair_validator is not None:
                self.pair_validator.tensorboard = logger.tensorboard
            validation_reporter = ValidationReporter(
                self.validation_output_dir, tensorboard=logger.tensorboard)
        else:
            validation_reporter = None
        timer = StepTimer(self.device)
        global_step = state["global_optimizer_step"]
        resumed_validation = state.get("validation_metadata") or {}
        last_online_metrics = resumed_validation.get("online_metrics")
        for epoch in range(state["current_epoch"], cfg["max_epochs"] + 1):
            contract = self.sampler.begin_epoch(
                epoch, rank=self.rank, world_size=self.world_size,
                dist_module=self.dist)
            checkpoint_manifest = state.get("checkpoint_manifest")
            if epoch == state["current_epoch"] and checkpoint_manifest is not None:
                for key in ("epoch", "manifest_sha256", "order_digest"):
                    if checkpoint_manifest[key] != contract[key]:
                        raise RuntimeError("resume epoch manifest {} mismatch".format(key))
            order = self.sampler.order(epoch)
            offset = (
                state["within_epoch_case_offset"]
                if epoch == state["current_epoch"] else 0)
            if offset % cfg["gradient_accumulation_steps"]:
                raise RuntimeError("resume offset must be at an optimizer boundary")
            if logger is not None:
                logger.begin_epoch(epoch)
            step_metrics = WeightedMetrics()
            step_case_count = 0
            self.optimizer.zero_grad(set_to_none=True)
            for pos in range(offset, len(order), cfg["microbatch_size"]):
                if step_case_count == 0:
                    timer.start()
                data_token = timer.data_start()
                indices = order[pos:pos + cfg["microbatch_size"]]
                cases = self.processing([
                    self.sampler.rows[index] for index in indices])
                timer.data_end(data_token)
                next_case_count = step_case_count + len(cases)
                accumulation_ready = (
                    next_case_count == cfg["gradient_accumulation_steps"])
                sync_context = nullcontext()
                if hasattr(self.model, "no_sync") and not accumulation_ready:
                    sync_context = self.model.no_sync()
                with sync_context:
                    micro_losses = []
                    for case in cases:
                        forward_token = timer.forward_start()
                        losses, stats = self.actor(case)
                        timer.forward_end(forward_token)
                        micro_losses.append(
                            losses["L_total"].mean()
                            / float(cfg["gradient_accumulation_steps"]))
                        step_metrics.add(extract_case_metrics(losses, stats), 1)
                    torch.stack(micro_losses).sum().backward()
                step_case_count = next_case_count
                if not accumulation_ready:
                    continue

                global_step += 1
                lr = legacy.cosine_with_warmup(
                    global_step, self.report["total_optimizer_steps"],
                    self.report["warmup_steps"], cfg["scheduler"]["min_lr"],
                    cfg["student_lr"])
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                norm_before = float(clip_grad_norm_(
                    self.model.parameters(),
                    cfg["gradient_clip_max_norm"]).item())
                norm_after = gradient_norm(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                timing = timer.finish(step_case_count)
                local_values = {
                    **step_metrics.means(), **timing,
                    "lr": lr,
                    "grad_norm_before_clip": norm_before,
                    "grad_norm_after_clip": norm_after,
                    "clip_applied": float(
                        norm_before > cfg["gradient_clip_max_norm"]),
                    "gpu_memory_allocated": float(
                        torch.cuda.memory_allocated(self.device)
                        if self.device.type == "cuda" else 0),
                    "gpu_memory_reserved": float(
                        torch.cuda.memory_reserved(self.device)
                        if self.device.type == "cuda" else 0),
                }
                averaged, global_cases = distributed_case_weighted_means(
                    local_values, step_case_count, self.device,
                    self.dist if self.world_size > 1 else None)
                if logger is not None:
                    step_in_epoch = (pos + 1) // cfg["gradient_accumulation_steps"]
                    row = {
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "processed_cases": global_cases,
                        "fps_interval": 0.0,
                        "fps_epoch": 0.0,
                        **averaged,
                    }
                    logger.record_step(row)
                step_metrics.reset()
                step_case_count = 0
                if global_step % self.args.interrupt_checkpoint_interval == 0:
                    self._checkpoint(
                        epoch, pos + 1, global_step, interrupt=True,
                        validation_metadata={
                            "target_split_sha256": self.validation_contract.get(
                                "split_sha256"),
                            "pair_manifest_sha256": self.validation_contract.get(
                                "pair_manifest_sha256"),
                            "pair_metrics": None,
                            "online_metrics": None,
                            "best_metric": None,
                            "best_epoch": None,
                        })

            if step_case_count != 0:
                raise RuntimeError("epoch ended with a partial accumulation window")
            if logger is not None:
                logger.finish_epoch(global_step)
            pair_metrics, pair_isolation = (None, None)
            if self.pair_validator is not None:
                pair_metrics, pair_isolation = self.pair_validator.run(epoch)
            online_metrics, online_isolation, selected_best = (
                last_online_metrics, None, False)
            if (self.online_validator is not None
                    and online_epoch_due(epoch, self.online_interval)):
                online_metrics, online_isolation = self.online_validator.run(epoch)
                last_online_metrics = online_metrics
                if self.rank == 0:
                    selected_best = validation_reporter.record_online(
                        online_metrics)
                if self.dist is not None:
                    selection = [
                        selected_best,
                        validation_reporter.best if self.rank == 0 else None,
                    ]
                    self.dist.broadcast_object_list(selection, src=0)
                    selected_best, best_state = selection
                else:
                    best_state = validation_reporter.best
            else:
                if self.dist is not None:
                    best_payload = [
                        validation_reporter.best if self.rank == 0 else None]
                    self.dist.broadcast_object_list(best_payload, src=0)
                    best_state = best_payload[0]
                else:
                    best_state = validation_reporter.best
            epoch_path = self._checkpoint(
                epoch, len(order), global_step, interrupt=False,
                validation_metadata={
                    "target_split_sha256": self.validation_contract.get(
                        "split_sha256"),
                    "pair_manifest_sha256": self.validation_contract.get(
                        "pair_manifest_sha256"),
                    "pair_metrics": pair_metrics,
                    "pair_isolation": pair_isolation,
                    "online_metrics": online_metrics,
                    "online_isolation": online_isolation,
                    "best_metric": best_state,
                    "best_epoch": (
                        best_state.get("epoch") if best_state else None),
                })
            if self.rank == 0:
                shutil.copy2(epoch_path, Path(legacy.RUN_DIR) / "latest.pth")
                if selected_best:
                    shutil.copy2(
                        epoch_path, Path(legacy.RUN_DIR) / "best_val_auc.pth")
                validation_reporter.write_summary(
                    self.validation_contract.get("split_sha256"),
                    self.validation_contract.get("pair_manifest_sha256"),
                    pair_metrics=pair_metrics, online_metrics=online_metrics)
            if self.dist is not None:
                self.dist.barrier()
            state["within_epoch_case_offset"] = 0
            state["checkpoint_manifest"] = None
        if self.rank == 0:
            final_checkpoint = Path(legacy.RUN_DIR) / "epoch_30.pth"
            if not final_checkpoint.exists():
                raise RuntimeError("refusing export: epoch_30.pth is missing")
            export_student(
                Path(legacy.RUN_DIR) / "fcvc_student_epoch30.pth", self.model)
            logger.close()
