"""Teacher-free fixed-pair FCVC validation with strict training-state isolation."""

import copy
import csv
import hashlib
import io
import time
from pathlib import Path

import torch

from lib.train.fcvc_checkpoint import capture_rng_state, restore_rng_state
from lib.train.fcvc_logging import distributed_case_weighted_means
from tracking.audit_fcvc_scale import pred_components, prediction_mean_iou


PAIR_FIELDS = (
    "epoch", "cases", "loss_track", "loss_cls", "loss_giou", "loss_l1",
    "iou_local", "iou_collab", "iou_delta", "helpful_rate",
    "harmful_rate", "tied_rate", "safe_active",
    "residual_local_ratio", "null_attention", "sender1_attention",
    "sender2_attention", "fps", "manifest_sha256",
)


def _tensor_digest(value, digest):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8"))
            _tensor_digest(value[key], digest)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _tensor_digest(item, digest)
    else:
        digest.update(repr(value).encode("utf-8"))


def state_digest(value):
    digest = hashlib.sha256()
    _tensor_digest(value, digest)
    return digest.hexdigest()


def _training_module(model):
    module = model.module if hasattr(model, "module") else model
    return module.fcvc if getattr(module, "is_fcvc_training_graph", False) else module


def _case_values(fcvc, tracker, case, epsilon):
    row, local, bundles, gt, _, _, _ = case
    output = fcvc(local, bundles, forward_head=tracker.forward_head)
    local_loss = pred_components(local["local_output"], gt)
    collab_loss = pred_components(output["reported_output"], gt)
    local_iou = prediction_mean_iou(local["local_output"], gt)
    collab_iou = prediction_mean_iou(output["reported_output"], gt)
    delta = collab_iou - local_iou
    residual = output["high_writer"]["residual"].detach()
    local_feature = local["high_search"].detach()
    return {
        "loss_track": float(collab_loss["L_track"].mean().item()),
        "loss_cls": float(collab_loss["L_cls"].mean().item()),
        "loss_giou": float(collab_loss["L_giou"].mean().item()),
        "loss_l1": float(collab_loss["L_l1"].mean().item()),
        "iou_local": local_iou,
        "iou_collab": collab_iou,
        "iou_delta": delta,
        "helpful_rate": float(delta > epsilon),
        "harmful_rate": float(delta < -epsilon),
        "tied_rate": float(abs(delta) <= epsilon),
        "safe_active": float(
            collab_loss["L_track"].mean().item()
            > local_loss["L_track"].mean().item()),
        "residual_local_ratio": float(
            residual.norm().item() / local_feature.norm().clamp_min(1e-6).item()),
        "null_attention": float(output["global_match"][
            "null_attention_ratio"].detach().mean().item()),
        "sender1_attention": float(output["global_match"][
            "sender_contribution"][..., 0].detach().mean().item()),
        "sender2_attention": float(output["global_match"][
            "sender_contribution"][..., 1].detach().mean().item()),
        "uses_gt_in_student_input": str(row.get(
            "uses_gt_in_student_input", "false")).strip().lower()
            not in ("false", "0", "none", ""),
    }


def _append_unique_csv(path, fields, row, key="epoch"):
    path = Path(path)
    existing = []
    has_header = path.exists() and path.stat().st_size > 0
    if has_header:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise ValueError("unexpected validation CSV schema")
            existing = list(reader)
        if any(int(item[key]) == int(row[key]) for item in existing):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not has_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in fields})
    return True


class PairValidator:
    def __init__(self, model, frozen_tracker, processing, sampler, optimizer,
                 training_sampler, output_dir, device, rank=0, world_size=6,
                 dist_module=None, epsilon=1e-6, tensorboard=None):
        self.model = model
        self.frozen_tracker = frozen_tracker
        self.processing = processing
        self.sampler = sampler
        self.optimizer = optimizer
        self.training_sampler = training_sampler
        self.output_dir = Path(output_dir)
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dist = dist_module
        self.epsilon = float(epsilon)
        self.tensorboard = tensorboard

    def run(self, epoch, max_local_batches=None, write_outputs=True):
        if self.world_size != 6:
            raise ValueError("pair validation requires six ranks")
        rows = self.sampler.partition(self.rank, self.world_size)
        if max_local_batches is not None:
            rows = rows[:int(max_local_batches) * 3]
        if len(rows) % 3:
            raise RuntimeError("validation partition has a partial batch")
        was_training = self.model.training
        rng_before = capture_rng_state()
        params_before = state_digest(_training_module(self.model).state_dict())
        optimizer_before = state_digest(self.optimizer.state_dict())
        sampler_before = copy.deepcopy({
            "contract": self.training_sampler.current_contract,
            "rows": self.training_sampler.rows,
        })
        sums = {}
        local_cases = 0
        start = time.perf_counter()
        self.model.eval()
        self.frozen_tracker.eval()
        try:
            with torch.inference_mode():
                for start_index in range(0, len(rows), 3):
                    cases = self.processing(rows[start_index:start_index + 3])
                    step_sums = {}
                    for case in cases:
                        values = _case_values(
                            _training_module(self.model), self.frozen_tracker,
                            case, self.epsilon)
                        if values.pop("uses_gt_in_student_input"):
                            raise RuntimeError("validation student input contains GT")
                        for name, value in values.items():
                            sums[name] = sums.get(name, 0.0) + value
                            step_sums[name] = step_sums.get(name, 0.0) + value
                        local_cases += 1
                    step_means = {
                        name: value / len(cases) for name, value in step_sums.items()}
                    global_step, global_count = distributed_case_weighted_means(
                        step_means, len(cases), self.device,
                        self.dist if self.world_size > 1 else None)
                    step_number = start_index // 3 + 1
                    total_steps = len(rows) // 3
                    if self.rank == 0 and (
                            step_number == total_steps
                            or (step_number == 50 and total_steps >= 50)):
                        elapsed = max(time.perf_counter() - start, 1e-12)
                        print(
                            "[val: {epoch}, {step} / {total_steps}] FPS: {fps:.1f}  ,  "
                            "Loss/track: {loss_track:.5f}  ,  Loss/giou: {loss_giou:.5f}  ,  "
                            "Loss/l1: {loss_l1:.5f}  ,  IoU/local: {iou_local:.5f}  ,  "
                            "IoU/collab: {iou_collab:.5f}  ,  IoU/delta: {iou_delta:.5f}  ,  "
                            "helpful: {helpful_rate:.5f}  ,  harmful: {harmful_rate:.5f}  ,  "
                            "safe_active: {safe_active:.5f}  ,  res_ratio: {residual_local_ratio:.5f}  ,  "
                            "null_attn: {null_attention:.5f}".format(
                                epoch=epoch, step=step_number,
                                total_steps=total_steps,
                                fps=(step_number * global_count) / elapsed,
                                **global_step))
        finally:
            restore_rng_state(rng_before)
            self.model.train(was_training)

        local_means = {name: value / local_cases for name, value in sums.items()}
        metrics, global_cases = distributed_case_weighted_means(
            local_means, local_cases, self.device,
            self.dist if self.world_size > 1 else None)
        metrics["iou_delta"] = metrics["iou_collab"] - metrics["iou_local"]
        metrics.update({
            "epoch": int(epoch),
            "cases": global_cases,
            "fps": global_cases / max(time.perf_counter() - start, 1e-12),
            "manifest_sha256": self.sampler.sha256,
        })
        isolation = {
            "parameters_unchanged": params_before == state_digest(
                _training_module(self.model).state_dict()),
            "optimizer_unchanged": optimizer_before == state_digest(
                self.optimizer.state_dict()),
            "rng_restored": state_digest(rng_before) == state_digest(capture_rng_state()),
            "training_sampler_unchanged": sampler_before == {
                "contract": self.training_sampler.current_contract,
                "rows": self.training_sampler.rows,
            },
            "teacher_called": False,
            "gt_roi_used": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
        }
        if not all(value is True or value == 0 for value in isolation.values()):
            raise RuntimeError("pair validation state isolation failed: {}".format(isolation))
        if self.dist is not None:
            self.dist.barrier()
        if self.rank == 0 and write_outputs:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            _append_unique_csv(
                self.output_dir / "pair_metrics.csv", PAIR_FIELDS, metrics)
            if self.tensorboard is not None:
                tags = {
                    "Loss_track": "loss_track", "IoU_local": "iou_local",
                    "IoU_collab": "iou_collab", "IoU_delta": "iou_delta",
                    "helpful_rate": "helpful_rate", "harmful_rate": "harmful_rate",
                    "safe_active": "safe_active",
                    "residual_local_ratio": "residual_local_ratio",
                }
                for tag, field in tags.items():
                    self.tensorboard.add_scalar(
                        "ValidationPair/" + tag, metrics[field], epoch)
            print(
                "Validation epoch {epoch:02d}\n- cases: {cases}\n"
                "- IoU/local: {iou_local:.6f}\n- IoU/collab: {iou_collab:.6f}\n"
                "- IoU/delta: {iou_delta:.6f}\n"
                "- helpful/harmful/tied: {helpful_rate:.6f}/{harmful_rate:.6f}/{tied_rate:.6f}\n"
                "- Loss/track: {loss_track:.6f}\n- safe_active: {safe_active:.6f}\n"
                "- residual/local ratio: {residual_local_ratio:.6f}".format(**metrics))
        return metrics, isolation
