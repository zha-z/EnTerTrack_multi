"""Optimizer-step logging for FCVC training.

This module is deliberately independent from model, data, and optimizer
construction so its formatting and resume behavior can be tested without
running training.
"""

import csv
import math
import sys
import time
from pathlib import Path

import torch


LOSS_WEIGHTS = {
    "align": 0.50,
    "recon": 1.00,
    "safe": 0.50,
    "cycle": 0.10,
    "teacher": 0.25,
}

CASE_METRIC_FIELDS = (
    "loss_total", "loss_track", "loss_cls", "loss_giou", "loss_l1",
    "loss_align_raw", "loss_align_weighted",
    "loss_recon_raw", "loss_recon_weighted",
    "loss_safe_raw", "loss_safe_weighted",
    "loss_cycle_raw", "loss_cycle_weighted",
    "loss_teacher_raw", "loss_teacher_weighted",
    "iou", "safe_active_rate", "residual_local_ratio", "null_attention",
    "sender1_attention", "sender2_attention",
)

STEP_FIELDS = (
    "epoch", "step_in_epoch", "global_step", "processed_cases", "lr",
    "fps_interval", "fps_epoch", "data_time", "forward_time", "total_time",
) + CASE_METRIC_FIELDS + (
    "grad_norm_before_clip", "grad_norm_after_clip", "clip_applied",
    "gpu_memory_allocated", "gpu_memory_reserved",
)

EPOCH_FIELDS = (
    "epoch", "global_step", "processed_cases", "lr", "fps", "data_time",
    "forward_time", "total_time",
) + CASE_METRIC_FIELDS + (
    "grad_norm_before_clip", "grad_norm_after_clip", "clip_rate",
    "gpu_memory_allocated", "gpu_memory_reserved",
)

_STEP_AVERAGE_FIELDS = tuple(
    field for field in STEP_FIELDS
    if field not in {
        "epoch", "step_in_epoch", "global_step", "processed_cases",
        "fps_interval", "fps_epoch",
    }
)


def finite_float(value, name):
    if torch.is_tensor(value):
        value = value.detach().float().mean().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be a finite Python float".format(name))
    return value


def logical_batch_case_count(position, total_cases, logical_batch_size):
    batch_start = (int(position) // int(logical_batch_size)) * int(logical_batch_size)
    return min(int(logical_batch_size), int(total_cases) - batch_start)


def extract_case_metrics(losses, diagnostics):
    raw = {
        "loss_total": finite_float(losses["L_total"], "L_total"),
        "loss_track": finite_float(losses["L_track"], "L_track"),
        "loss_cls": finite_float(losses["L_cls"], "L_cls"),
        "loss_giou": finite_float(losses["L_giou"], "L_giou"),
        "loss_l1": finite_float(losses["L_l1"], "L_l1"),
        "loss_align_raw": finite_float(losses["L_align"], "L_align"),
        "loss_recon_raw": finite_float(losses["L_recon"], "L_recon"),
        "loss_safe_raw": finite_float(losses["L_safe"], "L_safe"),
        "loss_cycle_raw": finite_float(losses["L_cycle"], "L_cycle"),
        "loss_teacher_raw": finite_float(
            losses["L_teacher_track"], "L_teacher_track"),
        "iou": finite_float(diagnostics["collaborative_mean_iou"], "IoU"),
        "safe_active_rate": finite_float(
            (losses["L_safe"].detach() > 0).float().mean(), "safe_active_rate"),
        "residual_local_ratio": finite_float(
            diagnostics["residual_local_feature_norm_ratio"], "residual_local_ratio"),
        "null_attention": finite_float(
            diagnostics["global_matcher_null_attention_ratio"], "null_attention"),
        "sender1_attention": finite_float(
            diagnostics["sender_1_attention_contribution"], "sender1_attention"),
        "sender2_attention": finite_float(
            diagnostics["sender_2_attention_contribution"], "sender2_attention"),
    }
    raw.update({
        "loss_align_weighted": LOSS_WEIGHTS["align"] * raw["loss_align_raw"],
        "loss_recon_weighted": LOSS_WEIGHTS["recon"] * raw["loss_recon_raw"],
        "loss_safe_weighted": LOSS_WEIGHTS["safe"] * raw["loss_safe_raw"],
        "loss_cycle_weighted": LOSS_WEIGHTS["cycle"] * raw["loss_cycle_raw"],
        "loss_teacher_weighted": LOSS_WEIGHTS["teacher"] * raw["loss_teacher_raw"],
    })
    component_sum = sum((
        raw["loss_track"], raw["loss_align_weighted"],
        raw["loss_recon_weighted"], raw["loss_safe_weighted"],
        raw["loss_cycle_weighted"], raw["loss_teacher_weighted"],
    ))
    if not math.isclose(raw["loss_total"], component_sum, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            "Loss/total component mismatch: {:.9g} != {:.9g}".format(
                raw["loss_total"], component_sum))
    return {field: finite_float(raw[field], field) for field in CASE_METRIC_FIELDS}


def gradient_norm(parameters):
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().norm(2).square())
    if not squares:
        return 0.0
    return finite_float(torch.stack(squares).sum().sqrt(), "gradient_norm")


def distributed_case_weighted_means(values, local_cases, device, dist_module=None):
    """All-reduce metric sums and case count; return identical global means."""
    names = tuple(sorted(values))
    vector = torch.tensor(
        [finite_float(values[name], name) * int(local_cases) for name in names]
        + [float(local_cases)], device=device, dtype=torch.float64)
    if dist_module is not None:
        dist_module.all_reduce(vector)
    count = float(vector[-1].item())
    if count <= 0:
        raise ValueError("global metric case count must be positive")
    return {
        name: float(vector[index].item() / count)
        for index, name in enumerate(names)
    }, int(count)


class WeightedMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cases = 0
        self.sums = {}
        self.elapsed_total = 0.0

    def add(self, values, n):
        n = int(n)
        if n <= 0:
            raise ValueError("metric weight must be positive")
        self.cases += n
        for name, value in values.items():
            if name in ("fps_interval", "fps_epoch"):
                continue
            number = finite_float(value, name)
            self.sums[name] = self.sums.get(name, 0.0) + number * n
        if "total_time" in values:
            self.elapsed_total += finite_float(values["total_time"], "total_time") * n

    def means(self):
        if self.cases <= 0:
            raise ValueError("cannot average an empty metric window")
        return {name: total / self.cases for name, total in self.sums.items()}

    def fps(self):
        if self.elapsed_total <= 0.0:
            return 0.0
        return self.cases / self.elapsed_total


def should_print_step(step_in_epoch, steps_per_epoch, print_interval=50):
    step = int(step_in_epoch)
    return step == 1 or step == int(steps_per_epoch) or step % int(print_interval) == 0


def format_console_line(epoch, step_in_epoch, steps_per_epoch, interval, epoch_stats):
    return (
        "[train: {epoch}, {step} / {total}] "
        "FPS: {fps_i:.1f} ({fps_e:.1f})  ,  "
        "DataTime: {data_i:.3f} ({data_e:.3f})  ,  "
        "ForwardTime: {forward:.3f}  ,  TotalTime: {total_time:.3f}  ,  "
        "LR: {lr:.2e}  ,  Loss/total: {loss_total:.5f}  ,  "
        "Loss/track: {loss_track:.5f}  ,  Loss/giou: {loss_giou:.5f}  ,  "
        "Loss/l1: {loss_l1:.5f}  ,  Loss/align_w: {loss_align_weighted:.5f}  ,  "
        "Loss/recon_w: {loss_recon_weighted:.5f}  ,  "
        "Loss/safe_w: {loss_safe_weighted:.5f}  ,  "
        "Loss/cycle_w: {loss_cycle_weighted:.5f}  ,  "
        "Loss/teacher_w: {loss_teacher_weighted:.5f}  ,  IoU: {iou:.5f}  ,  "
        "safe_active: {safe_active_rate:.5f}  ,  "
        "res_ratio: {residual_local_ratio:.5f}  ,  "
        "null_attn: {null_attention:.5f}"
    ).format(
        epoch=int(epoch), step=int(step_in_epoch), total=int(steps_per_epoch),
        fps_i=interval["fps"], fps_e=epoch_stats["fps"],
        data_i=interval["data_time"], data_e=epoch_stats["data_time"],
        forward=interval["forward_time"], total_time=interval["total_time"],
        lr=interval["lr"], loss_total=interval["loss_total"],
        loss_track=interval["loss_track"], loss_giou=interval["loss_giou"],
        loss_l1=interval["loss_l1"],
        loss_align_weighted=interval["loss_align_weighted"],
        loss_recon_weighted=interval["loss_recon_weighted"],
        loss_safe_weighted=interval["loss_safe_weighted"],
        loss_cycle_weighted=interval["loss_cycle_weighted"],
        loss_teacher_weighted=interval["loss_teacher_weighted"],
        iou=interval["iou"], safe_active_rate=interval["safe_active_rate"],
        residual_local_ratio=interval["residual_local_ratio"],
        null_attention=interval["null_attention"],
    )


class StepTimer:
    """CUDA-event segment timing with synchronization only at step boundaries."""

    def __init__(self, device):
        self.device = device
        self.is_cuda = getattr(device, "type", str(device)) == "cuda"
        self.active = False

    def _sync(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self.group_start = time.perf_counter()
        self.data_time = 0.0
        self.forward_time = 0.0
        self.data_events = []
        self.forward_events = []
        self.active = True

    def _segment_start(self):
        if self.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def _segment_end(self, token, events, attribute):
        if self.is_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events.append((token, end))
        else:
            setattr(self, attribute, getattr(self, attribute) + time.perf_counter() - token)

    def data_start(self):
        return self._segment_start()

    def data_end(self, token):
        self._segment_end(token, self.data_events, "data_time")

    def forward_start(self):
        return self._segment_start()

    def forward_end(self, token):
        self._segment_end(token, self.forward_events, "forward_time")

    def finish(self, processed_cases):
        self._sync()
        if self.is_cuda:
            self.data_time = sum(
                start.elapsed_time(end) for start, end in self.data_events) / 1000.0
            self.forward_time = sum(
                start.elapsed_time(end) for start, end in self.forward_events) / 1000.0
        total = time.perf_counter() - self.group_start
        cases = int(processed_cases)
        self.active = False
        return {
            "data_time": self.data_time / cases,
            "forward_time": self.forward_time / cases,
            "total_time": total / cases,
        }


class MetricsCSVWriter:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.step_path = self.run_dir / "step_metrics.csv"
        self.epoch_path = self.run_dir / "epoch_metrics.csv"
        self.step_rows = self._read_rows(self.step_path, STEP_FIELDS)
        self.epoch_rows = self._read_rows(self.epoch_path, EPOCH_FIELDS)
        self.global_steps = {int(row["global_step"]) for row in self.step_rows}
        self.completed_epochs = {int(row["epoch"]) for row in self.epoch_rows}

    @staticmethod
    def _read_rows(path, fields):
        if not path.exists() or path.stat().st_size == 0:
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise ValueError("unexpected CSV schema in {}".format(path))
            return list(reader)

    @staticmethod
    def _append(path, fields, row):
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({field: row[field] for field in fields})

    def append_step(self, row):
        global_step = int(row["global_step"])
        if global_step in self.global_steps:
            return False
        self._append(self.step_path, STEP_FIELDS, row)
        stored = {field: str(row[field]) for field in STEP_FIELDS}
        self.step_rows.append(stored)
        self.global_steps.add(global_step)
        return True

    def existing_epoch_window(self, epoch):
        window = WeightedMetrics()
        for row in self.step_rows:
            if int(row["epoch"]) == int(epoch):
                window.add(
                    {field: float(row[field]) for field in _STEP_AVERAGE_FIELDS},
                    int(row["processed_cases"]),
                )
        return window

    def append_epoch(self, epoch, global_step):
        epoch = int(epoch)
        if epoch in self.completed_epochs:
            return False
        window = self.existing_epoch_window(epoch)
        mean = window.means()
        row = {
            "epoch": epoch,
            "global_step": int(global_step),
            "processed_cases": window.cases,
            "lr": mean["lr"],
            "fps": window.fps(),
            "data_time": mean["data_time"],
            "forward_time": mean["forward_time"],
            "total_time": mean["total_time"],
        }
        for field in CASE_METRIC_FIELDS:
            row[field] = mean[field]
        row.update({
            "grad_norm_before_clip": mean["grad_norm_before_clip"],
            "grad_norm_after_clip": mean["grad_norm_after_clip"],
            "clip_rate": mean["clip_applied"],
            "gpu_memory_allocated": mean["gpu_memory_allocated"],
            "gpu_memory_reserved": mean["gpu_memory_reserved"],
        })
        self._append(self.epoch_path, EPOCH_FIELDS, row)
        self.epoch_rows.append({field: str(row[field]) for field in EPOCH_FIELDS})
        self.completed_epochs.add(epoch)
        return True


class FCVCStepLogger:
    def __init__(self, run_dir, steps_per_epoch, print_interval=50, stream=None):
        self.writer = MetricsCSVWriter(run_dir)
        self.steps_per_epoch = int(steps_per_epoch)
        self.print_interval = int(print_interval)
        self.stream = stream if stream is not None else sys.stdout
        self.log_path = Path(run_dir) / "train.log"
        self.interval = WeightedMetrics()
        self.epoch = None
        self.epoch_window = WeightedMetrics()
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard = SummaryWriter(log_dir=str(Path(run_dir) / "tensorboard"))
        except (ImportError, ModuleNotFoundError):
            self.tensorboard = None

    def begin_epoch(self, epoch):
        self.epoch = int(epoch)
        self.interval.reset()
        self.epoch_window = self.writer.existing_epoch_window(epoch)

    @staticmethod
    def _window_values(window):
        values = window.means()
        values["fps"] = window.fps()
        return values

    def record_step(self, row):
        if int(row["epoch"]) != self.epoch:
            raise ValueError("begin_epoch must be called before record_step")
        if int(row["global_step"]) in self.writer.global_steps:
            return None
        cases = int(row["processed_cases"])
        self.interval.add(row, cases)
        self.epoch_window.add(row, cases)
        row = dict(row)
        row["fps_interval"] = self.interval.fps()
        row["fps_epoch"] = self.epoch_window.fps()
        if not self.writer.append_step(row):
            raise RuntimeError("global_step changed during CSV append")
        if self.tensorboard is not None:
            for name, value in row.items():
                if name not in {"epoch", "step_in_epoch", "global_step"}:
                    self.tensorboard.add_scalar("train/" + name, value,
                                                int(row["global_step"]))
        step = int(row["step_in_epoch"])
        if not should_print_step(step, self.steps_per_epoch, self.print_interval):
            return None
        line = format_console_line(
            self.epoch, step, self.steps_per_epoch,
            self._window_values(self.interval),
            self._window_values(self.epoch_window),
        )
        print(line, file=self.stream)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.interval.reset()
        return line

    def finish_epoch(self, global_step):
        return self.writer.append_epoch(self.epoch, global_step)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
