#!/usr/bin/env python3
"""Read-only FCVC training-curve analysis."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fcvc-training-analysis")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_step_metrics(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("step_metrics.csv has no optimizer-step rows")
    output = {}
    for key in rows[0]:
        try:
            output[key] = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError):
            continue
    return output


def plot_groups(metrics, output_dir):
    step = metrics["global_step"]
    groups = {
        "loss_curves.png": (
            "Tracking losses",
            ("loss_total", "loss_track", "loss_cls", "loss_giou", "loss_l1"),
        ),
        "auxiliary_loss_curves.png": (
            "Weighted FCVC losses",
            ("loss_align_weighted", "loss_recon_weighted", "loss_safe_weighted",
             "loss_cycle_weighted", "loss_teacher_weighted"),
        ),
        "behavior_curves.png": (
            "FCVC behavior",
            ("iou", "safe_active_rate", "residual_local_ratio", "null_attention",
             "sender1_attention", "sender2_attention"),
        ),
        "optimization_curves.png": (
            "Optimization",
            ("lr", "grad_norm_before_clip", "grad_norm_after_clip", "clip_applied"),
        ),
        "performance_curves.png": (
            "Throughput and timing",
            ("fps_interval", "fps_epoch", "data_time", "forward_time", "total_time"),
        ),
    }
    generated = []
    for filename, (title, fields) in groups.items():
        fig, axes = plt.subplots(len(fields), 1, figsize=(10, 2.4 * len(fields)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, field in zip(axes, fields):
            if field not in metrics:
                raise ValueError("step_metrics.csv missing {}".format(field))
            axis.plot(step, metrics[field], linewidth=1.0)
            axis.set_ylabel(field)
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("global optimizer step")
        fig.suptitle(title)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        generated.append(str(path))
    return generated


def analyze(script, config, save_dir):
    run_dir = Path(save_dir).resolve() / script / config
    step_path = run_dir / "step_metrics.csv"
    epoch_path = run_dir / "epoch_metrics.csv"
    if not step_path.is_file():
        raise FileNotFoundError("missing {}".format(step_path))
    if not epoch_path.is_file():
        raise FileNotFoundError("missing {}".format(epoch_path))
    metrics = load_step_metrics(step_path)
    output_dir = run_dir / "training_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = plot_groups(metrics, output_dir)
    summary = {
        "input_step_metrics": str(step_path),
        "input_epoch_metrics": str(epoch_path),
        "optimizer_steps": int(metrics["global_step"].size),
        "first_global_step": int(metrics["global_step"][0]),
        "last_global_step": int(metrics["global_step"][-1]),
        "output_dir": str(output_dir),
        "generated": generated,
        "model_loaded": False,
        "image_loaded": False,
        "test_set_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("training_analysis_output={}".format(output_dir))
    for path in generated + [str(summary_path)]:
        print("generated={}".format(path))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze FCVC training CSV logs")
    parser.add_argument("--script", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--save_dir", default="./output")
    args = parser.parse_args()
    analyze(args.script, args.config, args.save_dir)


if __name__ == "__main__":
    main()
