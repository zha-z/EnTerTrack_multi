"""OSTrack-compatible read-only analysis for existing tracking results."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from lib.test.analysis.extract_results import calc_seq_err_robust


def _load_boxes(path):
    try:
        boxes = np.loadtxt(str(path), delimiter=",", dtype=float)
    except ValueError:
        boxes = np.loadtxt(str(path), dtype=float)
    return boxes.reshape(-1, 4)


def _curves(pred, target, target_visible=None, dataset="threemdot"):
    """Calculate metrics with the repository's standard OSTrack evaluator.

    This intentionally delegates sequence alignment, first-frame handling,
    visibility masking, and error calculation to ``calc_seq_err_robust`` so
    this compact JSON/CSV exporter cannot silently drift from
    ``extract_results.py``.
    """
    pred = torch.as_tensor(np.asarray(pred, dtype=float).reshape(-1, 4))
    target = torch.as_tensor(np.asarray(target, dtype=float).reshape(-1, 4))
    visible = (
        torch.as_tensor(np.asarray(target_visible), dtype=torch.uint8).reshape(-1)
        if target_visible is not None else None
    )
    overlap, error, norm_error, valid = calc_seq_err_robust(
        pred, target, dataset, visible
    )
    sequence_length = int(target.shape[0])
    if sequence_length <= 0:
        raise ValueError("cannot evaluate an empty sequence")

    overlap_thresholds = torch.arange(0.0, 1.05, 0.05, dtype=torch.float64)
    center_thresholds = torch.arange(0, 51, dtype=torch.float64)
    norm_center_thresholds = center_thresholds / 100.0
    success_curve = (
        (overlap.view(-1, 1) > overlap_thresholds.view(1, -1))
        .sum(0).double() / sequence_length
    )
    precision_curve = (
        (error.view(-1, 1) <= center_thresholds.view(1, -1))
        .sum(0).double() / sequence_length
    )
    norm_precision_curve = (
        (norm_error.view(-1, 1) <= norm_center_thresholds.view(1, -1))
        .sum(0).double() / sequence_length
    )
    return {
        "auc": float(success_curve.mean().item()),
        "precision": float(precision_curve[20].item()),
        "normalized_precision": float(norm_precision_curve[20].item()),
        "mean_iou": float(overlap[valid].mean().item()),
        "frame_count": sequence_length,
    }


def _view(name):
    return {"-1": "A", "-2": "B", "-3": "C"}.get(name[-2:], "unknown")


def _target(name):
    return name.rsplit("-", 1)[0]


def _bootstrap(values, seed=42, samples=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    means = values[rng.randint(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def analyze(tracker_name, tracker_param, dataset_name, run_id=None,
            compare_param=None, output_dir=None):
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.environment import env_settings

    env = env_settings()
    suffix = tracker_param if run_id is None else "{}_{:03d}".format(tracker_param, run_id)
    results_dir = Path(env.results_path) / tracker_name / suffix
    compare_dir = (
        Path(env.results_path) / tracker_name / compare_param
        if compare_param else None
    )
    dataset = get_dataset(dataset_name)
    rows = []
    comparisons = []
    runtime = []
    state_digests = []
    for sequence in dataset:
        name = sequence.name
        result_path = results_dir / (name + ".txt")
        if not result_path.is_file():
            continue
        pred = _load_boxes(result_path)
        target = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        if len(pred) == 0 or len(target) == 0:
            continue
        metrics = _curves(
            pred,
            target,
            target_visible=getattr(sequence, "target_visible", None),
            dataset=sequence.dataset,
        )
        rows.append({"sequence": name, "target": _target(name), "view": _view(name), **metrics})
        if compare_dir is not None:
            reference_path = compare_dir / (name + ".txt")
            if reference_path.is_file():
                reference = _load_boxes(reference_path)
                if len(reference) > 0:
                    reference_metrics = _curves(
                        reference,
                        target,
                        target_visible=getattr(sequence, "target_visible", None),
                        dataset=sequence.dataset,
                    )
                    comparisons.append({
                        "sequence": name,
                        "auc_delta": metrics["auc"] - reference_metrics["auc"],
                        "precision_delta": metrics["precision"] - reference_metrics["precision"],
                        "normalized_precision_delta": metrics["normalized_precision"] - reference_metrics["normalized_precision"],
                    })
        time_path = results_dir / (name + "_time.txt")
        if time_path.is_file():
            values = np.loadtxt(str(time_path), dtype=float).reshape(-1)
            runtime.extend(values[np.isfinite(values)].tolist())
        state_path = results_dir / (name + "_state_digest.txt")
        if state_path.is_file():
            state_digests.extend(state_path.read_text(encoding="utf-8").splitlines())
    if not rows:
        raise FileNotFoundError("no existing prediction results in {}".format(results_dir))

    def grouped(field):
        output = {}
        for value in sorted({row[field] for row in rows}):
            group = [row for row in rows if row[field] == value]
            output[value] = {
                key: float(np.nanmean([item[key] for item in group]))
                for key in ("auc", "precision", "normalized_precision", "mean_iou")
            }
        return output

    aucs = [row["auc"] for row in rows]
    summary = {
        "tracker_name": tracker_name,
        "tracker_param": tracker_param,
        "dataset_name": dataset_name,
        "sequence_count": len(rows),
        "evaluation_protocol": {
            "name": "ostrack",
            "implementation": "lib.test.analysis.extract_results.calc_seq_err_robust",
            "success_comparator": ">",
            "uses_target_visible": True,
            "exclude_invalid_frames": False,
            "sequence_macro_average": True,
        },
        "overall": {
            key: float(np.nanmean([row[key] for row in rows]))
            for key in ("auc", "precision", "normalized_precision", "mean_iou")
        },
        "per_target": grouped("target"),
        "per_view": grouped("view"),
        "bootstrap_auc_95ci": _bootstrap(aucs),
        "runtime": {
            "frame_count": len(runtime),
            "mean_seconds": float(np.mean(runtime)) if runtime else float("nan"),
            "fps": float(1.0 / np.mean(runtime)) if runtime and np.mean(runtime) > 0 else float("nan"),
        },
        "state_identity": {
            "digest_count": len(state_digests),
            "digest_sha256": hashlib.sha256("\n".join(state_digests).encode("utf-8")).hexdigest()
            if state_digests else None,
        },
        "comparison": {
            "reference": compare_param,
            "matched_sequences": len(comparisons),
            "mean_auc_delta": float(np.mean([row["auc_delta"] for row in comparisons]))
            if comparisons else float("nan"),
            "harmful_sequences": [
                row for row in comparisons if row["auc_delta"] < 0.0
            ],
        },
    }
    output_dir = Path(output_dir or results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sequence_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze existing tracking predictions")
    parser.add_argument("--tracker_name", required=True)
    parser.add_argument("--tracker_param", required=True)
    parser.add_argument("--dataset_name", "--dataset", dest="dataset_name", required=True)
    parser.add_argument("--runid", type=int, default=None)
    parser.add_argument("--compare_param", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args(argv)
    arguments = vars(args)
    arguments["run_id"] = arguments.pop("runid")
    return analyze(**arguments)
