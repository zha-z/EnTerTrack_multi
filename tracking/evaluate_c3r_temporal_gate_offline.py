"""Preregistered fold1 inner-dev regression screen for Temporal Gate v2."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.models.entertrack.temporal_gate import load_temporal_gate_checkpoint
from lib.train.dataset.c3r_temporal_gate import (
    C3RTemporalGateDataset,
    collate_temporal_gate,
    read_id_file,
)


BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260719

# Frozen before any fold1 image, GT, prediction, label, or metric access.
GATE1_V2_THRESHOLDS = {
    "target_clustered_spearman_min": 0.20,
    "spearman_bootstrap_ci_low_strict_min": 0.00,
    "pearson_min": 0.20,
    "mae_relative_improvement_min": 0.02,
    "rmse_relative_improvement_min": 0.02,
    "sign_roc_auc_min": 0.60,
    "sign_pr_auc_uplift_min": 0.05,
    "receiver_macro_spearman_min": 0.10,
    "sender_macro_spearman_min": 0.10,
    "prediction_std_min": 0.01,
    "positive_prevalence_min": 0.10,
    "negative_prevalence_min": 0.10,
}


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def pearson(targets: Sequence[float], predictions: Sequence[float]) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(targets) < 2:
        return float("nan")
    if targets.std() == 0.0 or predictions.std() == 0.0:
        return 0.0
    return float(np.corrcoef(targets, predictions)[0, 1])


def spearman(targets: Sequence[float], predictions: Sequence[float]) -> float:
    return pearson(_rankdata(targets), _rankdata(predictions))


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = _rankdata(scores)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives)


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    ranked = labels[np.argsort(-scores, kind="mergesort")]
    precision = np.cumsum(ranked == 1) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def _core_metrics(targets: Sequence[float], predictions: Sequence[float]):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    zero_mae = float(np.mean(np.abs(targets)))
    zero_rmse = float(np.sqrt(np.mean(targets ** 2)))
    signs = (targets > 0.0).astype(np.int64)
    return {
        "target_clustered_spearman": spearman(targets, predictions),
        "pearson": pearson(targets, predictions),
        "mae": mae,
        "rmse": rmse,
        "zero_predictor_mae": zero_mae,
        "zero_predictor_rmse": zero_rmse,
        "mae_relative_improvement": (
            (zero_mae - mae) / zero_mae if zero_mae > 0.0 else float("nan")),
        "rmse_relative_improvement": (
            (zero_rmse - rmse) / zero_rmse
            if zero_rmse > 0.0 else float("nan")),
        "sign_roc_auc": roc_auc(signs, predictions),
        "sign_pr_auc": pr_auc(signs, predictions),
    }


def _group_stability(targets, predictions, groups):
    grouped = defaultdict(lambda: ([], []))
    for target, prediction, group in zip(targets, predictions, groups):
        grouped[group][0].append(target)
        grouped[group][1].append(prediction)
    values = defaultdict(list)
    for target_values, prediction_values in grouped.values():
        signs = [int(value > 0.0) for value in target_values]
        for name, value in (
                ("spearman", spearman(target_values, prediction_values)),
                ("pearson", pearson(target_values, prediction_values)),
                ("sign_roc_auc", roc_auc(signs, prediction_values))):
            if math.isfinite(value):
                values[name].append(value)
    report = {"groups": len(grouped)}
    for name in ("spearman", "pearson", "sign_roc_auc"):
        observed = values[name]
        fallback = 0.5 if name == "sign_roc_auc" else 0.0
        report[name + "_finite_groups"] = len(observed)
        report[name + "_macro"] = (
            float(np.mean(observed)) if observed else fallback)
        report[name + "_std"] = (
            float(np.std(observed)) if observed else 0.0)
        report[name + "_min"] = (
            float(np.min(observed)) if observed else fallback)
        report[name + "_max"] = (
            float(np.max(observed)) if observed else fallback)
    return report


def target_cluster_bootstrap(targets, predictions, target_ids,
                             draws=BOOTSTRAP_DRAWS,
                             seed=BOOTSTRAP_SEED):
    target_values = sorted(set(target_ids))
    if len(target_values) < 2:
        raise RuntimeError("target-cluster bootstrap requires at least two targets")
    grouped = {
        target: [index for index, value in enumerate(target_ids)
                 if value == target]
        for target in target_values
    }
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(int(draws)):
        chosen = rng.choice(
            target_values, size=len(target_values), replace=True)
        indices = [index for target in chosen for index in grouped[target]]
        metrics = _core_metrics(
            [targets[index] for index in indices],
            [predictions[index] for index in indices])
        for name, value in metrics.items():
            if math.isfinite(value):
                samples[name].append(value)
    intervals = {}
    for name, values in samples.items():
        if values:
            intervals[name] = [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
    return intervals


def evaluate(model, dataset: C3RTemporalGateDataset,
             batch_size: int = 256) -> Dict[str, object]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_temporal_gate)
    targets, predictions, gates = [], [], []
    target_ids, receivers, senders = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            gate, raw_utility = model(batch["history"], batch["lengths"])
            if not bool(torch.isfinite(gate).all().item()) or not bool(
                    torch.isfinite(raw_utility).all().item()):
                raise RuntimeError("non-finite Temporal Gate v2 output")
            targets.extend(float(value) for value in batch[
                "delta_diou"].reshape(-1).tolist())
            predictions.extend(float(value) for value in raw_utility.reshape(-1).tolist())
            gates.extend(float(value) for value in gate.reshape(-1).tolist())
            target_ids.extend(item["target_id"] for item in batch["metadata"])
            receivers.extend(item["receiver_id"] for item in batch["metadata"])
            senders.extend(item["sender_id"] for item in batch["metadata"])
    if not targets:
        raise RuntimeError("empty inner-dev regression dataset")
    positive_prevalence = float(np.mean(np.asarray(targets) > 0.0))
    negative_prevalence = 1.0 - positive_prevalence
    if positive_prevalence == 0.0 or negative_prevalence == 0.0:
        raise RuntimeError("sign metrics require positive and non-positive utility")
    result = _core_metrics(targets, predictions)
    result.update({
        "rows": len(targets),
        "positive_prevalence": positive_prevalence,
        "negative_prevalence": negative_prevalence,
        "zero_prevalence": float(np.mean(np.asarray(targets) == 0.0)),
        "prediction_std": float(np.std(predictions)),
        "prediction_mean": float(np.mean(predictions)),
        "gate_std": float(np.std(gates)),
        "gate_min": float(np.min(gates)),
        "gate_max": float(np.max(gates)),
        "receiver_stability": _group_stability(
            targets, predictions, receivers),
        "sender_stability": _group_stability(
            targets, predictions, senders),
        "target_stability": _group_stability(
            targets, predictions, target_ids),
        "target_cluster_bootstrap_95ci": target_cluster_bootstrap(
            targets, predictions, target_ids),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gate1_v2_thresholds": dict(GATE1_V2_THRESHOLDS),
    })
    ci_low = result["target_cluster_bootstrap_95ci"][
        "target_clustered_spearman"][0]
    checks = {
        "target_clustered_spearman": result[
            "target_clustered_spearman"] >= 0.20,
        "spearman_bootstrap_ci_low": ci_low > 0.00,
        "pearson": result["pearson"] >= 0.20,
        "mae_relative_improvement": result[
            "mae_relative_improvement"] >= 0.02,
        "rmse_relative_improvement": result[
            "rmse_relative_improvement"] >= 0.02,
        "sign_roc_auc": result["sign_roc_auc"] >= 0.60,
        "sign_pr_auc_uplift": result["sign_pr_auc"] >= (
            positive_prevalence + 0.05),
        "receiver_macro_spearman": result[
            "receiver_stability"]["spearman_macro"] >= 0.10,
        "sender_macro_spearman": result[
            "sender_stability"]["spearman_macro"] >= 0.10,
        "prediction_std": result["prediction_std"] >= 0.01,
        "positive_prevalence": positive_prevalence >= 0.10,
        "negative_prevalence": negative_prevalence >= 0.10,
    }
    result["gate1_v2_checks"] = checks
    result["gate1_v2_offline_pass"] = bool(all(checks.values()))
    scalar_values = [value for value in result.values()
                     if isinstance(value, (float, int))]
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise RuntimeError("offline regression diagnostics contain non-finite values")
    result["finite_diagnostics"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-jsonl", required=True)
    parser.add_argument("--inner-dev-ids", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = C3RTemporalGateDataset.from_jsonl(
        args.labeled_jsonl,
        allowed_targets=read_id_file(args.inner_dev_ids))
    model = load_temporal_gate_checkpoint(
        args.checkpoint, expected_sha256=args.checkpoint_sha256)
    report = evaluate(model, dataset)
    Path(args.output).write_text(json.dumps(
        report, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
