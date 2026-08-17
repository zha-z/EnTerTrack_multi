#!/usr/bin/env python3
"""Analyze completed J0/J1 target-group CV results without running trackers."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/controlled_baselines/pcum_joint_adaptation"
REGISTRY = OUT / "cv_experiment_registry.csv"
ASSIGNMENT = OUT / "cv_manifests/fold_assignment.csv"
GT_ROOT = Path("/data2/Three-MDOT")
METRICS = ("auc", "precision", "norm_precision")


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def target_id(sequence):
    return sequence.rsplit("-", 1)[0]


def load_xywh(path):
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except ValueError:
        arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :4]


def load_vector(path):
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def sequence_metrics(pred, gt):
    pred = np.array(pred, dtype=np.float64, copy=True)
    gt = np.asarray(gt, dtype=np.float64)
    pred[0] = gt[0]
    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    pc = pred[:, :2] + 0.5 * (pred[:, 2:] - 1.0)
    gc = gt[:, :2] + 0.5 * (gt[:, 2:] - 1.0)
    err = np.sqrt(((pc - gc) ** 2).sum(axis=1))
    errn = np.sqrt((((pc / gt[:, 2:]) - (gc / gt[:, 2:])) ** 2).sum(axis=1))
    tl = np.maximum(pred[:, :2], gt[:, :2])
    br = np.minimum(pred[:, :2] + pred[:, 2:] - 1.0, gt[:, :2] + gt[:, 2:] - 1.0)
    wh = np.maximum(br - tl + 1.0, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    err[~valid] = np.inf
    errn[~valid] = -1.0
    iou[~valid] = -1.0
    length = len(gt)
    th_o = np.arange(0, 1.0 + 0.05, 0.05)
    th_c = np.arange(0, 51)
    th_n = th_c / 100.0
    return {
        "auc": float(((iou[:, None] > th_o[None, :]).sum(axis=0) / length).mean() * 100.0),
        "precision": float(((err[:, None] <= th_c[None, :]).sum(axis=0) / length)[20] * 100.0),
        "norm_precision": float(((errn[:, None] <= th_n[None, :]).sum(axis=0) / length)[20] * 100.0),
    }


def result_dir(results_root, config, runid):
    return Path(results_root) / "entertrack" / ("%s_%03d" % (config, int(runid)))


def evaluate_target(result_path, target, sequences):
    rows = []
    for seq in sequences:
        pred_path = result_path / ("%s.txt" % seq)
        score_path = result_path / ("%s_max_score.txt" % seq)
        apce_path = result_path / ("%s_APCE.txt" % seq)
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        if not apce_path.is_file():
            raise FileNotFoundError(apce_path)
        gt_path = GT_ROOT / target / seq / "groundtruth.txt"
        gt = load_xywh(gt_path)
        pred = load_xywh(pred_path)
        score = load_vector(score_path)
        apce = load_vector(apce_path)
        if len(pred) != len(gt):
            raise RuntimeError("Length mismatch %s pred=%d gt=%d" % (seq, len(pred), len(gt)))
        if len(score) != len(gt):
            raise RuntimeError("Score length mismatch %s score=%d gt=%d" % (seq, len(score), len(gt)))
        if len(apce) != len(gt):
            raise RuntimeError("APCE length mismatch %s apce=%d gt=%d" % (seq, len(apce), len(gt)))
        if not np.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction in %s" % pred_path)
        if not np.isfinite(score).all():
            raise RuntimeError("Non-finite score in %s" % score_path)
        if not np.isfinite(apce).all():
            raise RuntimeError("Non-finite APCE in %s" % apce_path)
        row = {"sequence": seq, "target": target, "view": seq.rsplit("-", 1)[-1]}
        row.update(sequence_metrics(pred, gt))
        rows.append(row)
    out = {"target": target}
    for metric in METRICS:
        out[metric] = float(np.mean([row[metric] for row in rows]))
    out["sequence_rows"] = rows
    return out


def bootstrap(values, seed=20260715, samples=100000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "std_delta": float(values.std(ddof=1)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "prob_delta_gt_0": float((draws > 0).mean()),
        "positive_targets": int((values > 1e-9).sum()),
        "negative_targets": int((values < -1e-9).sum()),
        "tied_targets": int((np.abs(values) <= 1e-9).sum()),
    }


def analyze(registry_path=REGISTRY, assignment_path=ASSIGNMENT,
            results_root=ROOT / "output/test/tracking_results",
            output_dir=OUT / "cv_analysis", bootstrap_samples=100000):
    registry = read_csv(registry_path)
    assignments = read_csv(assignment_path)
    holdout_by_fold = {}
    seqs_by_target = {}
    for row in assignments:
        if row["role"] != "holdout":
            continue
        fold = int(row["fold_id"])
        target = row["target_id"]
        seqs = row["view_sequences"].split("|")
        holdout_by_fold.setdefault(fold, []).append(target)
        seqs_by_target[target] = seqs
    if sorted(holdout_by_fold) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Missing fold assignment; complete conclusion refused")

    reg = {(int(r["fold_id"]), r["model_role"]): r for r in registry}
    target_rows = []
    seq_rows = []
    fold_rows = []
    missing = []
    for fold in range(5):
        fold_deltas = {metric: [] for metric in METRICS}
        for role in ("J0", "J1"):
            if (fold, role) not in reg:
                missing.append("registry fold %d %s" % (fold, role))
        if missing:
            continue
        j0_dir = result_dir(results_root, reg[(fold, "J0")]["config"], reg[(fold, "J0")]["evaluation_runid"])
        j1_dir = result_dir(results_root, reg[(fold, "J1")]["config"], reg[(fold, "J1")]["evaluation_runid"])
        if not j0_dir.is_dir():
            missing.append(str(j0_dir))
        if not j1_dir.is_dir():
            missing.append(str(j1_dir))
        if missing:
            continue
        for target in holdout_by_fold[fold]:
            j0 = evaluate_target(j0_dir, target, seqs_by_target[target])
            j1 = evaluate_target(j1_dir, target, seqs_by_target[target])
            row = {"fold_id": fold, "target": target}
            for metric in METRICS:
                row["j0_" + metric] = j0[metric]
                row["j1_" + metric] = j1[metric]
                row["delta_" + metric] = j1[metric] - j0[metric]
                fold_deltas[metric].append(row["delta_" + metric])
            target_rows.append(row)
            for left, right in zip(j0["sequence_rows"], j1["sequence_rows"]):
                seq_row = {
                    "fold_id": fold,
                    "target": target,
                    "sequence": left["sequence"],
                    "view": left["view"],
                }
                for metric in METRICS:
                    seq_row["j0_" + metric] = left[metric]
                    seq_row["j1_" + metric] = right[metric]
                    seq_row["delta_" + metric] = right[metric] - left[metric]
                seq_rows.append(seq_row)
        fold_row = {"fold_id": fold}
        for metric in METRICS:
            fold_row["mean_delta_" + metric] = float(np.mean(fold_deltas[metric]))
        fold_rows.append(fold_row)

    if missing:
        raise RuntimeError("Missing fold result(s); complete conclusion refused: %s" % ", ".join(missing[:10]))
    if len(target_rows) != 23:
        raise RuntimeError("Expected 23 held-out targets, got %d" % len(target_rows))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "target_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_rows[0].keys()))
        writer.writeheader()
        writer.writerows(target_rows)
    with (output_dir / "sequence_view_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seq_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seq_rows)
    with (output_dir / "fold_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    summary_rows = []
    metric_summary = {}
    for metric in METRICS:
        values = [row["delta_" + metric] for row in target_rows]
        stats = bootstrap(values, seed=20260715, samples=bootstrap_samples)
        stats["j0_mean"] = float(np.mean([row["j0_" + metric] for row in target_rows]))
        stats["j1_mean"] = float(np.mean([row["j1_" + metric] for row in target_rows]))
        loo = []
        for fold in range(5):
            fold_values = [row["delta_" + metric] for row in target_rows if row["fold_id"] != fold]
            loo.append(float(np.mean(fold_values)))
        stats["leave_one_fold_out_values"] = loo
        stats["leave_one_fold_out_min"] = min(loo)
        stats["leave_one_fold_out_max"] = max(loo)
        stats["leave_one_fold_out_sign_flip"] = any(v <= 0 for v in loo) if stats["mean_delta"] > 0 else any(v > 0 for v in loo)
        positive = sorted(
            ((row["target"], row["delta_" + metric]) for row in target_rows
             if row["delta_" + metric] > 1e-9),
            key=lambda item: item[1], reverse=True)
        negative = sorted(
            ((row["target"], row["delta_" + metric]) for row in target_rows
             if row["delta_" + metric] < -1e-9),
            key=lambda item: item[1])
        positive_sum = float(sum(value for _, value in positive))
        stats["positive_gain_sum"] = positive_sum
        stats["positive_gain_concentration_max"] = (
            float(positive[0][1] / positive_sum) if positive_sum > 0 else None)
        stats["top_positive_targets"] = [
            {"target": target, "delta": float(value)} for target, value in positive[:5]]
        stats["top_negative_targets"] = [
            {"target": target, "delta": float(value)} for target, value in negative[:5]]
        metric_summary[metric] = stats
        row = {"metric": metric}
        row.update({key: value for key, value in stats.items()
                    if not isinstance(value, (list, dict))})
        summary_rows.append(row)

    view_metrics = {}
    for view, label in (("1", "Drone_A"), ("2", "Drone_B"), ("3", "Drone_C")):
        rows = [row for row in seq_rows if row["view"] == view]
        if len(rows) != 23:
            raise RuntimeError("Expected 23 target-view rows for %s, got %d" % (label, len(rows)))
        view_metrics[label] = {}
        for metric in METRICS:
            view_metrics[label][metric] = {
                "j0_mean": float(np.mean([row["j0_" + metric] for row in rows])),
                "j1_mean": float(np.mean([row["j1_" + metric] for row in rows])),
                "mean_delta": float(np.mean([row["delta_" + metric] for row in rows])),
            }

    auc = metric_summary["auc"]
    precision = metric_summary["precision"]
    norm_precision = metric_summary["norm_precision"]
    fold_auc = [row["mean_delta_auc"] for row in fold_rows]
    gate_checks = {
        "auc_mean_at_least_0_50": auc["mean_delta"] >= 0.50,
        "precision_mean_positive": precision["mean_delta"] > 0,
        "norm_precision_mean_positive": norm_precision["mean_delta"] > 0,
        "auc_at_least_13_positive_targets": auc["positive_targets"] >= 13,
        "auc_positive_targets_exceed_negative": auc["positive_targets"] > auc["negative_targets"],
        "at_least_4_positive_folds": sum(value > 0 for value in fold_auc) >= 4,
        "all_leave_one_fold_out_auc_positive": all(
            value > 0 for value in auc["leave_one_fold_out_values"]),
        "max_positive_auc_gain_concentration_at_most_0_25": (
            auc["positive_gain_concentration_max"] is not None and
            auc["positive_gain_concentration_max"] <= 0.25),
        "fixed_epoch15_seed42_no_selection": True,
    }
    primary_pass = all(gate_checks.values())
    if not primary_pass:
        tier = "C_NEGATIVE_OR_CONDITIONAL"
    elif auc["ci_low"] > 0:
        tier = "A_STRONG"
    else:
        tier = "B_EXPLORATORY_TREND"
    summary = {
        "protocol": {
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": 20260715,
            "independent_unit": "target",
            "target_count": 23,
            "view_sequence_count": 69,
        },
        "metrics": metric_summary,
        "view_metrics": view_metrics,
        "fold_deltas": fold_rows,
        "accuracy_gate": {
            "checks": gate_checks,
            "primary_pass": primary_pass,
            "tier": tier,
        },
    }
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def synthetic_smoke():
    with tempfile.TemporaryDirectory(prefix="j0j1-cv-smoke-") as tmp:
        tmp = Path(tmp)
        registry = read_csv(REGISTRY)
        assignments = read_csv(ASSIGNMENT)
        for row in registry:
            directory = result_dir(tmp, row["config"], row["evaluation_runid"])
            directory.mkdir(parents=True, exist_ok=True)
            fold = row["fold_id"]
            for a in assignments:
                if a["fold_id"] != fold or a["role"] != "holdout":
                    continue
                for seq in a["view_sequences"].split("|"):
                    gt = load_xywh(GT_ROOT / target_id(seq) / seq / "groundtruth.txt")
                    np.savetxt(directory / ("%s.txt" % seq), gt, delimiter=",")
                    np.savetxt(directory / ("%s_max_score.txt" % seq), np.ones(len(gt)))
                    np.savetxt(directory / ("%s_APCE.txt" % seq), np.ones(len(gt)))
        summary = analyze(results_root=tmp, output_dir=tmp / "analysis",
                          bootstrap_samples=1000)
        missing_refused = False
        first = next(path for path in tmp.glob("entertrack/*/*.txt")
                     if not path.name.endswith("_max_score.txt") and not path.name.endswith("_APCE.txt"))
        first.unlink()
        try:
            analyze(results_root=tmp, output_dir=tmp / "analysis2",
                    bootstrap_samples=1000)
        except (RuntimeError, FileNotFoundError):
            missing_refused = True
        if not missing_refused:
            raise RuntimeError("Synthetic missing-fold/result refusal failed")
        print(json.dumps({"synthetic_smoke": "PASS", "metrics": summary}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(REGISTRY))
    parser.add_argument("--assignment", default=str(ASSIGNMENT))
    parser.add_argument("--results-root", default=str(ROOT / "output/test/tracking_results"))
    parser.add_argument("--output-dir", default=str(OUT / "cv_analysis"))
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    summary = analyze(args.registry, args.assignment, args.results_root,
                      args.output_dir, args.bootstrap_samples)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
