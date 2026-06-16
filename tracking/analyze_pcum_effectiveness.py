import argparse
import csv
import os
import pickle
import re
from glob import glob
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _tracker_field(tracker, key, default=""):
    if isinstance(tracker, dict):
        return tracker.get(key, default)
    return getattr(tracker, key, default)


def _variant_name(display_name):
    match = re.search(r"\((Drone A|Drone B|Drone C|Fused)\)$", display_name)
    return match.group(1) if match else ""


def _base_name(display_name):
    return re.sub(r"\s+\((Drone A|Drone B|Drone C|Fused)\)$", "", display_name)


def _result_dir(results_root, param, run_id):
    return os.path.join(results_root, "%s_%03d" % (param, int(run_id or 0)))


def _fps_from_dir(path):
    times = []
    for filename in glob(os.path.join(path, "*_time.txt")):
        try:
            with open(filename, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    value = float(line.split()[0].replace(",", ""))
                    if value > 0:
                        times.append(value)
        except OSError:
            continue
    if not times:
        return float("nan"), float("nan")
    return 1.0 / mean(times), 1.0 / median(times)


def _build_rows(data, results_root):
    success = np.asarray(data["ave_success_rate_plot_overlap"], dtype=float)
    precision = np.asarray(data["ave_success_rate_plot_center"], dtype=float)
    norm_precision = np.asarray(data["ave_success_rate_plot_center_norm"], dtype=float)
    avg_overlap = np.asarray(data["avg_overlap_all"], dtype=float)

    rows = []
    for idx, tracker in enumerate(data["trackers"]):
        display = _tracker_field(tracker, "disp_name", str(tracker))
        param = _tracker_field(tracker, "param", "")
        run_id = int(_tracker_field(tracker, "run_id", 0) or 0)
        variant = _variant_name(display)
        result_dir = _result_dir(results_root, param, run_id)
        fps_mean, fps_median = _fps_from_dir(result_dir)
        rows.append({
            "index": idx,
            "display": display,
            "tracker": _base_name(display),
            "variant": variant,
            "param": param,
            "run_id": run_id,
            "auc": float(success[:, idx, :].mean()),
            "precision20": float(precision[:, idx, 20].mean()) if precision.shape[-1] > 20 else float(precision[:, idx, :].mean()),
            "norm_precision": float(norm_precision[:, idx, :].mean()),
            "avg_overlap": float(np.nanmean(avg_overlap[:, idx])),
            "fps_mean": fps_mean,
            "fps_median": fps_median,
        })
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _find_tracker_indices(rows, pattern):
    regex = re.compile(pattern)
    return [row["index"] for row in rows if regex.search(row["display"])]


def _plot_success_curves(data, rows, indices, output_path):
    if not indices:
        return
    thresholds = np.asarray(data["threshold_set_overlap"], dtype=float)
    success = np.asarray(data["ave_success_rate_plot_overlap"], dtype=float)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(9, 6))
    for idx in indices:
        curve = success[:, idx, :].mean(axis=0)
        row = rows[idx]
        plt.plot(thresholds, curve, linewidth=2, label="%s AUC %.3f" % (row["display"], row["auc"]))

    plt.xlabel("Overlap threshold")
    plt.ylabel("Success rate")
    plt.title("ThreeMDOT Success Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_auc_bars(rows, output_path, include_pattern):
    regex = re.compile(include_pattern)
    selected = [row for row in rows if regex.search(row["display"])]
    if not selected:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = [row["display"].replace("pcum_", "") for row in selected]
    values = [row["auc"] for row in selected]
    colors = ["#4C78A8" if "baseline" in row["display"] else "#F58518" for row in selected]

    height = max(5, len(selected) * 0.28)
    plt.figure(figsize=(10, height))
    y = np.arange(len(selected))
    plt.barh(y, values, color=colors)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("AUC")
    plt.title("PCUM AUC Comparison")
    plt.xlim(0.35, max(values) + 0.04)
    for yi, value in zip(y, values):
        plt.text(value + 0.003, yi, "%.3f" % value, va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_delta_analysis(data, rows, baseline_name, candidate_name, output_dir):
    success = np.asarray(data["ave_success_rate_plot_overlap"], dtype=float)
    sequences = data["sequences"]
    os.makedirs(output_dir, exist_ok=True)

    baseline = {row["variant"]: row for row in rows if row["tracker"] == baseline_name}
    candidate = {row["variant"]: row for row in rows if row["tracker"] == candidate_name}

    delta_rows = []
    for variant in ["Drone A", "Drone B", "Drone C", "Fused"]:
        if variant not in baseline or variant not in candidate:
            continue
        b_idx = baseline[variant]["index"]
        c_idx = candidate[variant]["index"]
        deltas = success[:, c_idx, :].mean(axis=1) - success[:, b_idx, :].mean(axis=1)
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        ties = int((deltas == 0).sum())

        for seq, delta in zip(sequences, deltas):
            seq_name = getattr(seq, "name", str(seq))
            delta_rows.append({
                "variant": variant,
                "sequence": seq_name,
                "delta_auc": float(delta),
            })

        sorted_idx = np.argsort(deltas)
        labels = [getattr(sequences[i], "name", str(sequences[i])) for i in sorted_idx]
        sorted_deltas = deltas[sorted_idx]
        colors = ["#E45756" if v < 0 else "#54A24B" for v in sorted_deltas]
        plt.figure(figsize=(11, max(5, len(labels) * 0.18)))
        y = np.arange(len(labels))
        plt.barh(y, sorted_deltas, color=colors)
        plt.axvline(0.0, color="black", linewidth=1)
        plt.yticks(y, labels, fontsize=7)
        plt.xlabel("AUC delta vs baseline")
        plt.title("%s vs %s: %s per-sequence delta (wins %d / losses %d / ties %d)" %
                  (candidate_name, baseline_name, variant, wins, losses, ties))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "delta_%s.png" % variant.lower().replace(" ", "_")), dpi=200)
        plt.close()

    _write_csv(os.path.join(output_dir, "per_sequence_delta.csv"), delta_rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze PCUM effectiveness from eval_data.pkl")
    parser.add_argument("--eval-pkl", default="output/test/result_plots/threemdot_test/eval_data.pkl")
    parser.add_argument("--results-root", default="output/test/tracking_results/entertrack")
    parser.add_argument("--output-dir", default="output/analysis/pcum_effectiveness")
    parser.add_argument("--baseline", default="pcum_ablation_baseline")
    parser.add_argument("--candidate", default="pcum_real_target_stable_28")
    args = parser.parse_args()

    with open(args.eval_pkl, "rb") as fh:
        data = pickle.load(fh)

    rows = _build_rows(data, args.results_root)
    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(os.path.join(args.output_dir, "tracker_summary.csv"), rows)

    include_pattern = r"(pcum_ablation_baseline|pcum_real_target_stable|pcum_real_allviews_stable)"
    _plot_auc_bars(rows, os.path.join(args.output_dir, "pcum_auc_bars.png"), include_pattern)

    curve_indices = []
    for target in [
        args.baseline + r" \(Fused\)",
        args.candidate + r" \(Fused\)",
        args.baseline + r" \(Drone A\)",
        args.candidate + r" \(Drone A\)",
        args.baseline + r" \(Drone B\)",
        args.candidate + r" \(Drone B\)",
        args.baseline + r" \(Drone C\)",
        args.candidate + r" \(Drone C\)",
    ]:
        curve_indices.extend(_find_tracker_indices(rows, target))
    _plot_success_curves(data, rows, curve_indices, os.path.join(args.output_dir, "success_curves.png"))

    _write_delta_analysis(data, rows, args.baseline, args.candidate, args.output_dir)

    print("Wrote PCUM effectiveness analysis to:", args.output_dir)
    print("Summary CSV:", os.path.join(args.output_dir, "tracker_summary.csv"))


if __name__ == "__main__":
    main()
