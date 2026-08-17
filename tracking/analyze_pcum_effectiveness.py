import argparse
import csv
import os
import pickle
import re
from glob import glob
from statistics import mean, median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

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


def _best_balanced_tracker(rows, views=("Drone A", "Drone B", "Drone C")):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["tracker"], {})[row["variant"]] = row

    best_name = None
    best_key = None
    for tracker, by_view in grouped.items():
        if not all(view in by_view for view in views):
            continue
        aucs = [float(by_view[view]["auc"]) for view in views]
        key = (min(aucs), sum(aucs) / len(aucs))
        if best_key is None or key > best_key:
            best_key = key
            best_name = tracker
    return best_name


def _write_markdown_report(path, rows, baseline, candidate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    target_rows = [
        row for row in rows
        if row["tracker"] == candidate and row["variant"] in ("Drone A", "Drone B", "Drone C")
    ]
    target_ok = target_rows and all(row["auc"] >= 0.55 for row in target_rows)

    with open(path, "w") as fh:
        fh.write("# PCUM Ablation Report\n\n")
        fh.write("## Objective\n\n")
        fh.write(
            "Evaluate cross-UAV visual prompts on ThreeMDOT. "
            "The target is per-view Drone A/B/C AUC around or above 0.55.\n\n"
        )
        fh.write("## Compared Methods\n\n")
        fh.write("- baseline: EnTeRTrack/ARP without PCUM.\n")
        fh.write("- PCUM-local: local prompt only.\n")
        fh.write("- remote-visual-no-mask: real multi-view remote prompt without visible mask.\n")
        fh.write("- remote-visual-mask: real multi-view remote prompt with visible mask.\n")
        fh.write("- gated-ab-focus / gated-balanced-focus: real multi-view remote prompt with stronger A/B view loss weights.\n")
        fh.write("## Target Check\n\n")
        fh.write("- Baseline method: `%s`\n" % baseline)
        fh.write("- Main candidate: `%s`\n" % candidate)
        fh.write("- A/B/C AUC >= 0.55: `%s`\n\n" % ("PASS" if target_ok else "NOT_YET_VERIFIED_OR_FAILED"))
        fh.write("## Summary Table\n\n")
        fh.write("| Display | Variant | AUC | Norm Precision | Precision@20 | FPS mean |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fh.write(
                "| {display} | {variant} | {auc:.4f} | {norm_precision:.4f} | "
                "{precision20:.4f} | {fps_mean:.2f} |\n".format(**row)
            )
        fh.write("\n## Generated Artifacts\n\n")
        fh.write("- `tracker_summary.csv`: numeric tracker summary.\n")
        fh.write("- `target_check.md`: explicit A/B/C AUC target verdict.\n")
        fh.write("- `balanced_target_check.csv`: ranked methods by minimum and mean single-view AUC.\n")
        fh.write("- `pcum_auc_bars.png`: AUC comparison bar plot.\n")
        fh.write("- `success_curves.png`: selected success curves.\n")
        fh.write("- `delta_drone_a.png`, `delta_drone_b.png`, `delta_drone_c.png`, `delta_fused.png`: per-sequence deltas when available.\n")


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
    parser.add_argument(
        "--candidate",
        default="pcum_real_target_stable_28",
        help="Candidate tracker name, or 'auto' to select the highest min(A/B/C AUC).",
    )
    parser.add_argument(
        "--include-pattern",
        default=r"(pcum_ablation_baseline|pcum_real_target_stable|pcum_real_allviews_stable)",
        help="Regex used to select trackers for the AUC bar plot.",
    )
    args = parser.parse_args()

    with open(args.eval_pkl, "rb") as fh:
        data = pickle.load(fh)

    rows = _build_rows(data, args.results_root)
    candidate = args.candidate
    if candidate == "auto":
        candidate = _best_balanced_tracker(rows) or args.candidate
        print("Auto-selected candidate:", candidate)

    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(os.path.join(args.output_dir, "tracker_summary.csv"), rows)
    _write_markdown_report(os.path.join(args.output_dir, "report.md"), rows, args.baseline, candidate)

    _plot_auc_bars(rows, os.path.join(args.output_dir, "pcum_auc_bars.png"), args.include_pattern)

    curve_indices = []
    for target in [
        args.baseline + r" \(Fused\)",
        candidate + r" \(Fused\)",
        args.baseline + r" \(Drone A\)",
        candidate + r" \(Drone A\)",
        args.baseline + r" \(Drone B\)",
        candidate + r" \(Drone B\)",
        args.baseline + r" \(Drone C\)",
        candidate + r" \(Drone C\)",
    ]:
        curve_indices.extend(_find_tracker_indices(rows, target))
    _plot_success_curves(data, rows, curve_indices, os.path.join(args.output_dir, "success_curves.png"))

    _write_delta_analysis(data, rows, args.baseline, candidate, args.output_dir)

    print("Wrote PCUM effectiveness analysis to:", args.output_dir)
    print("Summary CSV:", os.path.join(args.output_dir, "tracker_summary.csv"))


if __name__ == "__main__":
    main()
