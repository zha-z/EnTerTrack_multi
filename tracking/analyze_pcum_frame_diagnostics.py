import argparse
import csv
import json
import os
from collections import defaultdict
from glob import glob

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import _init_paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib.test.evaluation.environment import env_settings
from lib.test.utils.pcum_diagnostics import DIAGNOSTIC_COLUMNS


NUMERIC_FIELDS = {
    "uses_gt_visible_mask",
    "frame_id",
    "remote_uav_count",
    "local_iou",
    "raw_collaborative_iou",
    "final_iou",
    "instant_delta_iou",
    "fallback_delta_iou",
    "final_delta_iou",
    "local_confidence",
    "local_score_max",
    "local_apce",
    "local_response_entropy",
    "local_bbox_motion_distance",
    "raw_collaborative_score_max",
    "raw_collaborative_apce",
    "raw_collaborative_response_entropy",
    "alignment_gate_mean",
    "alignment_gate_std",
    "fusion_gate_mean",
    "fusion_gate_std",
    "fusion_gate_min",
    "fusion_gate_max",
    "prompt_norm",
    "aligned_prompt_norm",
    "fallback_triggered",
}


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_rows(results_dir):
    pattern = os.path.join(results_dir, "*__pcum_frame_diagnostics.csv")
    rows = []
    for path in sorted(glob(pattern)):
        with open(path, "r", newline="") as fh:
            for row in csv.DictReader(fh):
                for field in NUMERIC_FIELDS:
                    row[field] = _float(row.get(field, ""))
                row["source_file"] = path
                rows.append(row)
    return rows


def _write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _valid_eval_rows(rows):
    return [
        row for row in rows
        if row["frame_id"] > 0 and np.isfinite(row["instant_delta_iou"])
    ]


def _summarize(rows, neutral_eps):
    grouped = defaultdict(list)
    for row in _valid_eval_rows(rows):
        grouped[row["current_uav"]].append(row)
        grouped["ALL"].append(row)

    summaries = []
    for uav in ("A", "B", "C", "ALL"):
        values = np.asarray(
            [row["instant_delta_iou"] for row in grouped.get(uav, [])],
            dtype=float,
        )
        final_values = np.asarray(
            [row["final_delta_iou"] for row in grouped.get(uav, [])],
            dtype=float,
        )
        if values.size == 0:
            continue
        summaries.append({
            "uav": uav,
            "frames": int(values.size),
            "mean_instant_delta_iou": float(values.mean()),
            "mean_final_delta_iou": float(final_values.mean()),
            "positive_ratio": float((values > neutral_eps).mean()),
            "neutral_ratio": float((np.abs(values) <= neutral_eps).mean()),
            "negative_ratio": float((values < -neutral_eps).mean()),
        })
    return summaries


def _sequence_summary(rows):
    grouped = defaultdict(list)
    for row in _valid_eval_rows(rows):
        grouped[(row["sequence_name"], row["current_uav"])].append(row)

    output = []
    for (sequence, uav), group in grouped.items():
        instant = np.asarray([row["instant_delta_iou"] for row in group], dtype=float)
        final = np.asarray([row["final_delta_iou"] for row in group], dtype=float)
        output.append({
            "sequence_name": sequence,
            "current_uav": uav,
            "frames": len(group),
            "mean_instant_delta_iou": float(instant.mean()),
            "mean_final_delta_iou": float(final.mean()),
        })
    return sorted(output, key=lambda row: row["mean_instant_delta_iou"], reverse=True)


def _remote_confidence(row):
    try:
        confidences = json.loads(row.get("remote_confidences", "{}"))
        participated = json.loads(row.get("remote_participated", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return float("nan")
    selected = [
        float(value) for key, value in confidences.items()
        if participated.get(key, False)
    ]
    return float(np.mean(selected)) if selected else float("nan")


def _plot_sequence(rows, output_path):
    rows = sorted(rows, key=lambda row: row["frame_id"])
    frames = np.asarray([row["frame_id"] for row in rows], dtype=float)
    series = [
        ("instant_delta_iou", [row["instant_delta_iou"] for row in rows]),
        ("local_confidence", [row["local_confidence"] for row in rows]),
        ("remote_confidence", [_remote_confidence(row) for row in rows]),
        ("fusion_gate_mean", [row["fusion_gate_mean"] for row in rows]),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for axis, (name, values) in zip(axes, series):
        values = np.asarray(values, dtype=float)
        axis.plot(frames, values, linewidth=1.2)
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
        if name == "instant_delta_iou":
            axis.axhline(0.0, color="black", linewidth=0.8)
        if name == "fusion_gate_mean" and not np.isfinite(values).any():
            axis.text(
                0.5, 0.5, "Not defined for film fusion",
                transform=axis.transAxes, ha="center", va="center",
            )
    axes[-1].set_xlabel("frame_id")
    fig.suptitle("{} / UAV {}".format(rows[0]["sequence_name"], rows[0]["current_uav"]))
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(path, rows, summaries, top_positive, top_negative, neutral_eps):
    labels = sorted(set(row["diagnostic_label"] for row in rows))
    oracle = any(bool(row["uses_gt_visible_mask"]) for row in rows)
    with open(path, "w") as fh:
        fh.write("# PCUM Frame Diagnostic Report\n\n")
        fh.write("- Diagnostic label: `{}`\n".format(", ".join(labels)))
        fh.write("- GT-visible-mask oracle: `{}`\n".format(oracle))
        if oracle:
            fh.write("- Warning: this is oracle reproduction data and is not a formal paper result.\n")
        fh.write("- Neutral threshold: `abs(instant_delta_iou) <= {:.4f}`\n\n".format(neutral_eps))
        fh.write(
            "`instant_delta_iou` measures the one-frame effect of remote prompts under "
            "the same tracker history. It does not compare two independently evolving "
            "long-term tracker trajectories.\n\n"
        )
        fh.write("## UAV Summary\n\n")
        fh.write("| UAV | Frames | Mean instant delta | Mean final delta | Positive | Neutral | Negative |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            fh.write(
                "| {uav} | {frames} | {mean_instant_delta_iou:.6f} | "
                "{mean_final_delta_iou:.6f} | {positive_ratio:.2%} | "
                "{neutral_ratio:.2%} | {negative_ratio:.2%} |\n".format(**row)
            )
        for title, selected in (("Largest Positive", top_positive), ("Largest Negative", top_negative)):
            fh.write("\n## {} Sequences\n\n".format(title))
            fh.write("| Sequence | UAV | Frames | Mean instant delta | Mean final delta |\n")
            fh.write("|---|---:|---:|---:|---:|\n")
            for row in selected:
                fh.write(
                    "| {sequence_name} | {current_uav} | {frames} | "
                    "{mean_instant_delta_iou:.6f} | {mean_final_delta_iou:.6f} |\n".format(**row)
                )


def main():
    parser = argparse.ArgumentParser(description="Analyze frame-level PCUM prompt effects.")
    parser.add_argument("--tracker_name", default="entertrack")
    parser.add_argument("--tracker_param", required=True)
    parser.add_argument("--runid", type=int, default=0)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--neutral_eps", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    settings = env_settings()
    results_dir = args.results_dir or os.path.join(
        settings.results_path,
        args.tracker_name,
        "{}_{:03d}".format(args.tracker_param, args.runid),
    )
    output_dir = args.output_dir or os.path.join(
        "output",
        "analysis",
        "pcum_frame_diagnostics",
        "{}_{:03d}".format(args.tracker_param, args.runid),
    )
    os.makedirs(output_dir, exist_ok=True)

    rows = _load_rows(results_dir)
    if not rows:
        raise RuntimeError("No PCUM frame diagnostic CSV files found in {}".format(results_dir))
    summaries = _summarize(rows, args.neutral_eps)
    sequence_rows = _sequence_summary(rows)
    top_positive = [
        row for row in sequence_rows if row["mean_instant_delta_iou"] > 0.0
    ][:args.top_k]
    top_negative = sorted(
        [row for row in sequence_rows if row["mean_instant_delta_iou"] < 0.0],
        key=lambda row: row["mean_instant_delta_iou"],
    )[:args.top_k]

    merged_fields = DIAGNOSTIC_COLUMNS + ["source_file"]
    _write_csv(os.path.join(output_dir, "all_frames.csv"), rows, merged_fields)
    _write_csv(os.path.join(output_dir, "uav_summary.csv"), summaries)
    _write_csv(os.path.join(output_dir, "sequence_summary.csv"), sequence_rows)
    _write_csv(os.path.join(output_dir, "top_positive_sequences.csv"), top_positive)
    _write_csv(os.path.join(output_dir, "top_negative_sequences.csv"), top_negative)
    _write_report(
        os.path.join(output_dir, "report.md"),
        rows,
        summaries,
        top_positive,
        top_negative,
        args.neutral_eps,
    )

    selected_keys = {
        (row["sequence_name"], row["current_uav"])
        for row in top_positive + top_negative
    }
    grouped = defaultdict(list)
    for row in rows:
        key = (row["sequence_name"], row["current_uav"])
        if key in selected_keys:
            grouped[key].append(row)
    for (sequence, uav), group in grouped.items():
        _plot_sequence(
            group,
            os.path.join(output_dir, "curves", "{}__uav-{}.png".format(sequence, uav)),
        )

    print("Frame diagnostics: {}".format(len(rows)))
    print("Report: {}".format(os.path.join(output_dir, "report.md")))


if __name__ == "__main__":
    main()
