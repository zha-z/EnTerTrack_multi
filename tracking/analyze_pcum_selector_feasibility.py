#!/usr/bin/env python3
"""Audit PCUM offline selector data and oracle feasibility."""

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.pcum_selector_utils import (
    FEATURE_COLUMNS,
    delta_stats,
    evaluate_rows_as_predictions,
    format_delta,
    format_metric,
    format_top,
    read_selector_csv,
    summarize_by_view,
    summarize_metric_rows,
    validate_feature_columns,
)


def label_stats(rows):
    total = len(rows)
    usable = [row for row in rows if int(row["ignore"]) == 0]
    positives = [row for row in usable if int(row["label"]) == 1]
    negatives = [row for row in usable if int(row["label"]) == 0]
    deltas = [float(row["loss_delta"]) for row in rows]
    by_sequence = defaultdict(list)
    for row in rows:
        if int(row["ignore"]) == 0:
            by_sequence[row["sequence"]].append(int(row["label"]))
    seq_ratios = []
    for sequence, labels in by_sequence.items():
        if labels:
            seq_ratios.append((sequence, sum(labels) / float(len(labels)) * 100.0, len(labels)))
    return {
        "total": total,
        "usable": len(usable),
        "positive": len(positives),
        "negative": len(negatives),
        "ignore": total - len(usable),
        "positive_ratio": len(positives) / max(total, 1) * 100.0,
        "negative_ratio": len(negatives) / max(total, 1) * 100.0,
        "ignore_ratio": (total - len(usable)) / max(total, 1) * 100.0,
        "loss_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "loss_delta_std": float(np.std(deltas)) if deltas else 0.0,
        "loss_delta_min": float(np.min(deltas)) if deltas else 0.0,
        "loss_delta_max": float(np.max(deltas)) if deltas else 0.0,
        "sequence_positive_ratios": sorted(seq_ratios, key=lambda item: item[1], reverse=True),
    }


def feature_snapshot(rows):
    lines = []
    lines.append("| Feature | Mean | Std | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for feature in FEATURE_COLUMNS:
        values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
        lines.append("| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |".format(
            feature, float(values.mean()), float(values.std()), float(values.min()), float(values.max())
        ))
    return lines


def oracle_section(rows, dataset_name):
    local_rows = evaluate_rows_as_predictions(rows, dataset_name, "local")
    collab_rows = evaluate_rows_as_predictions(rows, dataset_name, "collab")
    oracle_rows = evaluate_rows_as_predictions(rows, dataset_name, "oracle")
    local_summary = summarize_metric_rows(local_rows)
    collab_summary = summarize_metric_rows(collab_rows)
    oracle_summary = summarize_metric_rows(oracle_rows)
    return {
        "local_rows": local_rows,
        "collab_rows": collab_rows,
        "oracle_rows": oracle_rows,
        "local": local_summary,
        "collab": collab_summary,
        "oracle": oracle_summary,
        "oracle_delta_vs_collab": {
            "auc": oracle_summary["auc"] - collab_summary["auc"],
            "precision": oracle_summary["precision"] - collab_summary["precision"],
            "norm_precision": oracle_summary["norm_precision"] - collab_summary["norm_precision"],
        },
        "oracle_negative": delta_stats(oracle_rows, local_rows),
        "collab_negative": delta_stats(collab_rows, local_rows),
        "views": {
            "local": summarize_by_view(local_rows),
            "collab": summarize_by_view(collab_rows),
            "oracle": summarize_by_view(oracle_rows),
        },
    }


def write_report(path, train_stats, val_stats, oracle, min_oracle_gain):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# PCUM-v2 B1 Offline Selector Feasibility Report")
    lines.append("")
    lines.append("## 1. B1-0 数据导出与 No-GT Feature 审计")
    lines.append("")
    lines.append("- Feature 白名单只包含 prediction-only 字段：`{}`。".format("`, `".join(FEATURE_COLUMNS)))
    lines.append("- CSV 中的 `local_loss/collab_loss/loss_delta/label/ignore` 只用于 train/val 监督和审计，不进入 selector feature。")
    lines.append("- 未将 `target_visible`、GT visibility、annotation visibility、IoU、oracle mask 或 test GT 放入 feature 白名单。")
    lines.append("")

    lines.append("## 2. Label 分布")
    lines.append("")
    lines.append("| Split | Samples | Usable | Positive | Negative | Ignore | Positive ratio | Negative ratio | Ignore ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, stats in (("train", train_stats), ("val", val_stats)):
        lines.append("| {} | {} | {} | {} | {} | {} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
            split,
            stats["total"],
            stats["usable"],
            stats["positive"],
            stats["negative"],
            stats["ignore"],
            stats["positive_ratio"],
            stats["negative_ratio"],
            stats["ignore_ratio"],
        ))
    lines.append("")
    lines.append("| Split | loss_delta mean | std | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for split, stats in (("train", train_stats), ("val", val_stats)):
        lines.append("| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |".format(
            split,
            stats["loss_delta_mean"],
            stats["loss_delta_std"],
            stats["loss_delta_min"],
            stats["loss_delta_max"],
        ))
    lines.append("")

    lines.append("### 每序列 positive ratio Top/Bottom")
    lines.append("")
    for split, stats in (("train", train_stats), ("val", val_stats)):
        ratios = stats["sequence_positive_ratios"]
        top = ", ".join("{} {:.1f}%".format(seq, ratio) for seq, ratio, _ in ratios[:5])
        bottom = ", ".join("{} {:.1f}%".format(seq, ratio) for seq, ratio, _ in ratios[-5:])
        lines.append("- {} top: {}".format(split, top or "N/A"))
        lines.append("- {} bottom: {}".format(split, bottom or "N/A"))
    lines.append("")

    lines.append("## 3. Oracle Selector 上限（Validation Only）")
    lines.append("")
    lines.append("| Setting | AUC | Precision | Norm Precision | vs T1 | 正/负/无变化 | 负迁移率 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, summary, rows in (
        ("T1 local-only", oracle["local"], oracle["local_rows"]),
        ("A0 weighted raw", oracle["collab"], oracle["collab_rows"]),
        ("Oracle frame selector", oracle["oracle"], oracle["oracle_rows"]),
    ):
        delta = delta_stats(rows, oracle["local_rows"])
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} | {}/{}/{} | {:.2f}% |".format(
            name,
            summary["auc"],
            summary["precision"],
            summary["norm_precision"],
            format_delta(summary, oracle["local"]),
            delta["positive"],
            delta["negative"],
            delta["same"],
            delta["negative_rate"],
        ))
    lines.append("")
    lines.append("- Oracle 相对 A0 weighted raw 的提升：{:+.3f} / {:+.3f} / {:+.3f}。".format(
        oracle["oracle_delta_vs_collab"]["auc"],
        oracle["oracle_delta_vs_collab"]["precision"],
        oracle["oracle_delta_vs_collab"]["norm_precision"],
    ))
    if oracle["oracle_delta_vs_collab"]["auc"] < min_oracle_gain:
        lines.append("- **停止建议：oracle AUC 上限提升低于 {:.3f}，继续训练 selector 的收益空间不足。**".format(min_oracle_gain))
    else:
        lines.append("- Oracle 上限仍有可利用空间，可以继续 B1-1 offline selector。")
    lines.append("")

    lines.append("### Validation 分视角")
    lines.append("")
    lines.append("| Setting | Drone A | Drone B | Drone C |")
    lines.append("|---|---|---|---|")
    for key, label in (("local", "T1 local-only"), ("collab", "A0 weighted raw"), ("oracle", "Oracle frame selector")):
        views = oracle["views"][key]
        lines.append("| {} | {} | {} | {} |".format(
            label,
            format_metric(views["Drone A"]),
            format_metric(views["Drone B"]),
            format_metric(views["Drone C"]),
        ))
    lines.append("")

    lines.append("## 4. 当前状态")
    lines.append("")
    lines.append("B1-0 完成后需要运行 `tracking/train_pcum_selector_offline.py` 和 `tracking/eval_pcum_selector_threshold.py` 才能给出最终 selector feasibility 结论。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="output/pcum_v2_b1_selector/data/train_selector_samples.csv")
    parser.add_argument("--val-csv", default="output/pcum_v2_b1_selector/data/val_selector_samples.csv")
    parser.add_argument("--dataset", default="threemdot_val")
    parser.add_argument("--output", default="output/pcum_v2_b1_selector/b1_offline_selector_feasibility_report.md")
    parser.add_argument("--min-oracle-gain", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_feature_columns(FEATURE_COLUMNS)
    train_rows = read_selector_csv(args.train_csv)
    val_rows = read_selector_csv(args.val_csv)
    train_stats = label_stats(train_rows)
    val_stats = label_stats(val_rows)
    oracle = oracle_section(val_rows, args.dataset)
    write_report(args.output, train_stats, val_stats, oracle, args.min_oracle_gain)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
