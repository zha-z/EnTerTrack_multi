#!/usr/bin/env python3
"""Evaluate offline PCUM selector thresholds on validation only."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.evaluation.datasets import get_dataset
from tracking.analyze_pcum_selector_feasibility import label_stats, oracle_section
from tracking.pcum_selector_utils import (
    FEATURE_COLUMNS,
    delta_stats,
    evaluate_rows_as_predictions,
    format_delta,
    format_metric,
    format_top,
    load_bbox_file,
    non_ignore_rows,
    normalize_features,
    read_selector_csv,
    result_dir,
    rows_to_feature_matrix,
    seq_metrics_from_array,
    summarize_by_view,
    summarize_metric_rows,
    validate_feature_columns,
)
from tracking.train_pcum_selector_offline import TinySelector, labels, metric_bundle


def load_pickle_model(path):
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    return bundle["model"], bundle["norm_stats"]


def load_mlp(path):
    checkpoint = torch.load(str(path), map_location="cpu")
    model = TinySelector(len(checkpoint["feature_columns"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["norm_stats"]


def predict_prob(model_kind, model, stats, rows):
    feature_columns = stats.get("feature_columns")
    x_raw = rows_to_feature_matrix(rows, feature_columns)
    x = normalize_features(x_raw, stats)
    if model_kind in ("logreg", "random_forest", "gradient_boosting"):
        return model.predict_proba(x)[:, 1]
    if model_kind == "mlp":
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).cpu().numpy()
    raise ValueError("Unsupported model kind: {}".format(model_kind))


def load_result_metrics(root, dataset_name, tracker, runid):
    dataset = get_dataset(dataset_name)
    directory = result_dir(root, tracker, runid)
    if not directory.is_dir():
        return None
    rows = []
    for seq in dataset:
        path = directory / "{}.txt".format(seq.name)
        if not path.is_file():
            return None
        pred = load_bbox_file(path)
        auc, precision, norm_precision = seq_metrics_from_array(pred, seq)
        rows.append({
            "sequence": seq.name,
            "view": seq.name.rsplit("-", 1)[-1],
            "auc": auc,
            "precision": precision,
            "norm_precision": norm_precision,
        })
    for row in rows:
        row["view"] = {"1": "Drone A", "2": "Drone B", "3": "Drone C"}.get(row["view"], "Unknown")
    return {"rows": rows, "summary": summarize_metric_rows(rows), "views": summarize_by_view(rows)}


def add_probs(rows, probs):
    out = []
    for row, prob in zip(rows, probs):
        copied = dict(row)
        copied["selector_prob"] = "{:.10f}".format(float(prob))
        out.append(copied)
    return out


def threshold_results(rows_with_prob, dataset_name, thresholds):
    out = {}
    local_rows = evaluate_rows_as_predictions(rows_with_prob, dataset_name, "local")
    collab_rows = evaluate_rows_as_predictions(rows_with_prob, dataset_name, "collab")
    local_summary = summarize_metric_rows(local_rows)
    collab_summary = summarize_metric_rows(collab_rows)
    raw_delta = delta_stats(collab_rows, local_rows)
    for threshold in thresholds:
        selected_rows = evaluate_rows_as_predictions(rows_with_prob, dataset_name, "prob:{:.3f}".format(threshold))
        summary = summarize_metric_rows(selected_rows)
        delta = delta_stats(selected_rows, local_rows)
        collab_rate = np.mean([float(row["selector_prob"]) > threshold for row in rows_with_prob]) * 100.0
        out[threshold] = {
            "rows": selected_rows,
            "summary": summary,
            "delta": delta,
            "collab_rate": float(collab_rate),
            "local_rate": float(100.0 - collab_rate),
            "views": summarize_by_view(selected_rows),
        }
    return local_rows, collab_rows, local_summary, collab_summary, raw_delta, out


def choose_threshold(threshold_map, raw_summary, raw_delta, zero_summary, delay_summary, raw_views):
    candidates = []
    for threshold, result in threshold_map.items():
        summary = result["summary"]
        delta = result["delta"]
        not_down_views = 0
        for view in ("Drone A", "Drone B", "Drone C"):
            if result["views"][view]["auc"] >= raw_views[view]["auc"] - 1e-9:
                not_down_views += 1
        ok = (
            summary["auc"] >= raw_summary["auc"] - 1e-9
            and delta["negative_rate"] < raw_delta["negative_rate"]
            and (zero_summary is None or summary["auc"] > zero_summary["auc"])
            and (delay_summary is None or summary["auc"] > delay_summary["auc"])
            and 10.0 <= result["collab_rate"] <= 90.0
            and not_down_views >= 2
        )
        candidates.append((ok, summary["auc"], -delta["negative_rate"], threshold, not_down_views))
    valid = [item for item in candidates if item[0]]
    if not valid:
        return None
    valid.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return valid[0][3]


def fmt_summary_delta(summary, base):
    return "{:+.3f} / {:+.3f} / {:+.3f}".format(
        summary["auc"] - base["auc"],
        summary["precision"] - base["precision"],
        summary["norm_precision"] - base["norm_precision"],
    )


def write_report(path, train_rows, val_rows, oracle, model_results, local_summary,
                 raw_summary, raw_delta, zero, delay, none):
    train_stats = label_stats(train_rows)
    val_stats = label_stats(val_rows)
    lines = []
    lines.append("# PCUM-v2 B1 Offline Selector Feasibility Report")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    selected_any = any(item["selected_threshold"] is not None for item in model_results.values())
    if not selected_any:
        lines.append("- Validation threshold sweep 未通过预设选择规则，停止 B1，不进入 tracker 集成或正式 test。")
    else:
        selected_desc = ", ".join(
            "{} threshold {:.2f}".format(name, item["selected_threshold"])
            for name, item in model_results.items()
            if item["selected_threshold"] is not None
        )
        lines.append("- Validation 有候选通过：{}；仍需用户确认后才能进入 tracker 集成或 test。".format(selected_desc))
    lines.append("- Feature 仅使用 prediction-only 白名单；GT 只用于 train/val label 与 oracle 上限。")
    lines.append("")

    lines.append("## 2. 数据导出与 Label 分布")
    lines.append("")
    lines.append("| Split | Samples | Usable | Positive | Negative | Ignore | Positive ratio | Negative ratio | Ignore ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, stats in (("train", train_stats), ("val", val_stats)):
        lines.append("| {} | {} | {} | {} | {} | {} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
            split, stats["total"], stats["usable"], stats["positive"], stats["negative"], stats["ignore"],
            stats["positive_ratio"], stats["negative_ratio"], stats["ignore_ratio"]
        ))
    lines.append("")

    lines.append("## 3. Oracle Upper Bound（Validation Only）")
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
            name, summary["auc"], summary["precision"], summary["norm_precision"], fmt_summary_delta(summary, oracle["local"]),
            delta["positive"], delta["negative"], delta["same"], delta["negative_rate"]
        ))
    lines.append("")

    lines.append("## 4. Classifier 可分性")
    lines.append("")
    lines.append("| Model | ROC-AUC | PR-AUC | Acc. | Precision | Recall | F1 | Prob mean/std |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, item in model_results.items():
        metrics = item["metrics"]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}/{:.4f} |".format(
            name, metrics["roc_auc"], metrics["pr_auc"], metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], metrics["prob_mean"], metrics["prob_std"]
        ))
    lines.append("")

    lines.append("## 5. Threshold Sweep（Validation Only）")
    lines.append("")
    for model_name, item in model_results.items():
        lines.append("### {}".format(model_name))
        lines.append("")
        lines.append("| Threshold | AUC | Precision | Norm Precision | vs T1 | vs A0 raw | vs zero | vs delay | vs none | 正/负/无变化 | 负迁移率 | Collab/Local |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for threshold, result in sorted(item["threshold_map"].items()):
            summary = result["summary"]
            delta = result["delta"]
            zero_delta = fmt_summary_delta(summary, zero["summary"]) if zero else "N/A"
            delay_delta = fmt_summary_delta(summary, delay["summary"]) if delay else "N/A"
            none_delta = fmt_summary_delta(summary, none["summary"]) if none else "N/A"
            lines.append("| {:.2f} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {} | {} | {} | {}/{}/{} | {:.2f}% | {:.2f}%/{:.2f}% |".format(
                threshold, summary["auc"], summary["precision"], summary["norm_precision"],
                fmt_summary_delta(summary, local_summary), fmt_summary_delta(summary, raw_summary),
                zero_delta, delay_delta, none_delta, delta["positive"], delta["negative"], delta["same"],
                delta["negative_rate"], result["collab_rate"], result["local_rate"]
            ))
        lines.append("")
    lines.append("")

    lines.append("## 6. Validation 分视角 AUC")
    lines.append("")
    lines.append("| Threshold | Drone A | Drone B | Drone C |")
    lines.append("|---:|---:|---:|---:|")
    for model_name, item in model_results.items():
        for threshold, result in sorted(item["threshold_map"].items()):
            views = result["views"]
            lines.append("| {} {:.2f} | {:.3f} | {:.3f} | {:.3f} |".format(
                model_name, threshold, views["Drone A"]["auc"], views["Drone B"]["auc"], views["Drone C"]["auc"]
            ))
    lines.append("")

    lines.append("## 7. Top Positive / Negative")
    lines.append("")
    lines.append("| Threshold | Top positive | Top negative |")
    lines.append("|---:|---|---|")
    for model_name, item in model_results.items():
        for threshold, result in sorted(item["threshold_map"].items()):
            lines.append("| {} {:.2f} | {} | {} |".format(
                model_name,
                threshold,
                format_top(result["delta"]["top_positive"]),
                format_top(result["delta"]["top_negative"]),
            ))
    lines.append("")

    lines.append("## 8. 判断")
    lines.append("")
    if not selected_any:
        lines.append("- 没有 threshold 同时满足：AUC 不低于 A0 raw、负迁移率更低、超过 zero/delay、collab 使用率在 10%-90%、至少两个视角不低于 A0 raw。")
        lines.append("- 按停止条件，不跑 `threemdot_test`，不进入正式结果。")
    else:
        for model_name, item in model_results.items():
            if item["selected_threshold"] is None:
                lines.append("- {} 未通过 validation 选择规则。".format(model_name))
            else:
                lines.append("- {} 通过 validation 选择规则，threshold={:.2f}；正式 test 需要用户另行确认。".format(
                    model_name, item["selected_threshold"]
                ))
    lines.append("")
    lines.append("本阶段未训练 PCUM、未修改 PCUM 模型结构、未运行 threemdot_test。")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--val-csv", default="output/pcum_v2_b1_selector/data/val_selector_samples.csv")
    parser.add_argument("--train-csv", default="output/pcum_v2_b1_selector/data/train_selector_samples.csv")
    parser.add_argument("--dataset", default="threemdot_val")
    parser.add_argument("--model", choices=("logreg", "random_forest", "gradient_boosting", "mlp", "all"), default="all")
    parser.add_argument("--logreg", default="output/pcum_v2_b1_selector/checkpoints/logreg_selector.pkl")
    parser.add_argument("--random-forest", default="output/pcum_v2_b1_selector/checkpoints/random_forest_selector.pkl")
    parser.add_argument("--gradient-boosting", default="output/pcum_v2_b1_selector/checkpoints/gradient_boosting_selector.pkl")
    parser.add_argument("--mlp", default="output/pcum_v2_b1_selector/checkpoints/mlp_selector.pth")
    parser.add_argument("--zero-tracker", default="pcum_v2_a0_softmax_t010_ep0015_zero")
    parser.add_argument("--zero-runid", type=int, default=18301)
    parser.add_argument("--delay-tracker", default="pcum_v2_a0_softmax_t010_ep0015_delay")
    parser.add_argument("--delay-runid", type=int, default=18302)
    parser.add_argument("--none-tracker", default="")
    parser.add_argument("--none-runid", type=int, default=0)
    parser.add_argument("--output", default="output/pcum_v2_b1_selector/b1_offline_selector_training_and_threshold_report.md")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_feature_columns(FEATURE_COLUMNS)
    val_rows = read_selector_csv(args.val_csv)
    train_rows = read_selector_csv(args.train_csv)
    usable = non_ignore_rows(val_rows)
    y_val = labels(usable)
    thresholds = [i / 10.0 for i in range(1, 10)]
    model_specs = []
    if args.model in ("logreg", "all"):
        model_specs.append(("logreg", args.logreg))
    if args.model in ("random_forest", "all"):
        model_specs.append(("random_forest", args.random_forest))
    if args.model in ("gradient_boosting", "all"):
        model_specs.append(("gradient_boosting", args.gradient_boosting))
    if args.model in ("mlp", "all"):
        model_specs.append(("mlp", args.mlp))

    model_results = {}
    local_rows = raw_rows = local_summary = raw_summary = raw_delta = None
    raw_views = None
    for model_name, model_path in model_specs:
        model, stats = load_mlp(model_path) if model_name == "mlp" else load_pickle_model(model_path)
        probs = predict_prob(model_name, model, stats, val_rows)
        rows_with_prob = add_probs(val_rows, probs)
        usable_probs = predict_prob(model_name, model, stats, usable)
        local_rows, raw_rows, local_summary, raw_summary, raw_delta, threshold_map = threshold_results(
            rows_with_prob, args.dataset, thresholds
        )
        if raw_views is None:
            raw_views = summarize_by_view(raw_rows)
        model_results[model_name] = {
            "metrics": metric_bundle(y_val, usable_probs),
            "threshold_map": threshold_map,
            "selected_threshold": None,
        }
    raw_views = summarize_by_view(raw_rows)
    zero = load_result_metrics(args.root, args.dataset, args.zero_tracker, args.zero_runid)
    delay = load_result_metrics(args.root, args.dataset, args.delay_tracker, args.delay_runid)
    none = None
    if args.none_tracker and args.none_runid:
        none = load_result_metrics(args.root, args.dataset, args.none_tracker, args.none_runid)
    for model_name, item in model_results.items():
        item["selected_threshold"] = choose_threshold(
            item["threshold_map"],
            raw_summary,
            raw_delta,
            zero["summary"] if zero else None,
            delay["summary"] if delay else None,
            raw_views,
        )
    oracle = oracle_section(val_rows, args.dataset)
    write_report(args.output, train_rows, val_rows, oracle, model_results, local_summary,
                 raw_summary, raw_delta, zero, delay, none)
    print("Wrote {}".format(args.output))
    if not any(item["selected_threshold"] is not None for item in model_results.values()):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
