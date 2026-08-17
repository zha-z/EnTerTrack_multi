#!/usr/bin/env python3
"""Analyze PCUM-v2 A0 weighted remote aggregation validation runs."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation.datasets import get_dataset
from lib.test.utils.load_text import load_text


THRESHOLD_OVERLAP = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64)
THRESHOLD_CENTER = torch.arange(0, 51, dtype=torch.float64)
THRESHOLD_CENTER_NORM = torch.arange(0, 51, dtype=torch.float64) / 100.0


RUNS = {
    "T1 local-only": ("pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t1_local_only", 16151),
    "T2 mean": ("pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_raw", 16152),
    "softmax t=0.10": ("pcum_v2_a0_softmax_t010_ep0015", 18101),
    "softmax t=0.25": ("pcum_v2_a0_softmax_t025_ep0015", 18125),
    "softmax t=0.50": ("pcum_v2_a0_softmax_t050_ep0015", 18150),
    "softmax t=1.00": ("pcum_v2_a0_softmax_t100_ep0015", 18200),
    "sigmoid t=0.25": ("pcum_v2_a0_sigmoid_t025_ep0015", 18225),
}

OPTIONAL_RUNS = {
    "T2 zero": None,
    "T2 delay": None,
}


def result_dir(root, tracker, runid):
    return Path(root) / "output" / "test" / "tracking_results" / "entertrack" / "{}_{:03d}".format(tracker, runid)


def view_name(seq_name):
    suffix = seq_name.rsplit("-", 1)[-1]
    return {"1": "Drone A", "2": "Drone B", "3": "Drone C"}.get(suffix, "Unknown")


def seq_metrics(pred_path, seq):
    pred_bb = torch.tensor(load_text(str(pred_path), delimiter=("\t", ","), dtype=np.float64))
    anno_bb = torch.tensor(seq.ground_truth_rect)
    target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if getattr(seq, "target_visible", None) is not None else None
    err_overlap, err_center, err_center_normalized, _ = calc_seq_err_robust(
        pred_bb, anno_bb, seq.dataset, target_visible
    )
    seq_length = anno_bb.shape[0]
    auc = (err_overlap.view(-1, 1) > THRESHOLD_OVERLAP.view(1, -1)).sum(0).float().mean().item() / seq_length * 100.0
    precision = (err_center.view(-1, 1) <= THRESHOLD_CENTER.view(1, -1)).sum(0).float()[20].item() / seq_length * 100.0
    norm_precision = (
        (err_center_normalized.view(-1, 1) <= THRESHOLD_CENTER_NORM.view(1, -1)).sum(0).float()[20].item()
        / seq_length
        * 100.0
    )
    return auc, precision, norm_precision


def evaluate_run(root, dataset, tracker, runid):
    directory = result_dir(root, tracker, runid)
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    rows = []
    for seq in dataset:
        pred_path = directory / "{}.txt".format(seq.name)
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        auc, precision, norm_precision = seq_metrics(pred_path, seq)
        rows.append(
            {
                "sequence": seq.name,
                "view": view_name(seq.name),
                "auc": auc,
                "precision": precision,
                "norm_precision": norm_precision,
            }
        )
    return rows


def summarize_rows(rows):
    return {
        "auc": float(np.mean([r["auc"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "norm_precision": float(np.mean([r["norm_precision"] for r in rows])),
    }


def summarize_by_view(rows):
    out = {}
    for view in ("Drone A", "Drone B", "Drone C"):
        view_rows = [r for r in rows if r["view"] == view]
        out[view] = summarize_rows(view_rows)
    return out


def delta_stats(rows, base_rows):
    base = {r["sequence"]: r for r in base_rows}
    deltas = []
    for row in rows:
        b = base[row["sequence"]]
        deltas.append((row["sequence"], row["view"], row["auc"] - b["auc"]))
    pos = sum(1 for _, _, d in deltas if d > 1e-6)
    neg = sum(1 for _, _, d in deltas if d < -1e-6)
    same = len(deltas) - pos - neg
    return {
        "positive": pos,
        "negative": neg,
        "same": same,
        "negative_rate": neg / len(deltas) * 100.0,
        "top_positive": sorted(deltas, key=lambda x: x[2], reverse=True)[:5],
        "top_negative": sorted(deltas, key=lambda x: x[2])[:5],
    }


def weight_stats(root, tracker, runid, dataset):
    directory = result_dir(root, tracker, runid)
    rows = []
    for seq in dataset:
        path = directory / "{}_pcum_remote_weights.txt".format(seq.name)
        if not path.is_file():
            continue
        data = np.loadtxt(str(path), delimiter="\t")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = data[np.isfinite(data[:, 0])]
        if data.size == 0:
            continue
        rows.append(data)
    if not rows:
        return None
    data = np.concatenate(rows, axis=0)
    selected = data[:, 3].astype(int)
    total = selected.size
    selected_ratio = {}
    for idx, name in ((0, "A"), (1, "B"), (2, "C")):
        selected_ratio[name] = float(np.mean(selected == idx) * 100.0)
    return {
        "entropy_mean": float(np.mean(data[:, 0])),
        "entropy_std": float(np.std(data[:, 0])),
        "max_weight_mean": float(np.mean(data[:, 1])),
        "max_weight_p90": float(np.percentile(data[:, 1], 90)),
        "weight_mean": float(np.mean(data[:, 2])),
        "valid_count_mean": float(np.mean(data[:, 4])),
        "quality_mean": float(np.mean(data[:, 5])),
        "quality_min": float(np.min(data[:, 6])),
        "quality_max": float(np.max(data[:, 7])),
        "fallback_rate": float(np.mean(data[:, 8] > 0.5) * 100.0),
        "selected_ratio": selected_ratio,
        "frames": int(total),
    }


def fmt_metric(summary):
    return "{auc:.3f} / {precision:.3f} / {norm_precision:.3f}".format(**summary)


def fmt_delta(summary, base):
    return "{:+.3f} / {:+.3f} / {:+.3f}".format(
        summary["auc"] - base["auc"],
        summary["precision"] - base["precision"],
        summary["norm_precision"] - base["norm_precision"],
    )


def fmt_top(items):
    return ", ".join("{} {:+.3f}".format(seq, delta) for seq, _, delta in items)


def write_report(path, results, weights, best_name, compatibility_ok):
    t1 = results["T1 local-only"]["summary"]
    mean = results["T2 mean"]["summary"]
    zero = results.get("T2 zero", {}).get("summary")
    delay = results.get("T2 delay", {}).get("summary")

    lines = []
    lines.append("# PCUM-v2 A0 Weighted Remote Aggregation Validation Report")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append("- Mean compatibility: {}。".format("通过，runid 18052 与 no-GT raw runid 12552 的 105 个 bbox 完全一致" if compatibility_ok else "未通过"))
    lines.append("- Raw weighted validation 中最佳设置为 **{}**。".format(best_name))
    best = results[best_name]["summary"]
    lines.append("- 最佳 validation AUC/Precision/Norm Precision 为 **{}**。".format(fmt_metric(best)))
    lines.append("- 相对 T1 的提升为 **{}**；相对 mean raw 的变化为 **{}**。".format(fmt_delta(best, t1), fmt_delta(best, mean)))
    if zero and delay:
        lines.append("- 与 zero/delay 对照：weighted-raw 相对 zero 为 **{}**，相对 delay 为 **{}**。".format(fmt_delta(best, zero), fmt_delta(best, delay)))
        if zero["auc"] > best["auc"]:
            lines.append("- 仍然存在 **zero > raw**，prompt 语义可靠性问题没有被 A0 解决。")
        else:
            lines.append("- 当前最佳 weighted raw 已不低于 zero，prompt 语义可靠性有改善迹象。")
    lines.append("")

    lines.append("## 2. 总体结果")
    lines.append("")
    lines.append("| Setting | AUC | Precision | Norm Precision | vs T1 | vs Mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in results:
        s = results[name]["summary"]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            name, s["auc"], s["precision"], s["norm_precision"], fmt_delta(s, t1), fmt_delta(s, mean)
        ))
    lines.append("")

    if zero and delay:
        lines.append("## 3. Zero / Delay 对照")
        lines.append("")
        lines.append("| Setting | vs Zero | vs Delay |")
        lines.append("|---|---:|---:|")
        for name in [n for n in results if n.startswith("softmax") or n.startswith("sigmoid")]:
            s = results[name]["summary"]
            lines.append("| {} | {} | {} |".format(name, fmt_delta(s, zero), fmt_delta(s, delay)))
        lines.append("")

    lines.append("## 4. A/B/C 分视角")
    lines.append("")
    lines.append("| Setting | Drone A | Drone B | Drone C |")
    lines.append("|---|---|---|---|")
    for name, item in results.items():
        view = item["view"]
        lines.append("| {} | {} | {} | {} |".format(
            name,
            fmt_metric(view["Drone A"]),
            fmt_metric(view["Drone B"]),
            fmt_metric(view["Drone C"]),
        ))
    lines.append("")

    lines.append("## 5. 负迁移率")
    lines.append("")
    lines.append("序列级 delta 定义为当前设置 AUC - T1 AUC。")
    lines.append("")
    lines.append("| Setting | 正/负/无变化 | 负迁移率 | Top positive | Top negative |")
    lines.append("|---|---:|---:|---|---|")
    for name, item in results.items():
        if name == "T1 local-only":
            continue
        ds = item["delta_vs_t1"]
        lines.append("| {} | {}/{}/{} | {:.2f}% | {} | {} |".format(
            name,
            ds["positive"],
            ds["negative"],
            ds["same"],
            ds["negative_rate"],
            fmt_top(ds["top_positive"][:3]),
            fmt_top(ds["top_negative"][:3]),
        ))
    lines.append("")

    lines.append("## 6. Remote Weight Diagnostics")
    lines.append("")
    lines.append("| Setting | entropy mean/std | max weight mean/p90 | valid count | quality mean/min/max | fallback | selected A/B/C |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, ws in weights.items():
        if ws is None:
            continue
        lines.append("| {} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:.2f} | {:.4f}/{:.4f}/{:.4f} | {:.2f}% | {:.1f}/{:.1f}/{:.1f}% |".format(
            name,
            ws["entropy_mean"],
            ws["entropy_std"],
            ws["max_weight_mean"],
            ws["max_weight_p90"],
            ws["valid_count_mean"],
            ws["quality_mean"],
            ws["quality_min"],
            ws["quality_max"],
            ws["fallback_rate"],
            ws["selected_ratio"]["A"],
            ws["selected_ratio"]["B"],
            ws["selected_ratio"]["C"],
        ))
    lines.append("")

    lines.append("## 7. 是否建议进入 5 epoch DDP 短训")
    lines.append("")
    best_delta_mean = best["auc"] - mean["auc"]
    best_delta_t1 = best["auc"] - t1["auc"]
    if best_delta_mean > 0 and best_delta_t1 > 0:
        lines.append("- 建议进入一次 5 epoch DDP 短训，使用当前最佳 weighted aggregation 设置。")
    elif best_delta_t1 > 0:
        lines.append("- 暂不建议直接进入 DDP 短训：weighted 仍高于 T1，但没有超过 mean raw。")
    else:
        lines.append("- 不建议进入 DDP 短训：validation 上 weighted raw 未带来正 remote 增益。")
    lines.append("- 本阶段只运行测试和分析，未训练、未提交 Git、未删除文件。")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="threemdot_val")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="output/pcum_v2_a_weighted/a0_weighted_aggregation_validation_report.md")
    parser.add_argument("--compatibility-ok", action="store_true")
    parser.add_argument("--zero-tracker", default=None)
    parser.add_argument("--zero-runid", type=int, default=None)
    parser.add_argument("--delay-tracker", default=None)
    parser.add_argument("--delay-runid", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    dataset = get_dataset(args.dataset)

    run_map = dict(RUNS)
    if args.zero_tracker and args.zero_runid is not None:
        run_map["T2 zero"] = (args.zero_tracker, args.zero_runid)
    if args.delay_tracker and args.delay_runid is not None:
        run_map["T2 delay"] = (args.delay_tracker, args.delay_runid)

    results = {}
    weights = {}
    for name, (tracker, runid) in run_map.items():
        rows = evaluate_run(root, dataset, tracker, runid)
        results[name] = {
            "summary": summarize_rows(rows),
            "view": summarize_by_view(rows),
            "rows": rows,
        }
        weights[name] = weight_stats(root, tracker, runid, dataset)

    t1_rows = results["T1 local-only"]["rows"]
    for item in results.values():
        item["delta_vs_t1"] = delta_stats(item["rows"], t1_rows)

    candidates = [name for name in results if name.startswith("softmax") or name.startswith("sigmoid")]
    best_name = max(candidates, key=lambda name: results[name]["summary"]["auc"])

    write_report(Path(args.output), results, weights, best_name, args.compatibility_ok)
    print("wrote {}".format(args.output))
    print("best={} metrics={}".format(best_name, fmt_metric(results[best_name]["summary"])))


if __name__ == "__main__":
    main()
