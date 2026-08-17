#!/usr/bin/env python3
"""Analyze PCUM-v2 B0 deterministic selector validation and test runs."""

import argparse
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

MARGINS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20)
VALIDATION_RUNS = {
    margin: ("pcum_v2_b0_selector_{}_ep15_val".format("m{:03d}".format(int(round(margin * 100)))), 18600 + int(round(margin * 100)))
    for margin in MARGINS
}
BASELINE_VAL_RUNS = {
    "T1 local-only": ("pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t1_local_only", 16151),
    "A0 weighted raw": ("pcum_v2_a0_softmax_t010_ep0015", 18101),
    "A0 weighted zero": ("pcum_v2_a0_softmax_t010_ep0015_zero", 18301),
    "A0 weighted delay": ("pcum_v2_a0_softmax_t010_ep0015_delay", 18302),
}
BASELINE_TEST_RUNS = {
    "T1 local-only": ("pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t1_local_only", 12551),
    "A0 weighted raw": ("pcum_v2_a0_weighted_softmax_t010_ep15_t2_raw_test", 18552),
    "A0 weighted zero": ("pcum_v2_a0_weighted_softmax_t010_ep15_t2_zero_test", 18554),
    "A0 weighted delay": ("pcum_v2_a0_weighted_softmax_t010_ep15_t2_delay_test", 18555),
}


def result_dir(root, tracker, runid):
    return Path(root) / "output" / "test" / "tracking_results" / "entertrack" / "{}_{:03d}".format(tracker, runid)


def view_name(seq_name):
    suffix = seq_name.rsplit("-", 1)[-1]
    return {"1": "Drone A", "2": "Drone B", "3": "Drone C"}.get(suffix, "Unknown")


def seq_metrics(pred_path, seq):
    pred_bb = torch.tensor(load_text(str(pred_path), delimiter=("\t", ","), dtype=np.float64))
    if pred_bb.ndim == 1:
        pred_bb = pred_bb.reshape(1, -1)
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


def evaluate_run(root, dataset, tracker, runid, required=True):
    directory = result_dir(root, tracker, runid)
    if not directory.is_dir():
        if required:
            raise FileNotFoundError(directory)
        return None
    rows = []
    for seq in dataset:
        pred_path = directory / "{}.txt".format(seq.name)
        if not pred_path.is_file():
            if required:
                raise FileNotFoundError(pred_path)
            return None
        auc, precision, norm_precision = seq_metrics(pred_path, seq)
        rows.append({
            "sequence": seq.name,
            "view": view_name(seq.name),
            "auc": auc,
            "precision": precision,
            "norm_precision": norm_precision,
        })
    return rows


def summarize_rows(rows):
    return {
        "auc": float(np.mean([r["auc"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "norm_precision": float(np.mean([r["norm_precision"] for r in rows])),
    }


def summarize_by_view(rows):
    return {
        view: summarize_rows([row for row in rows if row["view"] == view])
        for view in ("Drone A", "Drone B", "Drone C")
    }


def delta_stats(rows, base_rows):
    base = {row["sequence"]: row for row in base_rows}
    deltas = [
        (row["sequence"], row["view"], row["auc"] - base[row["sequence"]]["auc"])
        for row in rows
    ]
    pos = sum(1 for _, _, delta in deltas if delta > 1e-6)
    neg = sum(1 for _, _, delta in deltas if delta < -1e-6)
    same = len(deltas) - pos - neg
    return {
        "positive": pos,
        "negative": neg,
        "same": same,
        "negative_rate": neg / len(deltas) * 100.0,
        "top_positive": sorted(deltas, key=lambda item: item[2], reverse=True)[:5],
        "top_negative": sorted(deltas, key=lambda item: item[2])[:5],
    }


def selector_stats(root, dataset, tracker, runid):
    directory = result_dir(root, tracker, runid)
    rows = []
    for seq in dataset:
        path = directory / "{}_pcum_selector.txt".format(seq.name)
        if not path.is_file():
            continue
        data = np.loadtxt(str(path), delimiter="\t")
        data = np.atleast_2d(data)
        data = data[(data[:, 0] > 0.5) & np.isfinite(data[:, 2])]
        if data.size:
            rows.append(data)
    if not rows:
        return None
    data = np.concatenate(rows, axis=0)
    use_collab = data[:, 1] > 0.5
    return {
        "frames": int(data.shape[0]),
        "collab_rate": float(np.mean(use_collab) * 100.0),
        "local_rate": float((1.0 - np.mean(use_collab)) * 100.0),
        "conf_delta_mean": float(np.mean(data[:, 4])),
        "conf_delta_min": float(np.min(data[:, 4])),
        "conf_delta_max": float(np.max(data[:, 4])),
        "motion_mean": float(np.mean(data[:, 5])),
        "motion_min": float(np.min(data[:, 5])),
        "motion_max": float(np.max(data[:, 5])),
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


def build_results(root, dataset_name, runs):
    dataset = get_dataset(dataset_name)
    results = {}
    for name, (tracker, runid) in runs.items():
        rows = evaluate_run(root, dataset, tracker, runid)
        results[name] = {
            "tracker": tracker,
            "runid": runid,
            "rows": rows,
            "summary": summarize_rows(rows),
            "views": summarize_by_view(rows),
            "selector": selector_stats(root, dataset, tracker, runid),
        }
    return dataset, results


def choose_margin(results):
    t1 = results["T1 local-only"]
    raw = results["A0 weighted raw"]
    zero = results["A0 weighted zero"]
    delay = results["A0 weighted delay"]
    raw_delta = delta_stats(raw["rows"], t1["rows"])
    candidates = []
    for margin in MARGINS:
        name = "B0 margin {:.2f}".format(margin)
        result = results[name]
        summary = result["summary"]
        delta = delta_stats(result["rows"], t1["rows"])
        views = result["views"]
        raw_views = raw["views"]
        not_down_views = sum(
            1 for view in ("Drone A", "Drone B", "Drone C")
            if views[view]["auc"] >= raw_views[view]["auc"] - 1e-9
        )
        ok = (
            summary["auc"] >= raw["summary"]["auc"] - 1e-9
            and delta["negative_rate"] < raw_delta["negative_rate"]
            and summary["auc"] > zero["summary"]["auc"]
            and summary["auc"] > delay["summary"]["auc"]
            and not_down_views >= 2
        )
        candidates.append((ok, summary["auc"], -delta["negative_rate"], margin, not_down_views))
    valid = [item for item in candidates if item[0]]
    if not valid:
        return None
    valid.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return valid[0][3]


def write_report(path, val_results, test_results, selected_margin, compatibility_ok):
    lines = []
    lines.append("# PCUM-v2 B0 Deterministic Reliability Selector Report")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append("- Selector compatibility: {}。".format(
        "通过，RELIABILITY_SELECTOR=none 与 A0 weighted raw runid 18552 完全一致"
        if compatibility_ok else "未通过或未验证"
    ))
    if selected_margin is None:
        lines.append("- Validation sweep 没有找到同时满足 AUC、负迁移率、zero/delay 和分视角约束的 margin。")
        lines.append("- 因此不建议把 B0 selector 作为 PCUM-v2B 主结果；保留 PCUM-v2A A0 weighted raw 作为当前主结果。")
    else:
        lines.append("- Validation 选择 margin **{:.2f}**。".format(selected_margin))
        if test_results is None:
            lines.append("- 选中 margin 的正式 test 尚未检测到。")
        else:
            b0 = test_results["B0 selected"]["summary"]
            a0 = test_results["A0 weighted raw"]["summary"]
            lines.append("- Test B0 selected = **{}**，相对 A0 weighted raw 为 **{}**。".format(
                fmt_metric(b0), fmt_delta(b0, a0)
            ))
    lines.append("")

    def add_table(title, results):
        t1 = results["T1 local-only"]["summary"]
        raw = results["A0 weighted raw"]["summary"]
        lines.append("## {}".format(title))
        lines.append("")
        lines.append("| Setting | AUC | Precision | Norm Precision | vs T1 | vs A0 raw | 正/负/无变化 | 负迁移率 | Selector collab/local |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, result in results.items():
            summary = result["summary"]
            delta = delta_stats(result["rows"], results["T1 local-only"]["rows"])
            selector = result.get("selector")
            selector_text = "-"
            if selector is not None:
                selector_text = "{:.2f}% / {:.2f}%".format(
                    selector["collab_rate"], selector["local_rate"]
                )
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {}/{}/{} | {:.2f}% | {} |".format(
                name,
                summary["auc"],
                summary["precision"],
                summary["norm_precision"],
                fmt_delta(summary, t1),
                fmt_delta(summary, raw),
                delta["positive"],
                delta["negative"],
                delta["same"],
                delta["negative_rate"],
                selector_text,
            ))
        lines.append("")

    add_table("2. Validation 结果", val_results)

    lines.append("## 3. Validation 分视角 AUC")
    lines.append("")
    lines.append("| Setting | Drone A | Drone B | Drone C |")
    lines.append("|---|---:|---:|---:|")
    for name, result in val_results.items():
        views = result["views"]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            name,
            views["Drone A"]["auc"],
            views["Drone B"]["auc"],
            views["Drone C"]["auc"],
        ))
    lines.append("")

    lines.append("## 4. Validation Top Positive / Negative")
    lines.append("")
    lines.append("| Setting | Top positive | Top negative |")
    lines.append("|---|---|---|")
    for name, result in val_results.items():
        if name in ("T1 local-only",):
            continue
        delta = delta_stats(result["rows"], val_results["T1 local-only"]["rows"])
        lines.append("| {} | {} | {} |".format(
            name,
            fmt_top(delta["top_positive"]),
            fmt_top(delta["top_negative"]),
        ))
    lines.append("")

    lines.append("## 5. Selector 诊断")
    lines.append("")
    lines.append("| Setting | frames | collab rate | fallback local | conf delta mean/min/max | motion mean/min/max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, result in val_results.items():
        selector = result.get("selector")
        if selector is None:
            continue
        lines.append("| {} | {} | {:.2f}% | {:.2f}% | {:.6f}/{:.6f}/{:.6f} | {:.6f}/{:.6f}/{:.6f} |".format(
            name,
            selector["frames"],
            selector["collab_rate"],
            selector["local_rate"],
            selector["conf_delta_mean"],
            selector["conf_delta_min"],
            selector["conf_delta_max"],
            selector["motion_mean"],
            selector["motion_min"],
            selector["motion_max"],
        ))
    lines.append("")

    if test_results is not None:
        add_table("6. Selected Margin Test 结果", test_results)
        lines.append("## 7. Test 分视角 AUC")
        lines.append("")
        lines.append("| Setting | Drone A | Drone B | Drone C |")
        lines.append("|---|---:|---:|---:|")
        for name, result in test_results.items():
            views = result["views"]
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(
                name,
                views["Drone A"]["auc"],
                views["Drone B"]["auc"],
                views["Drone C"]["auc"],
            ))
        lines.append("")

    lines.append("## 8. 判断")
    lines.append("")
    if selected_margin is None:
        lines.append("- B0 selector 未通过预设 validation 选择规则。")
        margin_zero = val_results["B0 margin 0.00"]
        margin_zero_delta = delta_stats(
            margin_zero["rows"],
            val_results["T1 local-only"]["rows"],
        )
        raw_delta = delta_stats(
            val_results["A0 weighted raw"]["rows"],
            val_results["T1 local-only"]["rows"],
        )
        lines.append(
            "- margin=0.00 仅在 {:.2f}% 帧采用 collaborative，AUC 比 A0 raw 低 {:.3f}，负迁移率由 {:.2f}% 升至 {:.2f}%。".format(
                margin_zero["selector"]["collab_rate"],
                val_results["A0 weighted raw"]["summary"]["auc"] - margin_zero["summary"]["auc"],
                raw_delta["negative_rate"],
                margin_zero_delta["negative_rate"],
            )
        )
        lines.append(
            "- margin>=0.02 时 collaborative 使用率为 0%，全部回退 local，属于明显过度回退；当前 confidence delta 量级约为 1e-4，与 0.02-0.20 的 margin 不匹配。"
        )
        lines.append("- 不进入正式 test 或论文主结果；当前保留 PCUM-v2A A0 weighted raw。")
    else:
        lines.append("- B0 selector 是否作为 PCUM-v2B 主结果，应以本报告的 selected test 是否同时提升 AUC 并降低负迁移为准。")
        lines.append("- 若 test 未提升或负迁移未降低，保留 PCUM-v2A A0 weighted raw。")
    lines.append("")
    lines.append("本报告只分析测试结果；未训练、未修改模型结构、未提交 Git。")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--output", default="output/pcum_v2_b_selector/b0_selector_validation_and_test_report.md")
    parser.add_argument("--compatibility-ok", action="store_true")
    parser.add_argument("--selected-margin", type=float)
    args = parser.parse_args()

    val_runs = dict(BASELINE_VAL_RUNS)
    for margin, run in VALIDATION_RUNS.items():
        val_runs["B0 margin {:.2f}".format(margin)] = run
    _, val_results = build_results(args.root, "threemdot_val", val_runs)
    selected_margin = args.selected_margin
    if selected_margin is None:
        selected_margin = choose_margin(val_results)

    test_results = None
    if selected_margin is not None:
        tag = "m{:03d}".format(int(round(selected_margin * 100)))
        test_runs = dict(BASELINE_TEST_RUNS)
        test_runs["B0 selected"] = (
            "pcum_v2_b0_selector_{}_ep15_test".format(tag),
            18652,
        )
        dataset = get_dataset("threemdot_test")
        b0_dir = result_dir(args.root, test_runs["B0 selected"][0], test_runs["B0 selected"][1])
        if b0_dir.is_dir():
            test_results = {}
            for name, (tracker, runid) in test_runs.items():
                rows = evaluate_run(args.root, dataset, tracker, runid)
                test_results[name] = {
                    "tracker": tracker,
                    "runid": runid,
                    "rows": rows,
                    "summary": summarize_rows(rows),
                    "views": summarize_by_view(rows),
                    "selector": selector_stats(args.root, dataset, tracker, runid),
                }

    write_report(args.output, val_results, test_results, selected_margin, args.compatibility_ok)
    print("Wrote {}".format(args.output))
    if selected_margin is None:
        print("SELECTED_MARGIN=none")
    else:
        print("SELECTED_MARGIN={:.2f}".format(selected_margin))


if __name__ == "__main__":
    main()
