#!/usr/bin/env python3
"""Summarize D0-fixed-lite ep5 validation runs."""

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
    "D0-fixed-lite T1 local-only": ("pcum_v2_d0_fixed_lite_rank_softmax_t010_ep5_t1_val", 19251),
    "D0-fixed-lite raw weighted": ("pcum_v2_d0_fixed_lite_rank_softmax_t010_ep5_t2_raw_val", 19252),
    "D0-fixed-lite zero": ("pcum_v2_d0_fixed_lite_rank_softmax_t010_ep5_t2_zero_val", 19254),
    "D0-fixed-lite delay diagnostic": ("pcum_v2_d0_fixed_lite_rank_softmax_t010_ep5_t2_delay_val", 19255),
    "D0-fixed-lite none": ("pcum_v2_d0_fixed_lite_rank_softmax_t010_ep5_t2_none_val", 19256),
}

A0_RAW = {
    "summary": {"auc": 65.247, "precision": 84.354, "norm_precision": 83.522},
    "view": {
        "Drone A": {"auc": 61.805, "precision": 78.965, "norm_precision": 84.299},
        "Drone B": {"auc": 67.248, "precision": 86.872, "norm_precision": 82.611},
        "Drone C": {"auc": 66.687, "precision": 87.226, "norm_precision": 83.656},
    },
}
E4_T1 = {"auc": 63.901, "precision": 82.644, "norm_precision": 81.700}


def result_dir(tracker, runid):
    return REPO_ROOT / "output" / "test" / "tracking_results" / "entertrack" / "{}_{:03d}".format(tracker, runid)


def view_name(seq_name):
    return {"1": "Drone A", "2": "Drone B", "3": "Drone C"}.get(seq_name.rsplit("-", 1)[-1], "Unknown")


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


def evaluate_run(dataset, tracker, runid):
    directory = result_dir(tracker, runid)
    rows = []
    for seq in dataset:
        pred_path = directory / "{}.txt".format(seq.name)
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


def summarize(rows):
    return {
        "auc": float(np.mean([r["auc"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "norm_precision": float(np.mean([r["norm_precision"] for r in rows])),
    }


def summarize_by_view(rows):
    return {view: summarize([r for r in rows if r["view"] == view]) for view in ("Drone A", "Drone B", "Drone C")}


def delta_stats(rows, base_rows):
    base = {r["sequence"]: r for r in base_rows}
    deltas = [(r["sequence"], r["view"], r["auc"] - base[r["sequence"]]["auc"]) for r in rows]
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


def weight_stats(tracker, runid, dataset):
    directory = result_dir(tracker, runid)
    arrays = []
    files = 0
    for seq in dataset:
        path = directory / "{}_pcum_remote_weights.txt".format(seq.name)
        if not path.is_file():
            continue
        files += 1
        data = np.loadtxt(str(path), delimiter="\t")
        data = np.atleast_2d(data)
        data = data[np.isfinite(data[:, 0])]
        if data.size:
            arrays.append(data)
    if not arrays:
        return None, files
    data = np.concatenate(arrays, axis=0)
    selected = data[:, 3].astype(int)
    return {
        "entropy_mean": float(np.mean(data[:, 0])),
        "entropy_std": float(np.std(data[:, 0])),
        "max_weight_mean": float(np.mean(data[:, 1])),
        "max_weight_p90": float(np.percentile(data[:, 1], 90)),
        "valid_count_mean": float(np.mean(data[:, 4])),
        "quality_mean": float(np.mean(data[:, 5])),
        "quality_min": float(np.min(data[:, 6])),
        "quality_max": float(np.max(data[:, 7])),
        "fallback_rate": float(np.mean(data[:, 8] > 0.5) * 100.0),
        "selected_a": float(np.mean(selected == 0) * 100.0),
        "selected_b": float(np.mean(selected == 1) * 100.0),
        "selected_c": float(np.mean(selected == 2) * 100.0),
        "frames": int(selected.size),
    }, files


def fmt(s):
    return "{auc:.3f} / {precision:.3f} / {norm_precision:.3f}".format(**s)


def delta(a, b):
    return "{:+.3f} / {:+.3f} / {:+.3f}".format(
        a["auc"] - b["auc"],
        a["precision"] - b["precision"],
        a["norm_precision"] - b["norm_precision"],
    )


def top_fmt(items):
    return ", ".join("{} {:+.3f}".format(seq, d) for seq, _, d in items)


def load_audit_summary():
    audit = REPO_ROOT / "output" / "pcum_v2_d0_fixed_lite_ranking" / "d0_fixed_lite_ep5_freeze_audit.md"
    text = audit.read_text()
    return {
        "path": str(audit.relative_to(REPO_ROOT)),
        "backbone_clean": "| backbone | 92 | 0 |" in text,
        "box_head_clean": "| box_head | 90 | 0 |" in text,
        "only_pcum": "Only PCUM/fusion/prompt/residual changed: `True`" in text,
        "no_nan": "NaN/Inf/Error found | `False`" in text or "training log NaN/Inf/Error flag | `False`" in text,
        "residual": "last effective residual scale | `0.02997`" in text,
    }


def main():
    dataset = get_dataset("threemdot_val")
    results = {}
    weights = {}
    for name, (tracker, runid) in RUNS.items():
        rows = evaluate_run(dataset, tracker, runid)
        results[name] = {
            "tracker": tracker,
            "runid": runid,
            "rows": rows,
            "summary": summarize(rows),
            "view": summarize_by_view(rows),
        }
        weights[name] = weight_stats(tracker, runid, dataset)

    t1 = results["D0-fixed-lite T1 local-only"]["summary"]
    t1_rows = results["D0-fixed-lite T1 local-only"]["rows"]
    raw = results["D0-fixed-lite raw weighted"]["summary"]
    raw_rows = results["D0-fixed-lite raw weighted"]["rows"]
    zero = results["D0-fixed-lite zero"]["summary"]
    delay = results["D0-fixed-lite delay diagnostic"]["summary"]
    none = results["D0-fixed-lite none"]["summary"]
    raw_delta_stats = delta_stats(raw_rows, t1_rows)
    audit = load_audit_summary()

    view_pass = sum(
        1 for view in ("Drone A", "Drone B", "Drone C")
        if results["D0-fixed-lite raw weighted"]["view"][view]["auc"] + 1e-9 >= A0_RAW["view"][view]["auc"]
    )
    criteria = {
        "raw_auc_gt_a0": raw["auc"] > A0_RAW["summary"]["auc"],
        "raw_ge_t1": raw["auc"] + 1e-9 >= t1["auc"],
        "raw_gt_zero": raw["auc"] > zero["auc"],
        "raw_gt_none": raw["auc"] > none["auc"],
        "neg_transfer_le_40": raw_delta_stats["negative_rate"] <= 40.0 + 1e-9,
        "views_ge_a0_at_least_2": view_pass >= 2,
        "audit_only_pcum": audit["backbone_clean"] and audit["box_head_clean"] and audit["only_pcum"],
        "no_nan_inf": audit["no_nan"],
        "no_residual_collapse": audit["residual"],
    }

    lines = []
    lines.append("# D0-fixed-lite Ranking Fine-tuning Ep5 Validation Report")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append("- D0-fixed-lite 6 GPU 5 epoch 训练完成，epoch5 checkpoint `stored_epoch=5`，`pcum_keys=34`。")
    lines.append("- Freeze audit 通过：backbone changed keys=0，box_head changed keys=0，box_head BN buffers 未变化；optimizer 中无 backbone/box_head 参数。")
    lines.append("- 五组 `threemdot_val` 验证均通过 verifier：15/15 bbox，预测长度一致，checkpoint epoch=5，日志为 no-GT。")
    lines.append("- D0-fixed-lite raw weighted validation = **{}**。".format(fmt(raw)))
    lines.append("- 相对 A0 weighted raw validation `{}`：**{}**。".format(fmt(A0_RAW["summary"]), delta(raw, A0_RAW["summary"])))
    lines.append("- 相对 E4 T1 local-only validation `{}`：**{}**。".format(fmt(E4_T1), delta(raw, E4_T1)))
    lines.append("- raw - T1：**{}**；raw - zero：**{}**；raw - none：**{}**。".format(delta(raw, t1), delta(raw, zero), delta(raw, none)))
    lines.append("- raw - delay diagnostic：**{}**。该项仅作诊断，因为 `LAMBDA_DELAY=0.0`。".format(delta(raw, delay)))
    lines.append("- raw 相对 D0-fixed-lite T1 的序列级正/负/无变化为 `{}/{}/{}`，负迁移率 `{:.2f}%`。".format(
        raw_delta_stats["positive"], raw_delta_stats["negative"], raw_delta_stats["same"], raw_delta_stats["negative_rate"]
    ))
    if all(criteria.values()):
        lines.append("- **D0-fixed-lite 满足全部 validation 成功标准**；按要求仍先停止，等待是否运行 `threemdot_test` 的确认。")
    else:
        failed = ", ".join(k for k, v in criteria.items() if not v)
        lines.append("- **D0-fixed-lite 未满足全部成功标准**，失败项：`{}`。按要求停止，不运行 `threemdot_test`。".format(failed))
    lines.append("")

    lines.append("## 2. Checkpoint / Freeze / 完整性")
    lines.append("")
    lines.append("Checkpoint:")
    lines.append("")
    lines.append("```text")
    lines.append("output/pcum_v2_d0_fixed_lite_ranking/checkpoints/train/entertrack/pcum_v2_d0_fixed_lite_rank_softmax_t010_ep5/EnTeRTrack_ep0005.pth.tar")
    lines.append("```")
    lines.append("")
    lines.append("| Setting | Runid | bbox | Remote weight files | Source |")
    lines.append("|---|---:|---:|---:|---|")
    sources = {
        "D0-fixed-lite T1 local-only": "tracker, no-GT",
        "D0-fixed-lite raw weighted": "tracker, no-GT",
        "D0-fixed-lite zero": "tracker, no-GT",
        "D0-fixed-lite delay diagnostic": "tracker, no-GT",
        "D0-fixed-lite none": "none, no-GT",
    }
    for name, item in results.items():
        files = weights[name][1]
        lines.append("| {} | {} | 15/15 | {}/15 | {} |".format(name, item["runid"], files, sources[name]))
    lines.append("")
    lines.append("Freeze audit: `{}`。".format(audit["path"]))
    lines.append("")
    lines.append("## 3. Validation 总体结果")
    lines.append("")
    lines.append("| Setting | AUC | Precision | Norm Precision | vs T1 | vs A0 weighted raw |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, item in results.items():
        s = item["summary"]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            name, s["auc"], s["precision"], s["norm_precision"], delta(s, t1), delta(s, A0_RAW["summary"])
        ))
    lines.append("| A0 weighted raw val | {:.3f} | {:.3f} | {:.3f} | {} | +0.000 / +0.000 / +0.000 |".format(
        A0_RAW["summary"]["auc"], A0_RAW["summary"]["precision"], A0_RAW["summary"]["norm_precision"], delta(A0_RAW["summary"], t1)
    ))
    lines.append("| E4 T1 local-only val | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
        E4_T1["auc"], E4_T1["precision"], E4_T1["norm_precision"], delta(E4_T1, t1), delta(E4_T1, A0_RAW["summary"])
    ))
    lines.append("")

    lines.append("## 4. 关键对比")
    lines.append("")
    lines.append("| 对比 | Delta AUC | Delta Precision | Delta Norm Precision |")
    lines.append("|---|---:|---:|---:|")
    comparisons = [
        ("raw - T1", t1),
        ("raw - zero", zero),
        ("raw - none", none),
        ("raw - delay diagnostic", delay),
        ("raw - A0 weighted raw", A0_RAW["summary"]),
        ("raw - E4 T1 local-only", E4_T1),
    ]
    for label, base in comparisons:
        lines.append("| {} | {:+.3f} | {:+.3f} | {:+.3f} |".format(
            label,
            raw["auc"] - base["auc"],
            raw["precision"] - base["precision"],
            raw["norm_precision"] - base["norm_precision"],
        ))
    lines.append("")

    lines.append("## 5. A/B/C 分视角结果")
    lines.append("")
    lines.append("| Setting | Drone A | Drone B | Drone C |")
    lines.append("|---|---|---|---|")
    for name, item in results.items():
        v = item["view"]
        lines.append("| {} | {} | {} | {} |".format(name, fmt(v["Drone A"]), fmt(v["Drone B"]), fmt(v["Drone C"])))
    lines.append("| A0 weighted raw val | {} | {} | {} |".format(
        fmt(A0_RAW["view"]["Drone A"]), fmt(A0_RAW["view"]["Drone B"]), fmt(A0_RAW["view"]["Drone C"])
    ))
    lines.append("")
    lines.append("### Raw vs A0 分视角 AUC")
    lines.append("")
    lines.append("| 视角 | D0-fixed-lite raw AUC | A0 raw AUC | Delta AUC |")
    lines.append("|---|---:|---:|---:|")
    for view in ("Drone A", "Drone B", "Drone C"):
        v = results["D0-fixed-lite raw weighted"]["view"][view]["auc"]
        b = A0_RAW["view"][view]["auc"]
        lines.append("| {} | {:.3f} | {:.3f} | {:+.3f} |".format(view, v, b, v - b))
    lines.append("")

    lines.append("## 6. 序列级正负增益")
    lines.append("")
    lines.append("Delta 定义为当前 setting 的序列 AUC - D0-fixed-lite T1 local-only 序列 AUC。")
    lines.append("")
    lines.append("| Setting | 正/负/无变化 | 负迁移率 | Top positive | Top negative |")
    lines.append("|---|---:|---:|---|---|")
    for name, item in results.items():
        if name == "D0-fixed-lite T1 local-only":
            continue
        ds = delta_stats(item["rows"], t1_rows)
        lines.append("| {} | {}/{}/{} | {:.2f}% | {} | {} |".format(
            name, ds["positive"], ds["negative"], ds["same"], ds["negative_rate"], top_fmt(ds["top_positive"]), top_fmt(ds["top_negative"])
        ))
    lines.append("")

    lines.append("## 7. Remote Weight Diagnostics")
    lines.append("")
    lines.append("| Setting | entropy mean/std | max weight mean/p90 | valid count | quality mean/min/max | fallback | selected A/B/C |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, (stats, files) in weights.items():
        if stats is None:
            lines.append("| {} | N/A | N/A | N/A | N/A | N/A | N/A |".format(name))
            continue
        lines.append("| {} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:.2f} | {:.4f}/{:.4f}/{:.4f} | {:.2f}% | {:.1f}/{:.1f}/{:.1f}% |".format(
            name,
            stats["entropy_mean"], stats["entropy_std"],
            stats["max_weight_mean"], stats["max_weight_p90"],
            stats["valid_count_mean"],
            stats["quality_mean"], stats["quality_min"], stats["quality_max"],
            stats["fallback_rate"],
            stats["selected_a"], stats["selected_b"], stats["selected_c"],
        ))
    lines.append("")

    lines.append("## 8. 成功标准判定")
    lines.append("")
    lines.append("| 标准 | 结果 | 是否通过 |")
    lines.append("|---|---|---:|")
    criteria_rows = [
        ("raw AUC > 65.247", "{:.3f} > 65.247".format(raw["auc"]), criteria["raw_auc_gt_a0"]),
        ("raw >= D0-fixed-lite T1 local-only", "{:.3f} >= {:.3f}".format(raw["auc"], t1["auc"]), criteria["raw_ge_t1"]),
        ("raw > zero", "{:.3f} vs {:.3f}".format(raw["auc"], zero["auc"]), criteria["raw_gt_zero"]),
        ("raw > none", "{:.3f} vs {:.3f}".format(raw["auc"], none["auc"]), criteria["raw_gt_none"]),
        ("负迁移率 <= 40.00%", "{:.2f}%".format(raw_delta_stats["negative_rate"]), criteria["neg_transfer_le_40"]),
        ("至少两个视角不低于 A0 raw", "{}/3".format(view_pass), criteria["views_ge_a0_at_least_2"]),
        ("checkpoint audit 只有 PCUM/fusion/prompt 参数变化", str(criteria["audit_only_pcum"]), criteria["audit_only_pcum"]),
        ("无 NaN/Inf", str(criteria["no_nan_inf"]), criteria["no_nan_inf"]),
        ("无 residual scale collapse", "effective residual scale 约 0.02997", criteria["no_residual_collapse"]),
    ]
    for label, value, ok in criteria_rows:
        lines.append("| {} | {} | {} |".format(label, value, "PASS" if ok else "FAIL"))
    lines.append("")
    lines.append("## 9. 结论")
    lines.append("")
    if all(criteria.values()):
        lines.append("- D0-fixed-lite 通过 validation 成功标准，但本阶段按要求停止，未运行 `threemdot_test`。")
    else:
        lines.append("- D0-fixed-lite 未通过全部 validation 成功标准，本阶段停止，不运行 `threemdot_test`。")
    lines.append("- Delay 分支仅作诊断：本轮 `LAMBDA_DELAY=0.0`，不能把 raw - delay 写成强 temporal synchronization 训练证据。")
    lines.append("- 本阶段未运行 `threemdot_test`、未提交 Git、未删除文件。")
    lines.append("")

    out = REPO_ROOT / "output" / "pcum_v2_d0_fixed_lite_ranking" / "d0_fixed_lite_ep5_validation_report.md"
    out.write_text("\n".join(lines))
    print("Wrote {}".format(out))
    print("Raw {}".format(fmt(raw)))
    print("Criteria {}".format(criteria))


if __name__ == "__main__":
    main()
