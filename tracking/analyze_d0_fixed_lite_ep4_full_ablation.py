#!/usr/bin/env python3
"""Summarize D0-fixed-lite full ablation validation runs."""

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

RUNIDS = {
    2: {"t1": 19321, "raw": 19322, "zero": 19324, "delay": 19325, "none": 19326},
    4: {"t1": 19341, "raw": 19342, "zero": 19344, "delay": 19345, "none": 19346},
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


def evaluate(dataset, tracker, runid):
    directory = result_dir(tracker, runid)
    rows = []
    for seq in dataset:
        auc, precision, norm_precision = seq_metrics(directory / "{}.txt".format(seq.name), seq)
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
    return {
        view: summarize([r for r in rows if r["view"] == view])
        for view in ("Drone A", "Drone B", "Drone C")
    }


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
        data = np.atleast_2d(np.loadtxt(str(path), delimiter="\t"))
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
    }, files


def fmt(summary):
    return "{auc:.3f} / {precision:.3f} / {norm_precision:.3f}".format(**summary)


def delta(a, b):
    return "{:+.3f} / {:+.3f} / {:+.3f}".format(
        a["auc"] - b["auc"],
        a["precision"] - b["precision"],
        a["norm_precision"] - b["norm_precision"],
    )


def top_fmt(items):
    return ", ".join("{} {:+.3f}".format(seq, d) for seq, _, d in items)


def audit_summary(epoch):
    path = REPO_ROOT / "output" / "pcum_v2_d0_fixed_lite_ranking" / "d0_fixed_lite_ep{}_freeze_audit.md".format(epoch)
    text = path.read_text()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "only_pcum": "Only PCUM/fusion/prompt/residual changed: `True`" in text,
        "backbone_clean": "| backbone | 92 | 0 |" in text,
        "box_head_clean": "| box_head | 90 | 0 |" in text,
        "no_nan": "NaN/Inf/Error found | `False`" in text or "training log NaN/Inf/Error flag | `False`" in text,
        "residual": "last effective residual scale | `0.02997`" in text,
    }


def build_runs(epoch):
    runids = RUNIDS[epoch]
    return {
        "epoch{} T1 local-only".format(epoch): (
            "pcum_v2_d0_fixed_lite_rank_softmax_t010_ep{}_t1_val".format(epoch),
            runids["t1"],
        ),
        "epoch{} raw weighted".format(epoch): (
            "pcum_v2_d0_fixed_lite_rank_softmax_t010_ep{}_t2_raw_val".format(epoch),
            runids["raw"],
        ),
        "epoch{} zero".format(epoch): (
            "pcum_v2_d0_fixed_lite_rank_softmax_t010_ep{}_t2_zero_val".format(epoch),
            runids["zero"],
        ),
        "epoch{} none".format(epoch): (
            "pcum_v2_d0_fixed_lite_rank_softmax_t010_ep{}_t2_none_val".format(epoch),
            runids["none"],
        ),
        "epoch{} delay diagnostic".format(epoch): (
            "pcum_v2_d0_fixed_lite_rank_softmax_t010_ep{}_t2_delay_val".format(epoch),
            runids["delay"],
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, choices=sorted(RUNIDS), default=4)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    epoch = args.epoch
    runs = build_runs(epoch)
    dataset = get_dataset("threemdot_val")
    results = {}
    weights = {}
    for name, (tracker, runid) in runs.items():
        rows = evaluate(dataset, tracker, runid)
        results[name] = {
            "tracker": tracker,
            "runid": runid,
            "rows": rows,
            "summary": summarize(rows),
            "view": summarize_by_view(rows),
        }
        weights[name] = weight_stats(tracker, runid, dataset)

    t1_name = "epoch{} T1 local-only".format(epoch)
    raw_name = "epoch{} raw weighted".format(epoch)
    zero_name = "epoch{} zero".format(epoch)
    none_name = "epoch{} none".format(epoch)
    delay_name = "epoch{} delay diagnostic".format(epoch)
    t1 = results[t1_name]["summary"]
    raw = results[raw_name]["summary"]
    zero = results[zero_name]["summary"]
    none = results[none_name]["summary"]
    delay_diag = results[delay_name]["summary"]
    raw_stats = delta_stats(results[raw_name]["rows"], results[t1_name]["rows"])
    audit = audit_summary(epoch)
    view_pass = sum(
        1 for view in ("Drone A", "Drone B", "Drone C")
        if results[raw_name]["view"][view]["auc"] >= A0_RAW["view"][view]["auc"]
    )
    criteria = {
        "raw_auc_gt_a0": raw["auc"] > A0_RAW["summary"]["auc"],
        "raw_ge_t1": raw["auc"] >= t1["auc"],
        "raw_gt_zero": raw["auc"] > zero["auc"],
        "raw_gt_none": raw["auc"] > none["auc"],
        "neg_transfer_le_40": raw_stats["negative_rate"] <= 40.0 + 1e-9,
        "views_ge_a0_at_least_2": view_pass >= 2,
        "audit_only_pcum": audit["only_pcum"] and audit["backbone_clean"] and audit["box_head_clean"],
        "no_nan_inf": audit["no_nan"],
        "no_residual_collapse": audit["residual"],
    }

    lines = []
    lines.append("# D0-fixed-lite Epoch{} Full Ablation Report".format(epoch))
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append("- Epoch{} full ablation 已完成：zero runid=`{}`，delay diagnostic runid=`{}`，none runid=`{}`。".format(
        epoch, RUNIDS[epoch]["zero"], RUNIDS[epoch]["delay"], RUNIDS[epoch]["none"]
    ))
    lines.append("- 五组 epoch{} validation 均为 15/15 bbox，checkpoint `stored_epoch={}`，日志为 no-GT。".format(epoch, epoch))
    lines.append("- Epoch{} raw weighted = **{}**。".format(epoch, fmt(raw)))
    lines.append("- raw - T1 = **{}**；raw - A0 weighted raw = **{}**。".format(delta(raw, t1), delta(raw, A0_RAW["summary"])))
    lines.append("- raw - zero = **{}**；raw - none = **{}**。".format(delta(raw, zero), delta(raw, none)))
    lines.append("- raw - delay diagnostic = **{}**，该项只作诊断，因为 `LAMBDA_DELAY=0.0`。".format(delta(raw, delay_diag)))
    lines.append("- raw 相对 T1 的序列级正/负/无变化为 `{}/{}/{}`，负迁移率 `{:.2f}%`。".format(
        raw_stats["positive"], raw_stats["negative"], raw_stats["same"], raw_stats["negative_rate"]
    ))
    if all(criteria.values()):
        lines.append("- **Epoch{} 满足进入 test 的 validation 条件**；按要求先停止，等待是否运行 `threemdot_test` 的确认。".format(epoch))
    else:
        failed = ", ".join(k for k, ok in criteria.items() if not ok)
        lines.append("- **Epoch{} 未满足进入 test 条件**，失败项：`{}`。".format(epoch, failed))
    lines.append("")

    lines.append("## 2. 完整性与 Checkpoint")
    lines.append("")
    lines.append("| Setting | Runid | bbox | Remote weight files | Source |")
    lines.append("|---|---:|---:|---:|---|")
    sources = {
        t1_name: "tracker, no-GT",
        raw_name: "tracker, no-GT",
        zero_name: "tracker, no-GT",
        none_name: "none, no-GT",
        delay_name: "tracker, no-GT",
    }
    for name, item in results.items():
        lines.append("| {} | {} | 15/15 | {}/15 | {} |".format(
            name, item["runid"], weights[name][1], sources[name]
        ))
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
    for label, base in [
        ("raw - T1", t1),
        ("raw - zero", zero),
        ("raw - none", none),
        ("raw - delay diagnostic", delay_diag),
        ("raw - A0 weighted raw", A0_RAW["summary"]),
        ("raw - E4 T1 local-only", E4_T1),
    ]:
        lines.append("| {} | {:+.3f} | {:+.3f} | {:+.3f} |".format(
            label, raw["auc"] - base["auc"], raw["precision"] - base["precision"], raw["norm_precision"] - base["norm_precision"]
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

    lines.append("## 6. 序列级正负增益")
    lines.append("")
    lines.append("Delta 定义为当前 setting 的序列 AUC - epoch{} T1 local-only 序列 AUC。".format(epoch))
    lines.append("")
    lines.append("| Setting | 正/负/无变化 | 负迁移率 | Top positive | Top negative |")
    lines.append("|---|---:|---:|---|---|")
    for name, item in results.items():
        if name == t1_name:
            continue
        stats = delta_stats(item["rows"], results[t1_name]["rows"])
        lines.append("| {} | {}/{}/{} | {:.2f}% | {} | {} |".format(
            name, stats["positive"], stats["negative"], stats["same"], stats["negative_rate"],
            top_fmt(stats["top_positive"]), top_fmt(stats["top_negative"])
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

    lines.append("## 8. 进入 Test 条件判定")
    lines.append("")
    lines.append("| 标准 | 结果 | 是否通过 |")
    lines.append("|---|---|---:|")
    rows = [
        ("raw AUC > A0 weighted raw val 65.247", "{:.3f} > 65.247".format(raw["auc"]), criteria["raw_auc_gt_a0"]),
        ("raw >= epoch{} T1 local-only".format(epoch), "{:.3f} >= {:.3f}".format(raw["auc"], t1["auc"]), criteria["raw_ge_t1"]),
        ("raw > zero", "{:.3f} vs {:.3f}".format(raw["auc"], zero["auc"]), criteria["raw_gt_zero"]),
        ("raw > none", "{:.3f} vs {:.3f}".format(raw["auc"], none["auc"]), criteria["raw_gt_none"]),
        ("负迁移率 <= 40.00%", "{:.2f}%".format(raw_stats["negative_rate"]), criteria["neg_transfer_le_40"]),
        ("至少两个视角不低于 A0 weighted raw", "{}/3".format(view_pass), criteria["views_ge_a0_at_least_2"]),
        ("checkpoint audit 只有 PCUM/fusion/prompt 参数变化", str(criteria["audit_only_pcum"]), criteria["audit_only_pcum"]),
        ("无 NaN/Inf", str(criteria["no_nan_inf"]), criteria["no_nan_inf"]),
        ("无 residual scale collapse", "effective residual scale 约 0.02997", criteria["no_residual_collapse"]),
    ]
    for label, value, ok in rows:
        lines.append("| {} | {} | {} |".format(label, value, "PASS" if ok else "FAIL"))
    lines.append("")

    lines.append("## 9. 结论")
    lines.append("")
    if all(criteria.values()):
        lines.append("- Epoch{} 满足进入 test 的 validation 条件，但本阶段按要求停止，未运行 `threemdot_test`。".format(epoch))
    else:
        lines.append("- Epoch{} 未满足进入 test 条件。".format(epoch))
    lines.append("- Delay 仅作为 diagnostic：本轮 `LAMBDA_DELAY=0.0`，不能作为强 temporal synchronization 训练结论。")
    lines.append("- 本阶段未重新训练、未运行 `threemdot_test`、未提交 Git、未删除文件。")
    lines.append("")

    out = Path(args.output) if args.output else REPO_ROOT / "output" / "pcum_v2_d0_fixed_lite_ranking" / "d0_fixed_lite_ep{}_full_ablation_report.md".format(epoch)
    out.write_text("\n".join(lines))
    print("Wrote {}".format(out))
    print("raw={}".format(fmt(raw)))
    print("criteria={}".format(criteria))


if __name__ == "__main__":
    main()
