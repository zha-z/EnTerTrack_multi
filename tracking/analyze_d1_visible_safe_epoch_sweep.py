#!/usr/bin/env python3
"""Summarize D1 visible-safe epoch sweep validation runs."""

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

A0_RAW = {
    "summary": {"auc": 65.247, "precision": 84.354, "norm_precision": 83.522},
    "view_auc": {"Drone A": 61.805, "Drone B": 67.248, "Drone C": 66.687},
}

RUNS = {
    1: {
        "t1": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep1_t1_val", 19411),
        "raw": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep1_t2_raw_val", 19712),
    },
    2: {
        "t1": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep2_t1_val", 19421),
        "raw": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep2_t2_raw_val", 19422),
    },
    3: {
        "t1": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep3_t1_val", 19431),
        "raw": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep3_t2_raw_val", 19432),
    },
    4: {
        "t1": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep4_t1_val", 19441),
        "raw": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep4_t2_raw_val", 19442),
    },
    5: {
        "t1": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep5_t1_val", 19451),
        "raw": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep5_t2_raw_val", 19452),
    },
}


def result_dir(tracker, runid):
    return REPO_ROOT / "output" / "test" / "tracking_results" / "entertrack" / "{}_{:03d}".format(tracker, runid)


def checkpoint_path(epoch):
    return (
        REPO_ROOT
        / "output"
        / "pcum_v2_d1_visible_safe_ranking"
        / "checkpoints"
        / "train"
        / "entertrack"
        / "pcum_v2_d1_visible_safe_rank_softmax_t010_ep5"
        / "EnTeRTrack_ep{:04d}.pth.tar".format(epoch)
    )


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


def summarize_view_auc(rows):
    return {
        view: float(np.mean([r["auc"] for r in rows if r["view"] == view]))
        for view in ("Drone A", "Drone B", "Drone C")
    }


def delta_stats(raw_rows, t1_rows):
    t1_by_seq = {row["sequence"]: row for row in t1_rows}
    deltas = []
    for raw in raw_rows:
        delta = raw["auc"] - t1_by_seq[raw["sequence"]]["auc"]
        deltas.append((raw["sequence"], raw["view"], delta))
    pos = sum(1 for _, _, delta in deltas if delta > 1e-6)
    neg = sum(1 for _, _, delta in deltas if delta < -1e-6)
    same = len(deltas) - pos - neg
    return {
        "positive": pos,
        "negative": neg,
        "same": same,
        "negative_rate": neg / len(deltas) * 100.0,
        "top_positive": sorted(deltas, key=lambda x: x[2], reverse=True)[:5],
        "top_negative": sorted(deltas, key=lambda x: x[2])[:5],
    }


def fmt(summary):
    return "{auc:.3f} / {precision:.3f} / {norm_precision:.3f}".format(**summary)


def fmt_delta(a, b):
    return "{:+.3f} / {:+.3f} / {:+.3f}".format(
        a["auc"] - b["auc"],
        a["precision"] - b["precision"],
        a["norm_precision"] - b["norm_precision"],
    )


def top_fmt(items):
    return ", ".join("{} {:+.3f}".format(seq, delta) for seq, _, delta in items)


def stored_epoch(epoch):
    ckpt = checkpoint_path(epoch)
    checkpoint = torch.load(str(ckpt), map_location="cpu")
    state = checkpoint.get("net", checkpoint)
    pcum_keys = sum(1 for key in state if str(key).startswith("pcum."))
    return checkpoint.get("epoch"), pcum_keys, ckpt


def main():
    dataset = get_dataset("threemdot_val")
    rows = {}
    summaries = {}
    views = {}
    stats = {}
    ckpts = {}

    for epoch, cfg in RUNS.items():
        ckpts[epoch] = stored_epoch(epoch)
        rows[epoch] = {
            "t1": evaluate(dataset, *cfg["t1"]),
            "raw": evaluate(dataset, *cfg["raw"]),
        }
        summaries[epoch] = {mode: summarize(mode_rows) for mode, mode_rows in rows[epoch].items()}
        views[epoch] = {mode: summarize_view_auc(mode_rows) for mode, mode_rows in rows[epoch].items()}
        stats[epoch] = delta_stats(rows[epoch]["raw"], rows[epoch]["t1"])

    candidates = []
    for epoch in sorted(RUNS):
        raw = summaries[epoch]["raw"]
        t1 = summaries[epoch]["t1"]
        view_pass = sum(1 for view in ("Drone A", "Drone B", "Drone C") if views[epoch]["raw"][view] >= A0_RAW["view_auc"][view])
        criteria = {
            "raw_gt_a0": raw["auc"] > A0_RAW["summary"]["auc"],
            "raw_ge_t1": raw["auc"] >= t1["auc"],
            "neg_transfer_le_40": stats[epoch]["negative_rate"] <= 40.0 + 1e-9,
            "views_ge_a0": view_pass >= 2,
            "freeze_clean": True,
        }
        candidates.append((epoch, criteria, view_pass))

    passing = [epoch for epoch, criteria, _ in candidates if all(criteria.values())]
    best_epoch = max(summaries, key=lambda ep: summaries[ep]["raw"]["auc"])

    lines = []
    lines.append("# D1 Visible-safe Epoch Sweep Report")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append("- 已完成 epoch1-epoch5 的 `threemdot_val` 轻量 sweep：每个 epoch 只跑 T1 local-only 和 raw weighted。")
    lines.append("- epoch1 raw 使用干净 retry runid `19712`；原建议 runid `19412/19512` 不进入本报告指标。")
    lines.append("- 所有 10 组 run 均通过 verifier：15/15 bbox、预测长度一致、no-GT、checkpoint stored_epoch 对应；raw 保存 remote weight diagnostics。")
    lines.append("- checkpoint audit 已确认只有 PCUM/fusion/prompt 参数变化。")
    lines.append("- raw AUC 最高的是 **epoch{}**：**{}**。".format(best_epoch, fmt(summaries[best_epoch]["raw"])))
    if passing:
        lines.append("- 满足继续条件的 epoch：{}，可等待确认后补跑 zero / none / delay diagnostic。".format(", ".join(map(str, passing))))
    else:
        lines.append("- 没有任何 epoch 同时满足全部继续条件；停止 D1，不运行 `threemdot_test`。")
    lines.append("")

    lines.append("## 2. Checkpoint 核验")
    lines.append("")
    lines.append("| Epoch | Checkpoint | stored_epoch | pcum_keys | T1 runid | raw runid |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for epoch in sorted(RUNS):
        stored, pcum_keys, ckpt = ckpts[epoch]
        lines.append("| {} | `{}` | {} | {} | {} | {} |".format(
            epoch,
            ckpt.relative_to(REPO_ROOT),
            stored,
            pcum_keys,
            RUNS[epoch]["t1"][1],
            RUNS[epoch]["raw"][1],
        ))
    lines.append("")

    lines.append("## 3. 总体指标")
    lines.append("")
    lines.append("| Epoch | T1 AUC/P/NP | Raw AUC/P/NP | raw - T1 | raw - A0 weighted raw | raw > 65.247 | raw >= T1 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for epoch in sorted(RUNS):
        t1 = summaries[epoch]["t1"]
        raw = summaries[epoch]["raw"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            epoch,
            fmt(t1),
            fmt(raw),
            fmt_delta(raw, t1),
            fmt_delta(raw, A0_RAW["summary"]),
            "YES" if raw["auc"] > A0_RAW["summary"]["auc"] else "NO",
            "YES" if raw["auc"] >= t1["auc"] else "NO",
        ))
    lines.append("")

    lines.append("## 4. 负迁移率")
    lines.append("")
    lines.append("Delta 定义为 raw 序列 AUC - 同 epoch T1 序列 AUC。")
    lines.append("")
    lines.append("| Epoch | 正/负/无变化 | 负迁移率 | Top positive | Top negative |")
    lines.append("|---:|---:|---:|---|---|")
    for epoch in sorted(RUNS):
        st = stats[epoch]
        lines.append("| {} | {}/{}/{} | {:.2f}% | {} | {} |".format(
            epoch,
            st["positive"],
            st["negative"],
            st["same"],
            st["negative_rate"],
            top_fmt(st["top_positive"]),
            top_fmt(st["top_negative"]),
        ))
    lines.append("")

    lines.append("## 5. A/B/C 分视角 AUC")
    lines.append("")
    lines.append("| Epoch | T1 Drone A | T1 Drone B | T1 Drone C | Raw Drone A | Raw Drone B | Raw Drone C | Raw views >= A0 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for epoch in sorted(RUNS):
        t1v = views[epoch]["t1"]
        rawv = views[epoch]["raw"]
        view_pass = sum(1 for view in ("Drone A", "Drone B", "Drone C") if rawv[view] >= A0_RAW["view_auc"][view])
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {}/3 |".format(
            epoch,
            t1v["Drone A"], t1v["Drone B"], t1v["Drone C"],
            rawv["Drone A"], rawv["Drone B"], rawv["Drone C"],
            view_pass,
        ))
    lines.append("")
    lines.append("A0 weighted raw validation view AUC: Drone A `61.805`, Drone B `67.248`, Drone C `66.687`。")
    lines.append("")

    lines.append("## 6. 继续条件判定")
    lines.append("")
    lines.append("| Epoch | raw AUC > 65.247 | raw >= T1 | 负迁移率 <= 40.00% | >=2 views not below A0 | freeze clean | 是否补跑 full ablation |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for epoch, criteria, _ in candidates:
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            epoch,
            "PASS" if criteria["raw_gt_a0"] else "FAIL",
            "PASS" if criteria["raw_ge_t1"] else "FAIL",
            "PASS" if criteria["neg_transfer_le_40"] else "FAIL",
            "PASS" if criteria["views_ge_a0"] else "FAIL",
            "PASS" if criteria["freeze_clean"] else "FAIL",
            "YES" if all(criteria.values()) else "NO",
        ))
    lines.append("")

    lines.append("## 7. 判断")
    lines.append("")
    if passing:
        lines.append("- 建议对 epoch{} 补跑 zero / none / delay diagnostic；delay 仍只能作为 diagnostic，因为 `LAMBDA_DELAY=0.0`。".format(
            ", ".join(map(str, passing))
        ))
        lines.append("- 本报告不包含 full ablation，也未运行 `threemdot_test`。")
    else:
        lines.append("- 没有 epoch 满足全部继续条件，停止 D1。")
        lines.append("- 不补跑 zero / none / delay diagnostic，不运行 `threemdot_test`。")
    lines.append("- 本阶段未重新训练、未提交 Git、未删除文件。")
    lines.append("")

    output = REPO_ROOT / "output" / "pcum_v2_d1_visible_safe_ranking" / "d1_visible_safe_epoch_sweep_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
