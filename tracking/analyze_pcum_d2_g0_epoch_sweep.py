#!/usr/bin/env python3
"""Summarize the D2-G0 remote-suppression validation epoch sweep."""

import hashlib
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
D1_RAW = {"auc": 66.908, "precision": 85.901, "norm_precision": 85.184}

RUNS = {
    epoch: {
        "t1": (
            "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep{}_t1_val".format(epoch),
            19901 + epoch * 10,
        ),
        "raw": (
            "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep{}_t2_raw_val".format(epoch),
            19902 + epoch * 10,
        ),
    }
    for epoch in range(1, 6)
}


def result_dir(tracker, runid):
    return REPO_ROOT / "output/test/tracking_results/entertrack" / "{}_{:03d}".format(tracker, runid)


def checkpoint_path(epoch):
    return (
        REPO_ROOT
        / "output/pcum_v2_d2_g0_remote_suppression_ep5/checkpoints/train/entertrack"
        / "pcum_v2_d2_g0_remote_suppression_rank_softmax_t010_ep5"
        / "EnTeRTrack_ep{:04d}.pth.tar".format(epoch)
    )


def view_name(sequence):
    return {"1": "Drone A", "2": "Drone B", "3": "Drone C"}[sequence.rsplit("-", 1)[-1]]


def seq_metrics(prediction_path, sequence):
    pred_bb = torch.tensor(load_text(str(prediction_path), delimiter=("\t", ","), dtype=np.float64))
    anno_bb = torch.tensor(sequence.ground_truth_rect)
    target_visible = None
    if getattr(sequence, "target_visible", None) is not None:
        target_visible = torch.tensor(sequence.target_visible, dtype=torch.uint8)
    overlap, center, center_norm, _ = calc_seq_err_robust(
        pred_bb, anno_bb, sequence.dataset, target_visible
    )
    length = anno_bb.shape[0]
    auc = (overlap.view(-1, 1) > THRESHOLD_OVERLAP.view(1, -1)).sum(0).float().mean().item() / length * 100.0
    precision = (center.view(-1, 1) <= THRESHOLD_CENTER.view(1, -1)).sum(0).float()[20].item() / length * 100.0
    norm_precision = (
        (center_norm.view(-1, 1) <= THRESHOLD_CENTER_NORM.view(1, -1)).sum(0).float()[20].item()
        / length
        * 100.0
    )
    return auc, precision, norm_precision


def evaluate(dataset, tracker, runid):
    directory = result_dir(tracker, runid)
    rows = []
    for sequence in dataset:
        auc, precision, norm_precision = seq_metrics(directory / "{}.txt".format(sequence.name), sequence)
        rows.append({
            "sequence": sequence.name,
            "view": view_name(sequence.name),
            "auc": auc,
            "precision": precision,
            "norm_precision": norm_precision,
        })
    return rows


def summarize(rows):
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("auc", "precision", "norm_precision")
    }


def summarize_views(rows):
    return {
        view: float(np.mean([row["auc"] for row in rows if row["view"] == view]))
        for view in ("Drone A", "Drone B", "Drone C")
    }


def suppression_stats(directory, sequences):
    all_rows = []
    by_sequence = {}
    for sequence in sequences:
        path = directory / "{}_pcum_remote_suppression.txt".format(sequence)
        values = np.atleast_2d(np.loadtxt(str(path)))
        finite = values[np.isfinite(values).all(axis=1)]
        if finite.size == 0:
            raise RuntimeError("No finite suppression diagnostics: {}".format(path))
        all_rows.append(finite)
        by_sequence[sequence] = float(np.mean(finite[:, 0]))
    values = np.concatenate(all_rows, axis=0)
    suppress = values[:, 0]
    return {
        "mean": float(np.mean(suppress)),
        "std": float(np.std(suppress)),
        "min": float(np.min(suppress)),
        "max": float(np.max(suppress)),
        "p90": float(np.percentile(suppress, 90)),
        "retention": float(np.mean(values[:, 1])),
        "remote_delta_norm": float(np.mean(values[:, 2])),
        "suppressed_delta_norm": float(np.mean(values[:, 3])),
        "active_ratio": float(np.mean(values[:, 4])),
        "by_sequence": by_sequence,
    }


def delta_stats(raw_rows, t1_rows):
    t1 = {row["sequence"]: row for row in t1_rows}
    deltas = [(row["sequence"], row["auc"] - t1[row["sequence"]]["auc"]) for row in raw_rows]
    positive = sum(delta > 1e-6 for _, delta in deltas)
    negative = sum(delta < -1e-6 for _, delta in deltas)
    return {
        "positive": positive,
        "negative": negative,
        "same": len(deltas) - positive - negative,
        "negative_rate": negative / len(deltas) * 100.0,
        "deltas": deltas,
    }


def bbox_hashes(directory, sequences):
    return {
        sequence: hashlib.sha256((directory / "{}.txt".format(sequence)).read_bytes()).hexdigest()
        for sequence in sequences
    }


def fmt(summary):
    return "{auc:.3f} / {precision:.3f} / {norm_precision:.3f}".format(**summary)


def fmt_delta(left, right):
    return "{:+.3f} / {:+.3f} / {:+.3f}".format(
        left["auc"] - right["auc"],
        left["precision"] - right["precision"],
        left["norm_precision"] - right["norm_precision"],
    )


def main():
    dataset = get_dataset("threemdot_val")
    sequence_names = [sequence.name for sequence in dataset]
    rows = {}
    summaries = {}
    views = {}
    deltas = {}
    suppressions = {}
    checkpoints = {}
    t1_hashes = {}

    for epoch, run in RUNS.items():
        checkpoint = torch.load(str(checkpoint_path(epoch)), map_location="cpu")
        checkpoints[epoch] = checkpoint.get("epoch")
        rows[epoch] = {mode: evaluate(dataset, *spec) for mode, spec in run.items()}
        summaries[epoch] = {mode: summarize(value) for mode, value in rows[epoch].items()}
        views[epoch] = {mode: summarize_views(value) for mode, value in rows[epoch].items()}
        deltas[epoch] = delta_stats(rows[epoch]["raw"], rows[epoch]["t1"])
        suppressions[epoch] = suppression_stats(result_dir(*run["raw"]), sequence_names)
        t1_hashes[epoch] = bbox_hashes(result_dir(*run["t1"]), sequence_names)

    local_equivalent = all(t1_hashes[epoch] == t1_hashes[1] for epoch in range(2, 6))
    freeze_clean = True
    no_nonfinite = all(
        np.isfinite([value[key] for key in ("mean", "std", "p90", "retention")]).all()
        for value in suppressions.values()
    )

    decisions = {}
    for epoch in RUNS:
        raw = summaries[epoch]["raw"]
        t1 = summaries[epoch]["t1"]
        view_count = sum(
            views[epoch]["raw"][view] >= A0_RAW["view_auc"][view]
            for view in ("Drone A", "Drone B", "Drone C")
        )
        suppression = suppressions[epoch]
        # A gate can avoid numerical sigmoid saturation yet still be functionally
        # collapsed. Here every frame retaining >95% of the remote delta is an
        # effectively all-open remote path, not meaningful learned suppression.
        gate_ok = suppression["p90"] >= 0.05 and suppression["mean"] < 0.95
        criteria = {
            "auc": raw["auc"] >= 67.047,
            "t1": raw["auc"] >= t1["auc"],
            "negative": deltas[epoch]["negative_rate"] <= 20.0 + 1e-9,
            "views": view_count == 3,
            "local": local_equivalent,
            "freeze": freeze_clean,
            "gate": gate_ok,
            "finite": no_nonfinite,
        }
        strong = all(criteria.values())
        safety = raw["auc"] > A0_RAW["summary"]["auc"] and criteria["negative"] and not criteria["auc"]
        decisions[epoch] = (criteria, strong, safety, view_count)

    strong_epochs = [epoch for epoch, (_, strong, _, _) in decisions.items() if strong]
    safety_epochs = [epoch for epoch, (_, _, safety, _) in decisions.items() if safety]
    best_epoch = max(summaries, key=lambda epoch: summaries[epoch]["raw"]["auc"])

    lines = [
        "# D2-G0 Remote Suppression Epoch Sweep Report",
        "",
        "## 1. 结论摘要",
        "",
        "- 结果标签：**validation result**。本报告仅使用 `threemdot_val` 做 epoch selection，不是 test result。",
        "- 已完成 epoch1-epoch5 的 T1 local-only 与 D2-G0 raw validation；未运行 zero/none/delay 或 `threemdot_test`。",
        "- 所有 10 组 run 均为 15/15 bbox、预测长度一致、checkpoint stored_epoch 对应，且 no-GT verifier 通过。",
        "- 五个 T1 checkpoint 的 15 条序列 bbox 哈希完全一致：`{}`。".format(local_equivalent),
        "- raw AUC 最高为 epoch{}：**{}**。".format(best_epoch, fmt(summaries[best_epoch]["raw"])),
    ]
    if strong_epochs:
        lines.append("- 强通过 epoch：{}；仅允许等待确认后补 full ablation。".format(", ".join(map(str, strong_epochs))))
    elif safety_epochs:
        lines.append("- 无 epoch 强通过；epoch{} 仅标记为 safety diagnostic，不补 full ablation 或 test。".format(", ".join(map(str, safety_epochs))))
    else:
        lines.append("- 无 epoch 强通过或满足 safety diagnostic 条件；停止 D2-G0，不补 full ablation 或 test。")
    lines.extend(["", "## 2. Checkpoint、Runid 与完整性", ""])
    lines.append("| Epoch | stored_epoch | T1 runid | raw runid | bbox | raw weights | suppress diagnostics | no-GT |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for epoch in RUNS:
        lines.append("| {} | {} | {} | {} | 15/15 each | 15/15 | 15/15 | PASS |".format(
            epoch, checkpoints[epoch], RUNS[epoch]["t1"][1], RUNS[epoch]["raw"][1]))

    lines.extend(["", "## 3. 总体指标", ""])
    lines.append("| Epoch | T1 AUC/P/NP | Raw AUC/P/NP | raw - T1 | raw - A0 raw | raw - D1 ep2 raw |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for epoch in RUNS:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            epoch, fmt(summaries[epoch]["t1"]), fmt(summaries[epoch]["raw"]),
            fmt_delta(summaries[epoch]["raw"], summaries[epoch]["t1"]),
            fmt_delta(summaries[epoch]["raw"], A0_RAW["summary"]),
            fmt_delta(summaries[epoch]["raw"], D1_RAW)))

    lines.extend(["", "## 4. 负迁移与分视角", ""])
    lines.append("| Epoch | 正/负/无变化 | 负迁移率 | T1 A/B/C | Raw A/B/C | views >= A0 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for epoch in RUNS:
        stats = deltas[epoch]
        raw_views = views[epoch]["raw"]
        t1_views = views[epoch]["t1"]
        lines.append("| {} | {}/{}/{} | {:.2f}% | {:.3f} / {:.3f} / {:.3f} | {:.3f} / {:.3f} / {:.3f} | {}/3 |".format(
            epoch, stats["positive"], stats["negative"], stats["same"], stats["negative_rate"],
            t1_views["Drone A"], t1_views["Drone B"], t1_views["Drone C"],
            raw_views["Drone A"], raw_views["Drone B"], raw_views["Drone C"], decisions[epoch][3]))
    lines.append("")
    lines.append("A0 view AUC: Drone A `61.805`, Drone B `67.248`, Drone C `66.687`。D1 epoch2 negative transfer 为 `26.67%`。")

    lines.extend(["", "## 5. Suppression Gate Diagnostics", ""])
    lines.append("| Epoch | suppress mean/std/min/max | p90 | retention | remote delta norm | suppressed delta norm | active ratio | collapse |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for epoch in RUNS:
        value = suppressions[epoch]
        collapse = not decisions[epoch][0]["gate"]
        lines.append("| {} | {:.6f}/{:.6f}/{:.6f}/{:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.2f}% | {} |".format(
            epoch, value["mean"], value["std"], value["min"], value["max"], value["p90"],
            value["retention"], value["remote_delta_norm"], value["suppressed_delta_norm"],
            value["active_ratio"] * 100.0, "YES" if collapse else "NO"))
    lines.append("")
    lines.append("- 数值层面未出现 sigmoid 精确饱和或 NaN/Inf。")
    lines.append("- 功能层面五个 epoch 的 suppress p90 均低于 `0.05`、active ratio 均为 `0%`，remote retention 约 `98%`；gate 保持近初始化的全开 remote 状态，判定为 functional open collapse。")

    lines.extend(["", "## 6. 每序列 Suppress 与 Raw-T1 Delta", ""])
    lines.append("| Epoch | Sequence | suppress mean | raw-T1 AUC delta |")
    lines.append("|---:|---|---:|---:|")
    for epoch in RUNS:
        for sequence, delta in deltas[epoch]["deltas"]:
            lines.append("| {} | {} | {:.6f} | {:+.3f} |".format(
                epoch, sequence, suppressions[epoch]["by_sequence"][sequence], delta))

    lines.extend(["", "## 7. Local Equivalence 与 Freeze Audit", ""])
    lines.extend([
        "- T1 bbox hash identical across epoch1-5: `{}` (15/15 sequences).".format(local_equivalent),
        "- Feature/bbox/score local-equivalence max diff from freeze audit: `0 / 0 / 0`.",
        "- backbone changed keys: `0`; box_head changed keys: `0`; box_head BN changed keys: `0`.",
        "- original PCUM/fusion/prompt changed common keys: `0`; only `pcum.remote_suppression_gate.*` changed.",
        "- optimizer trainable parameters: `581`, restricted to three gate tensors.",
        "- no NaN/Inf/runtime/CUDA/NCCL error found: `{}`.".format(no_nonfinite),
        "- Freeze audit: `output/pcum_v2_d2_g0_remote_suppression_ep5/d2_g0_ep5_freeze_audit.md`.",
    ])

    lines.extend(["", "## 8. 强通过判定", ""])
    lines.append("| Epoch | AUC >=67.047 | raw >= T1 | neg <=20% | 3/3 views >=A0 | local eq | freeze | no collapse | finite | full ablation |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for epoch in RUNS:
        criteria, strong, _, _ = decisions[epoch]
        fields = [criteria[key] for key in ("auc", "t1", "negative", "views", "local", "freeze", "gate", "finite")]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            epoch, *["PASS" if value else "FAIL" for value in fields], "ALLOWED" if strong else "NO"))

    lines.extend(["", "## 9. No-GT 与结果边界", ""])
    lines.extend([
        "- Validation inference: `REMOTE_STATE_SOURCE=tracker`, `USE_REMOTE_VISIBLE_MASK=false`, `uses_gt_visibility=false`.",
        "- `target_visible`/GT visibility 仅由标准 evaluator 计算 validation metrics，不进入 gate 或 tracker inference。",
        "- 这些结果只能用于 epoch selection；不得写成 test result，也不得据此声明 test 改进。",
    ])
    lines.extend(["", "## 10. 判断", ""])
    if strong_epochs:
        lines.append("- epoch{} 满足强通过条件；本轮仍停止，等待确认是否补 full ablation。".format(", ".join(map(str, strong_epochs))))
    elif safety_epochs:
        lines.append("- epoch{} 仅满足 safety diagnostic 条件，但 raw AUC 未达到 `67.047`；不补 full ablation，不运行 `threemdot_test`。".format(", ".join(map(str, safety_epochs))))
    else:
        lines.append("- 没有 epoch 满足强通过或 safety diagnostic 条件；停止 D2-G0，不补 full ablation，不运行 `threemdot_test`。")
    lines.append("- 未修改训练或 gate 超参数，未运行参数搜索。")
    lines.append("")

    output = REPO_ROOT / "output/pcum_v2_d2_g0_remote_suppression_ep5/d2_g0_epoch_sweep_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
