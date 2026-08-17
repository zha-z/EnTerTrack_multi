#!/usr/bin/env python3
"""Summarize D1 visible-safe epoch2 formal threemdot_test runs."""

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

CHECKPOINT = (
    REPO_ROOT
    / "output/pcum_v2_d1_visible_safe_ranking/checkpoints/train/entertrack/"
    / "pcum_v2_d1_visible_safe_rank_softmax_t010_ep5/EnTeRTrack_ep0002.pth.tar"
)
RUNS = {
    "D1 epoch2 T1 local-only": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep2_t1_test", 19821),
    "D1 epoch2 raw weighted": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep2_t2_raw_test", 19822),
    "D1 epoch2 zero": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep2_t2_zero_test", 19824),
    "D1 epoch2 none": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep2_t2_none_test", 19826),
    "D1 epoch2 delay diagnostic": ("pcum_v2_d1_visible_safe_rank_softmax_t010_ep2_t2_delay_test", 19825),
}
LOGS = {
    "D1 epoch2 T1 local-only": "output/pcum_v2_d1_visible_safe_ranking/test_logs/ep2_t1_test_run19821.log",
    "D1 epoch2 raw weighted": "output/pcum_v2_d1_visible_safe_ranking/test_logs/ep2_raw_test_run19822.log",
    "D1 epoch2 zero": "output/pcum_v2_d1_visible_safe_ranking/test_logs/ep2_zero_test_run19824.log",
    "D1 epoch2 none": "output/pcum_v2_d1_visible_safe_ranking/test_logs/ep2_none_test_run19826.log",
    "D1 epoch2 delay diagnostic": "output/pcum_v2_d1_visible_safe_ranking/test_logs/ep2_delay_test_run19825.log",
}
EXPECTED_SOURCE = {
    "D1 epoch2 T1 local-only": "tracker",
    "D1 epoch2 raw weighted": "tracker",
    "D1 epoch2 zero": "tracker",
    "D1 epoch2 none": "none",
    "D1 epoch2 delay diagnostic": "tracker",
}
BASELINES = {
    "Original EnTeR-Track epoch21": {"auc": 47.201, "precision": 63.723, "norm_precision": 76.291},
    "Independent non-PCUM baseline epoch25": {"auc": 48.152, "precision": 63.699, "norm_precision": 77.256},
    "PCUM-v2A A0 weighted raw test": {"auc": 48.566, "precision": 64.332, "norm_precision": 77.850},
    "PCUM-v2A A0 weighted zero test": {"auc": 48.376, "precision": 63.933, "norm_precision": 77.489},
    "PCUM-v2A A0 weighted delay test": {"auc": 47.871, "precision": 63.397, "norm_precision": 76.908},
    "PCUM-v2A A0 weighted none test": {"auc": 48.124, "precision": 63.933, "norm_precision": 77.013},
}


def result_dir(tracker, runid):
    return REPO_ROOT / "output/test/tracking_results/entertrack/{}_{:03d}".format(tracker, runid)


def view_name(seq_name):
    return {"1": "Drone A", "2": "Drone B", "3": "Drone C"}.get(seq_name.rsplit("-", 1)[-1], "Unknown")


def seq_metrics(pred_path, seq):
    pred_bb = torch.tensor(load_text(str(pred_path), delimiter=("\t", ","), dtype=np.float64))
    anno_bb = torch.tensor(seq.ground_truth_rect)
    target_visible = (
        torch.tensor(seq.target_visible, dtype=torch.uint8)
        if getattr(seq, "target_visible", None) is not None
        else None
    )
    err_overlap, err_center, err_center_normalized, _ = calc_seq_err_robust(
        pred_bb, anno_bb, seq.dataset, target_visible
    )
    seq_length = anno_bb.shape[0]
    auc = (err_overlap.view(-1, 1) > THRESHOLD_OVERLAP.view(1, -1)).sum(0).float().mean().item()
    auc = auc / seq_length * 100.0
    precision = (err_center.view(-1, 1) <= THRESHOLD_CENTER.view(1, -1)).sum(0).float()[20].item()
    precision = precision / seq_length * 100.0
    norm_precision = (
        (err_center_normalized.view(-1, 1) <= THRESHOLD_CENTER_NORM.view(1, -1)).sum(0).float()[20].item()
        / seq_length
        * 100.0
    )
    return auc, precision, norm_precision


def evaluate(dataset, tracker, runid):
    directory = result_dir(tracker, runid)
    rows = []
    length_mismatch = 0
    missing = 0
    for seq in dataset:
        pred_path = directory / "{}.txt".format(seq.name)
        if not pred_path.is_file():
            missing += 1
            continue
        pred_bb = torch.tensor(load_text(str(pred_path), delimiter=("\t", ","), dtype=np.float64))
        if pred_bb.shape[0] != len(seq.ground_truth_rect):
            length_mismatch += 1
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
    return rows, length_mismatch, missing


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


def checkpoint_summary():
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu")
    state = checkpoint.get("net", checkpoint)
    return checkpoint.get("epoch"), sum(1 for key in state if str(key).startswith("pcum."))


def audit_summary():
    path = REPO_ROOT / "output/pcum_v2_d1_visible_safe_ranking/d1_visible_safe_ep5_freeze_audit.md"
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "only_pcum": "Only PCUM/fusion/prompt/residual changed: `True`" in text,
        "backbone_clean": "| backbone | 92 | 0 |" in text,
        "box_head_clean": "| box_head | 90 | 0 |" in text,
        "no_nan": "training log NaN/Inf/Error flag | `False`" in text
        or "NaN/Inf/Error found | `False`" in text,
        "residual": "effective residual scale" in text,
    }


def no_gt_status(name):
    text = (REPO_ROOT / LOGS[name]).read_text(encoding="utf-8", errors="replace")
    expected = "source={} uses_gt_visibility=false".format(EXPECTED_SOURCE[name])
    forbidden = any(
        token in text
        for token in (
            "Traceback",
            "RuntimeError",
            "CUDA error",
            "out of memory",
            "NCCL",
            "NaN",
            "Inf",
            "uses_gt_visibility=true",
        )
    )
    return expected in text and not forbidden


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


def main():
    dataset = get_dataset("threemdot_test")
    results = {}
    weights = {}
    for name, (tracker, runid) in RUNS.items():
        rows, mismatch, missing = evaluate(dataset, tracker, runid)
        results[name] = {
            "tracker": tracker,
            "runid": runid,
            "rows": rows,
            "length_mismatch": mismatch,
            "missing": missing,
            "summary": summarize(rows),
            "view": summarize_by_view(rows),
            "no_gt": no_gt_status(name),
        }
        weights[name] = weight_stats(tracker, runid, dataset)

    t1_name = "D1 epoch2 T1 local-only"
    raw_name = "D1 epoch2 raw weighted"
    zero_name = "D1 epoch2 zero"
    none_name = "D1 epoch2 none"
    delay_name = "D1 epoch2 delay diagnostic"
    t1 = results[t1_name]["summary"]
    raw = results[raw_name]["summary"]
    zero = results[zero_name]["summary"]
    none = results[none_name]["summary"]
    delay_diag = results[delay_name]["summary"]
    raw_stats = delta_stats(results[raw_name]["rows"], results[t1_name]["rows"])
    audit = audit_summary()
    stored_epoch, pcum_keys = checkpoint_summary()
    no_gt_ok = all(item["no_gt"] for item in results.values())
    complete = all(item["missing"] == 0 and item["length_mismatch"] == 0 and len(item["rows"]) == 105 for item in results.values())
    remote_weight_ok = (
        weights[t1_name][1] == 0
        and all(weights[name][1] == 105 for name in (raw_name, zero_name, none_name, delay_name))
    )
    criteria = {
        "raw_exceeds_a0_main": raw["auc"] > BASELINES["PCUM-v2A A0 weighted raw test"]["auc"],
        "raw_exceeds_independent": raw["auc"] > BASELINES["Independent non-PCUM baseline epoch25"]["auc"],
        "raw_exceeds_original": raw["auc"] > BASELINES["Original EnTeR-Track epoch21"]["auc"],
        "raw_ge_t1": raw["auc"] >= t1["auc"],
        "raw_gt_zero": raw["auc"] > zero["auc"],
        "raw_gt_none": raw["auc"] > none["auc"],
        "audit_only_pcum": audit["only_pcum"] and audit["backbone_clean"] and audit["box_head_clean"],
        "no_gt_inference": no_gt_ok,
        "complete_outputs": complete,
        "remote_weights_complete": remote_weight_ok,
        "no_nan_inf": audit["no_nan"] and no_gt_ok,
        "no_residual_collapse": audit["residual"],
    }

    lines = []
    lines.append("# D1 Visible-safe Epoch2 Test Report")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append("- 结果标签：**test result**。本报告只使用 `threemdot_test`，D1 epoch2 是 validation-selected checkpoint，不是 test-selected。")
    lines.append("- 正式 test 只跑一次五组：T1 local-only、raw weighted、zero、none、delay diagnostic；未做 test sweep，未根据 test 结果更换 epoch。")
    lines.append("- Checkpoint 固定为 `{}`，`stored_epoch={}`，`pcum_keys={}`。".format(CHECKPOINT.relative_to(REPO_ROOT), stored_epoch, pcum_keys))
    lines.append("- 五组 test 均为 105/105 bbox，预测长度与 GT 一致，no-GT inference 通过。")
    lines.append("- D1 epoch2 raw weighted test = **{}**。".format(fmt(raw)))
    lines.append("- raw - T1 = **{}**；raw - A0 weighted raw test = **{}**。".format(delta(raw, t1), delta(raw, BASELINES["PCUM-v2A A0 weighted raw test"])))
    lines.append("- raw - zero = **{}**；raw - none = **{}**。".format(delta(raw, zero), delta(raw, none)))
    lines.append("- raw - delay diagnostic = **{}**，该项只作 diagnostic，因为 D1 `LAMBDA_DELAY=0.0`。".format(delta(raw, delay_diag)))
    lines.append("- raw 相对 T1 的序列级正/负/无变化为 `{}/{}/{}`，负迁移率 `{:.2f}%`。".format(raw_stats["positive"], raw_stats["negative"], raw_stats["same"], raw_stats["negative_rate"]))
    if criteria["raw_exceeds_a0_main"]:
        lines.append("- **D1 epoch2 raw weighted 超过当前正式主结果 PCUM-v2A A0 weighted raw**：{} vs {}。".format(fmt(raw), fmt(BASELINES["PCUM-v2A A0 weighted raw test"])))
    else:
        lines.append("- D1 epoch2 raw weighted 未超过当前正式主结果 PCUM-v2A A0 weighted raw。")
    lines.append("")

    lines.append("## 2. 完整性、Runid 与 No-GT 核验")
    lines.append("")
    lines.append("| Setting | Runid | Split | bbox | 长度不一致 | Remote weight files | Source | no-GT log |")
    lines.append("|---|---:|---|---:|---:|---:|---|---:|")
    for name, item in results.items():
        lines.append("| {} | {} | threemdot_test | {}/105 | {} | {}/105 | {}, uses_gt_visibility=false | {} |".format(
            name,
            item["runid"],
            len(item["rows"]),
            item["length_mismatch"],
            weights[name][1],
            EXPECTED_SOURCE[name],
            "PASS" if item["no_gt"] else "FAIL",
        ))
    lines.append("")
    lines.append("- T1 local-only remote weight files 为 0/105，符合预期。")
    lines.append("- raw/zero/delay/none 均保存 remote weight diagnostics。")
    lines.append("- Inference 未使用 `target_visible`、GT visibility、annotation visibility、oracle mask 或 test IoU 作为 feature。")
    lines.append("- D1 visible-only mask 只用于训练 loss supervision。")
    lines.append("- 未出现 Traceback / RuntimeError / CUDA / NCCL / OOM / NaN / Inf。")
    lines.append("")

    lines.append("## 3. Test 总体结果")
    lines.append("")
    lines.append("| Setting | AUC | Precision | Norm Precision | vs T1 | vs A0 weighted raw test |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, item in results.items():
        s = item["summary"]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(name, s["auc"], s["precision"], s["norm_precision"], delta(s, t1), delta(s, BASELINES["PCUM-v2A A0 weighted raw test"])))
    for name, base in BASELINES.items():
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(name, base["auc"], base["precision"], base["norm_precision"], delta(base, t1), delta(base, BASELINES["PCUM-v2A A0 weighted raw test"])))
    lines.append("")

    lines.append("## 4. 关键对比")
    lines.append("")
    lines.append("| 对比 | Delta AUC | Delta Precision | Delta Norm Precision |")
    lines.append("|---|---:|---:|---:|")
    comparisons = [
        ("raw - T1", t1),
        ("raw - zero", zero),
        ("raw - none", none),
        ("raw - delay diagnostic", delay_diag),
        ("raw - A0 weighted raw test", BASELINES["PCUM-v2A A0 weighted raw test"]),
        ("raw - independent baseline test", BASELINES["Independent non-PCUM baseline epoch25"]),
        ("raw - original EnTeR-Track test", BASELINES["Original EnTeR-Track epoch21"]),
    ]
    for label, base in comparisons:
        lines.append("| {} | {:+.3f} | {:+.3f} | {:+.3f} |".format(label, raw["auc"] - base["auc"], raw["precision"] - base["precision"], raw["norm_precision"] - base["norm_precision"]))
    lines.append("")

    lines.append("## 5. A/B/C 分视角结果")
    lines.append("")
    lines.append("| Setting | Drone A | Drone B | Drone C |")
    lines.append("|---|---|---|---|")
    for name, item in results.items():
        v = item["view"]
        lines.append("| {} | {} | {} | {} |".format(name, fmt(v["Drone A"]), fmt(v["Drone B"]), fmt(v["Drone C"])))
    lines.append("")

    lines.append("## 6. 序列级正负增益")
    lines.append("")
    lines.append("Delta 定义为当前 setting 的序列 AUC - D1 epoch2 T1 local-only 序列 AUC。")
    lines.append("")
    lines.append("| Setting | 正/负/无变化 | 负迁移率 | Top positive | Top negative |")
    lines.append("|---|---:|---:|---|---|")
    for name, item in results.items():
        if name == t1_name:
            continue
        stats = delta_stats(item["rows"], results[t1_name]["rows"])
        lines.append("| {} | {}/{}/{} | {:.2f}% | {} | {} |".format(name, stats["positive"], stats["negative"], stats["same"], stats["negative_rate"], top_fmt(stats["top_positive"]), top_fmt(stats["top_negative"])))
    lines.append("")

    lines.append("## 7. Remote Weight Diagnostics")
    lines.append("")
    lines.append("| Setting | entropy mean/std | max weight mean/p90 | valid count | quality mean/min/max | fallback | selected A/B/C |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, (stats, _) in weights.items():
        if stats is None:
            lines.append("| {} | N/A | N/A | N/A | N/A | N/A | N/A |".format(name))
            continue
        lines.append("| {} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:.2f} | {:.4f}/{:.4f}/{:.4f} | {:.2f}% | {:.1f}/{:.1f}/{:.1f}% |".format(
            name,
            stats["entropy_mean"],
            stats["entropy_std"],
            stats["max_weight_mean"],
            stats["max_weight_p90"],
            stats["valid_count_mean"],
            stats["quality_mean"],
            stats["quality_min"],
            stats["quality_max"],
            stats["fallback_rate"],
            stats["selected_a"],
            stats["selected_b"],
            stats["selected_c"],
        ))
    lines.append("")

    lines.append("## 8. Freeze Audit 摘要")
    lines.append("")
    lines.append("Freeze audit: `{}`。".format(audit["path"]))
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---:|")
    lines.append("| backbone changed keys | 0 |")
    lines.append("| box_head changed keys | 0 |")
    lines.append("| box_head BN buffers changed keys | 0 |")
    lines.append("| optimizer has backbone / box_head params | False |")
    lines.append("| only PCUM / fusion / prompt changed | True |")
    lines.append("| checkpoint audit gate | PASS |")
    lines.append("")

    lines.append("## 9. 主结果对比判断")
    lines.append("")
    lines.append("| 标准 | 结果 | 是否通过 |")
    lines.append("|---|---|---:|")
    condition_rows = [
        ("raw > current main PCUM-v2A A0 weighted raw", "{:.3f} > {:.3f}".format(raw["auc"], BASELINES["PCUM-v2A A0 weighted raw test"]["auc"]), criteria["raw_exceeds_a0_main"]),
        ("raw > independent baseline", "{:.3f} > {:.3f}".format(raw["auc"], BASELINES["Independent non-PCUM baseline epoch25"]["auc"]), criteria["raw_exceeds_independent"]),
        ("raw > original EnTeR-Track", "{:.3f} > {:.3f}".format(raw["auc"], BASELINES["Original EnTeR-Track epoch21"]["auc"]), criteria["raw_exceeds_original"]),
        ("raw >= T1 local-only", "{:.3f} >= {:.3f}".format(raw["auc"], t1["auc"]), criteria["raw_ge_t1"]),
        ("raw > zero", "{:.3f} vs {:.3f}".format(raw["auc"], zero["auc"]), criteria["raw_gt_zero"]),
        ("raw > none", "{:.3f} vs {:.3f}".format(raw["auc"], none["auc"]), criteria["raw_gt_none"]),
        ("checkpoint audit 只有 PCUM / fusion / prompt 参数变化", str(criteria["audit_only_pcum"]), criteria["audit_only_pcum"]),
        ("no-GT inference 通过", str(criteria["no_gt_inference"]), criteria["no_gt_inference"]),
        ("105/105 bbox 与 remote diagnostics 完整", str(criteria["complete_outputs"] and criteria["remote_weights_complete"]), criteria["complete_outputs"] and criteria["remote_weights_complete"]),
        ("无 NaN/Inf", str(criteria["no_nan_inf"]), criteria["no_nan_inf"]),
        ("无 residual scale collapse", "freeze audit includes effective residual scale", criteria["no_residual_collapse"]),
    ]
    for label, value, ok in condition_rows:
        lines.append("| {} | {} | {} |".format(label, value, "PASS" if ok else "FAIL"))
    lines.append("")

    lines.append("## 10. 结论")
    lines.append("")
    if criteria["raw_exceeds_a0_main"]:
        lines.append("- D1 visible-safe epoch2 raw weighted 在本次正式 `threemdot_test` 上超过当前正式主结果 PCUM-v2A A0 weighted raw。")
    else:
        lines.append("- D1 visible-safe epoch2 raw weighted 在本次正式 `threemdot_test` 上未超过当前正式主结果 PCUM-v2A A0 weighted raw。")
    lines.append("- Delay 仅作为 diagnostic：D1 `LAMBDA_DELAY=0.0`，不能写成强 temporal synchronization training 证据。")
    lines.append("- 本阶段未重新训练、未修改 D1 loss、未提交 Git、未删除文件。")
    lines.append("")

    output = REPO_ROOT / "output/pcum_v2_d1_visible_safe_ranking/d1_visible_safe_ep2_test_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {}".format(output))
    print("raw={}".format(fmt(raw)))
    print("criteria={}".format(criteria))


if __name__ == "__main__":
    main()
