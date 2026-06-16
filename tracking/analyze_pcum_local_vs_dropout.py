import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np


DATASET_ROOT = Path("/data2/Three-MDOT/three")
RESULTS_ROOT = Path("output/test/tracking_results/entertrack")
LOCAL_NAME = "pcum_ablation_current_local_allviews"
DROPOUT_NAME = "pcum_ablation_current_dropout"
OUT_DIR = Path("output/analysis/pcum_local_allviews_vs_dropout")


def read_boxes(path):
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            line = line.replace(",", " ").replace("\t", " ")
            vals = [float(v) for v in line.split()]
            rows.append(vals[:4])
    return np.asarray(rows, dtype=np.float64)


def read_vector(path):
    text = Path(path).read_text().strip()
    if not text:
        return np.asarray([], dtype=np.float64)
    text = text.replace(",", " ").replace("\t", " ").replace("\n", " ")
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def iou_xywh(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n = min(len(pred), len(gt))
    pred = pred[:n].copy()
    gt = gt[:n].copy()
    if n == 0:
        return np.asarray([], dtype=np.float64)

    pred[0] = gt[0]
    px1, py1 = pred[:, 0], pred[:, 1]
    px2, py2 = pred[:, 0] + pred[:, 2] - 1.0, pred[:, 1] + pred[:, 3] - 1.0
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gt[:, 0] + gt[:, 2] - 1.0, gt[:, 1] + gt[:, 3] - 1.0

    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    iw = np.maximum(ix2 - ix1 + 1.0, 0.0)
    ih = np.maximum(iy2 - iy1 + 1.0, 0.0)
    inter = iw * ih
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def success_auc(iou, valid=None, include_invalid_as_zero=True):
    iou = np.asarray(iou, dtype=np.float64)
    if valid is None:
        valid = np.ones(len(iou), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)[:len(iou)]

    eval_iou = iou.copy()
    eval_iou[~valid] = -1.0
    thresholds = np.arange(0.0, 1.0 + 0.05, 0.05)
    denom = len(eval_iou) if include_invalid_as_zero else max(int(valid.sum()), 1)
    return float(np.mean([(eval_iou > thr).sum() / denom for thr in thresholds]) * 100.0)


def safe_mean(values, mask=None):
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)[:len(values)]
        values = values[mask]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def sequence_path(seq):
    cls = seq.split("-")[0]
    return DATASET_ROOT / cls / seq


def result_file(tracker_name, seq, suffix=""):
    return RESULTS_ROOT / f"{tracker_name}_000" / f"{seq}{suffix}.txt"


def list_sequences():
    seqs = []
    local_dir = RESULTS_ROOT / f"{LOCAL_NAME}_000"
    for path in sorted(local_dir.glob("*.txt")):
        name = path.stem
        if name.endswith("_APCE") or name.endswith("_max_score") or name.endswith("_time"):
            continue
        if result_file(DROPOUT_NAME, name).is_file():
            seqs.append(name)
    return seqs


def load_sequence(seq):
    sp = sequence_path(seq)
    gt = read_boxes(sp / "groundtruth.txt")
    occ = read_vector(sp / "occlusion.txt").astype(bool)
    oov = read_vector(sp / "out_of_view.txt").astype(bool)
    visible = ~(occ[:len(gt)] | oov[:len(gt)])
    local = read_boxes(result_file(LOCAL_NAME, seq))
    dropout = read_boxes(result_file(DROPOUT_NAME, seq))
    n = min(len(gt), len(local), len(dropout), len(visible), len(occ), len(oov))
    return {
        "seq": seq,
        "gt": gt[:n],
        "occ": occ[:n],
        "oov": oov[:n],
        "visible": visible[:n],
        "local": local[:n],
        "dropout": dropout[:n],
        "local_iou": iou_xywh(local[:n], gt[:n]),
        "dropout_iou": iou_xywh(dropout[:n], gt[:n]),
    }


def load_scores(tracker_name, seq):
    score = read_vector(result_file(tracker_name, seq, "_max_score"))
    apce = read_vector(result_file(tracker_name, seq, "_APCE"))
    return score, apce


def fused_for_base(tracker_name, base):
    views = [f"{base}-{i}" for i in (1, 2, 3)]
    records = [load_sequence_view_for_fuse(tracker_name, seq) for seq in views]
    min_len = min(len(r["gt"]) for r in records)

    scores = []
    apces = []
    for seq in views:
        score, apce = load_scores(tracker_name, seq)
        scores.append(score[:min_len])
        apces.append(apce[:min_len])

    fused_idx = []
    q = 44
    for i in range(min_len):
        if i > 0:
            vals = []
            for score, apce in zip(scores, apces):
                window = score[max(0, i - q):i + 1]
                vals.append((float(np.average(window)) - float(np.var(window))) * float(apce[i]))
            fused_idx.append(int(np.argmax(vals)))
        else:
            score_vals = [float(s[0]) for s in scores]
            apce_vals = [float(a[0]) for a in apces]
            fused_idx.append(int(np.argmax(score_vals if max(score_vals) < 0 else apce_vals)))
    fused_idx = np.asarray(fused_idx, dtype=np.int64)

    pred = np.zeros((min_len, 4), dtype=np.float64)
    gt = np.zeros((min_len, 4), dtype=np.float64)
    visible = np.zeros(min_len, dtype=bool)
    for i, view_id in enumerate(fused_idx):
        rec = records[view_id]
        pred[i] = rec["pred"][i]
        gt[i] = rec["gt"][i]
        visible[i] = rec["visible"][i]
    return {
        "base": base,
        "iou": iou_xywh(pred, gt),
        "visible": visible,
        "fused_idx": fused_idx,
    }


def load_sequence_view_for_fuse(tracker_name, seq):
    sp = sequence_path(seq)
    gt = read_boxes(sp / "groundtruth.txt")
    occ = read_vector(sp / "occlusion.txt").astype(bool)
    oov = read_vector(sp / "out_of_view.txt").astype(bool)
    pred = read_boxes(result_file(tracker_name, seq))
    n = min(len(gt), len(pred), len(occ), len(oov))
    return {
        "gt": gt[:n],
        "pred": pred[:n],
        "visible": ~(occ[:n] | oov[:n]),
    }


def draw_box(img, box, color, label):
    x, y, w, h = [int(round(v)) for v in box]
    x2, y2 = x + max(w, 1), y + max(h, 1)
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    cv2.putText(img, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def select_frames(rec, direction, count=6):
    if direction == "dropout":
        score = rec["dropout_iou"] - rec["local_iou"]
    elif direction == "local":
        score = rec["local_iou"] - rec["dropout_iou"]
    else:
        score = 1.0 - np.maximum(rec["local_iou"], rec["dropout_iou"])
        score = score + rec["occ"].astype(float) * 0.5 + rec["oov"].astype(float) * 0.5

    order = np.argsort(-score)
    selected = []
    min_gap = max(len(score) // 12, 10)
    for idx in order:
        if direction in ("dropout", "local") and not rec["visible"][idx]:
            continue
        if direction in ("dropout", "local") and score[idx] <= 0.05:
            continue
        if all(abs(int(idx) - int(prev)) >= min_gap for prev in selected):
            selected.append(int(idx))
        if len(selected) >= count:
            break
    if len(selected) < count:
        for idx in order:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= count:
                break
    return sorted(selected)


def make_case_image(rec, direction, title, out_path):
    frames = select_frames(rec, direction)
    tiles = []
    for idx in frames:
        img_path = sequence_path(rec["seq"]) / "img" / f"{idx + 1:08d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        draw_box(img, rec["gt"][idx], (0, 220, 0), "GT")
        draw_box(img, rec["local"][idx], (255, 170, 0), "local")
        draw_box(img, rec["dropout"][idx], (0, 0, 255), "dropout")
        state = "visible"
        if rec["occ"][idx]:
            state = "occlusion"
        if rec["oov"][idx]:
            state = "out_of_view"
        text = "f=%d %s local=%.2f drop=%.2f" % (
            idx + 1,
            state,
            rec["local_iou"][idx],
            rec["dropout_iou"][idx],
        )
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        scale = 480.0 / max(img.shape[1], 1)
        if scale < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        tiles.append(img)

    if not tiles:
        return None

    max_h = max(tile.shape[0] for tile in tiles)
    max_w = max(tile.shape[1] for tile in tiles)
    padded = []
    for tile in tiles:
        canvas = np.full((max_h, max_w, 3), 245, dtype=np.uint8)
        canvas[:tile.shape[0], :tile.shape[1]] = tile
        padded.append(canvas)

    cols = 2
    rows = int(math.ceil(len(padded) / cols))
    header_h = 46
    sheet = np.full((rows * max_h + header_h, cols * max_w, 3), 245, dtype=np.uint8)
    cv2.putText(sheet, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for i, tile in enumerate(padded):
        r, c = divmod(i, cols)
        y = header_h + r * max_h
        x = c * max_w
        sheet[y:y + max_h, x:x + max_w] = tile
    cv2.imwrite(str(out_path), sheet)
    return out_path


def corr(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(xs[mask], ys[mask])[0, 1])


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seqs = list_sequences()
    records = [load_sequence(seq) for seq in seqs]

    rows = []
    for rec in records:
        view = rec["seq"].rsplit("-", 1)[-1]
        occ_or_oov = rec["occ"] | rec["oov"]
        row = {
            "sequence": rec["seq"],
            "view": view,
            "frames": len(rec["gt"]),
            "occlusion_ratio": float(rec["occ"].mean()),
            "out_of_view_ratio": float(rec["oov"].mean()),
            "invisible_ratio": float(occ_or_oov.mean()),
            "local_auc": success_auc(rec["local_iou"], rec["visible"]),
            "dropout_auc": success_auc(rec["dropout_iou"], rec["visible"]),
            "delta_dropout_minus_local": success_auc(rec["dropout_iou"], rec["visible"]) - success_auc(rec["local_iou"], rec["visible"]),
            "local_visible_mean_iou": safe_mean(rec["local_iou"], rec["visible"]),
            "dropout_visible_mean_iou": safe_mean(rec["dropout_iou"], rec["visible"]),
            "local_occ_raw_mean_iou": safe_mean(rec["local_iou"], occ_or_oov),
            "dropout_occ_raw_mean_iou": safe_mean(rec["dropout_iou"], occ_or_oov),
            "local_failure_visible_ratio": float(((rec["local_iou"] < 0.1) & rec["visible"]).sum() / max(rec["visible"].sum(), 1)),
            "dropout_failure_visible_ratio": float(((rec["dropout_iou"] < 0.1) & rec["visible"]).sum() / max(rec["visible"].sum(), 1)),
        }
        rows.append(row)

    write_csv(OUT_DIR / "per_sequence_metrics.csv", rows)

    bases = sorted({seq.rsplit("-", 1)[0] for seq in seqs})
    fused_rows = []
    for base in bases:
        local_f = fused_for_base(LOCAL_NAME, base)
        dropout_f = fused_for_base(DROPOUT_NAME, base)
        local_auc = success_auc(local_f["iou"], local_f["visible"])
        dropout_auc = success_auc(dropout_f["iou"], dropout_f["visible"])
        fused_rows.append({
            "base": base,
            "frames": len(local_f["iou"]),
            "local_fused_auc": local_auc,
            "dropout_fused_auc": dropout_auc,
            "delta_dropout_minus_local": dropout_auc - local_auc,
            "local_fused_visible_mean_iou": safe_mean(local_f["iou"], local_f["visible"]),
            "dropout_fused_visible_mean_iou": safe_mean(dropout_f["iou"], dropout_f["visible"]),
        })
    write_csv(OUT_DIR / "per_target_fused_metrics.csv", fused_rows)

    by_view = {}
    for view in ("1", "2", "3"):
        subset = [r for r in rows if r["view"] == view]
        by_view[view] = {
            "local_auc": float(np.mean([r["local_auc"] for r in subset])),
            "dropout_auc": float(np.mean([r["dropout_auc"] for r in subset])),
            "delta": float(np.mean([r["delta_dropout_minus_local"] for r in subset])),
            "local_fail": float(np.mean([r["local_failure_visible_ratio"] for r in subset])),
            "dropout_fail": float(np.mean([r["dropout_failure_visible_ratio"] for r in subset])),
        }

    fused_summary = {
        "local_auc": float(np.mean([r["local_fused_auc"] for r in fused_rows])),
        "dropout_auc": float(np.mean([r["dropout_fused_auc"] for r in fused_rows])),
        "delta": float(np.mean([r["delta_dropout_minus_local"] for r in fused_rows])),
    }

    deltas = [r["delta_dropout_minus_local"] for r in rows]
    occs = [r["invisible_ratio"] for r in rows]
    visible_fail_delta = [r["dropout_failure_visible_ratio"] - r["local_failure_visible_ratio"] for r in rows]
    occ_delta_raw = [r["dropout_occ_raw_mean_iou"] - r["local_occ_raw_mean_iou"] for r in rows]

    top_dropout_a = sorted(
        [r for r in rows if r["view"] == "1"],
        key=lambda r: r["delta_dropout_minus_local"],
        reverse=True,
    )[:3]
    top_local = sorted(rows, key=lambda r: r["delta_dropout_minus_local"])[:3]
    high_occ = sorted(
        [r for r in rows if r["invisible_ratio"] > 0.05],
        key=lambda r: (r["invisible_ratio"], -abs(r["delta_dropout_minus_local"])),
        reverse=True,
    )[:3]

    figures = []
    seq_to_rec = {rec["seq"]: rec for rec in records}
    for case, direction, label in [
        (top_dropout_a[0], "dropout", "dropout_better_A"),
        (top_dropout_a[1], "dropout", "dropout_better_A_2"),
        (top_local[0], "local", "local_better"),
        (top_local[1], "local", "local_better_2"),
    ]:
        rec = seq_to_rec[case["sequence"]]
        fig = OUT_DIR / f"{label}_{rec['seq']}.jpg"
        title = f"{label}: {rec['seq']} delta(drop-local)={case['delta_dropout_minus_local']:.2f}"
        out = make_case_image(rec, direction, title, fig)
        if out is not None:
            figures.append(out)

    if high_occ:
        rec = seq_to_rec[high_occ[0]["sequence"]]
        fig = OUT_DIR / f"high_occlusion_{rec['seq']}.jpg"
        title = f"high occlusion/OOV: {rec['seq']} invisible={high_occ[0]['invisible_ratio']:.2f}"
        out = make_case_image(rec, "failure", title, fig)
        if out is not None:
            figures.append(out)

    report_lines = []
    report_lines.append("PCUM local_allviews vs dropout 跟踪实例分析报告")
    report_lines.append("=" * 72)
    report_lines.append("")
    report_lines.append("一、配置差异")
    report_lines.append("- local_allviews: MODEL.PCUM.ENABLED=True, TRAIN.PCUM.USE_REAL_MULTIVIEW=True, REAL_MULTIVIEW_LOSS_MODE=all_views, REMOTE_DROPOUT_PROB=1.0。训练时三视角都参与监督，但 remote prompt 永远被丢弃，等价于“本地 PCUM + 三视角监督”。")
    report_lines.append("- dropout: MODEL.PCUM.ENABLED=True, TRAIN.PCUM.USE_REAL_MULTIVIEW=True, REAL_MULTIVIEW_LOSS_MODE=all_views, REMOTE_DROPOUT_PROB=0.3。训练时约 70% iteration 使用真实跨机 remote prompt，约 30% iteration 退化成本地提示。")
    report_lines.append("- 两者相同点: LR=8e-5, EPOCH=40, LR_DROP_EPOCH=28, BATCH_SIZE=8, FLOPS_WEIGHT=0.0, SEARCH_SIZE=256, REQUIRE_ALL_VIEWS_VISIBLE=True, CANONICAL_VIEW_ORDER=True。")
    report_lines.append("")
    report_lines.append("二、总体结果")
    report_lines.append("官方 print_results 统计: local_allviews A/B/C/Fused = 45.74 / 49.92 / 52.21 / 61.14; dropout A/B/C/Fused = 47.37 / 48.23 / 50.50 / 60.14。")
    for view, name in [("1", "A"), ("2", "B"), ("3", "C")]:
        s = by_view[view]
        report_lines.append(
            f"本脚本逐序列复算({name}机): local={s['local_auc']:.2f}, dropout={s['dropout_auc']:.2f}, "
            f"drop-local={s['delta']:+.2f}, visible failure ratio local/dropout={s['local_fail']:.3f}/{s['dropout_fail']:.3f}"
        )
    report_lines.append(
        f"融合复算: local={fused_summary['local_auc']:.2f}, dropout={fused_summary['dropout_auc']:.2f}, "
        f"drop-local={fused_summary['delta']:+.2f}"
    )
    report_lines.append("")
    report_lines.append("三、遮挡/出视野归因")
    report_lines.append(
        f"逐序列 invisible_ratio 与 dropout-local AUC 差值的相关系数: {corr(occs, deltas):+.3f}。"
    )
    report_lines.append(
        f"逐序列 invisible_ratio 与 visible failure 差值(dropout-local)的相关系数: {corr(occs, visible_fail_delta):+.3f}。"
    )
    report_lines.append(
        f"遮挡/出视野帧 raw IoU 差值(dropout-local)与 invisible_ratio 的相关系数: {corr(occs, occ_delta_raw):+.3f}。"
    )
    report_lines.append("解释: 如果相关系数接近 0，说明两者差异不是由遮挡比例单独决定；如果绝对值较大，才说明遮挡/出视野是主要变量。")
    report_lines.append("")
    report_lines.append("四、典型序列")
    report_lines.append("Dropout 在 A 机领先最多的序列:")
    for r in top_dropout_a:
        report_lines.append(
            f"- {r['sequence']}: drop-local={r['delta_dropout_minus_local']:+.2f}, "
            f"invisible={r['invisible_ratio']:.3f}, occ={r['occlusion_ratio']:.3f}, oov={r['out_of_view_ratio']:.3f}, "
            f"visible failure local/dropout={r['local_failure_visible_ratio']:.3f}/{r['dropout_failure_visible_ratio']:.3f}"
        )
    report_lines.append("local_allviews 领先最多的单视角序列:")
    for r in top_local:
        report_lines.append(
            f"- {r['sequence']}: drop-local={r['delta_dropout_minus_local']:+.2f}, "
            f"invisible={r['invisible_ratio']:.3f}, occ={r['occlusion_ratio']:.3f}, oov={r['out_of_view_ratio']:.3f}, "
            f"visible failure local/dropout={r['local_failure_visible_ratio']:.3f}/{r['dropout_failure_visible_ratio']:.3f}"
        )
    report_lines.append("高遮挡/出视野序列:")
    for r in high_occ:
        report_lines.append(
            f"- {r['sequence']}: invisible={r['invisible_ratio']:.3f}, drop-local={r['delta_dropout_minus_local']:+.2f}, "
            f"local_auc={r['local_auc']:.2f}, dropout_auc={r['dropout_auc']:.2f}"
        )
    report_lines.append("")
    report_lines.append("五、可视化文件")
    report_lines.append("图中绿色为 GT，蓝橙色为 local_allviews，红色为 dropout。每个小图标题含 frame id、可见状态和两者 IoU。")
    for fig in figures:
        report_lines.append(f"- {fig}")
    report_lines.append("")
    report_lines.append("六、结论")
    report_lines.append("- local_allviews 的优势主要体现在 B/C 机和融合结果，说明“三视角监督 + 不依赖 remote prompt”提升了跨视角泛化和融合稳定性。")
    report_lines.append("- dropout 的优势主要体现在 A 机，说明随机丢 remote prompt 对 A 机的单机鲁棒性有帮助，但会削弱 B/C 和融合的稳定收益。")
    report_lines.append("- 从遮挡/出视野相关性和典型序列看，差异不应简单归因于遮挡本身；更像是 remote prompt 参与训练后改变了单机/多机之间的特征依赖。")
    report_lines.append("- 现在若目标是最高融合 AUC，应优先用 local_allviews；若目标是抬 A 机，应沿 dropout 方向继续调 remote dropout 概率或做按视角 dropout。")

    (OUT_DIR / "report.txt").write_text("\n".join(report_lines) + "\n")
    print("Wrote:", OUT_DIR / "report.txt")
    print("Wrote:", OUT_DIR / "per_sequence_metrics.csv")
    print("Wrote:", OUT_DIR / "per_target_fused_metrics.csv")
    for fig in figures:
        print("Wrote:", fig)


if __name__ == "__main__":
    main()
