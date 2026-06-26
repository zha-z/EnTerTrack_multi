import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np

import _init_paths  # noqa: F401
from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.tracker import _pcum_motion_reliability


VIEWS = OrderedDict([
    ("-1", "Drone A"),
    ("-2", "Drone B"),
    ("-3", "Drone C"),
])


def _load_array(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except ValueError:
        data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def _load_bbox(path):
    data = _load_array(path)
    if data is None:
        return None
    return data[:, :4]


def _load_vector(path):
    data = _load_array(path)
    if data is None:
        return None
    return data.reshape(-1)


def _iou_xywh(pred, gt):
    tl = np.maximum(pred[:, :2], gt[:, :2])
    br = np.minimum(pred[:, :2] + pred[:, 2:] - 1.0, gt[:, :2] + gt[:, 2:] - 1.0)
    wh = np.maximum(br - tl + 1.0, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    pred_area = np.maximum(pred[:, 2], 0.0) * np.maximum(pred[:, 3], 0.0)
    gt_area = np.maximum(gt[:, 2], 0.0) * np.maximum(gt[:, 3], 0.0)
    union = pred_area + gt_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _target_id(seq_name):
    return seq_name.rsplit("-", 1)[0]


def _view_suffix(seq_name):
    return "-" + seq_name.rsplit("-", 1)[1]


def _group_sequences(dataset):
    groups = OrderedDict()
    for seq in dataset:
        target = _target_id(seq.name)
        groups.setdefault(target, {})[_view_suffix(seq.name)] = seq
    return groups


def _result_dir(tracker_name, tracker_param, runid):
    env = env_settings()
    return os.path.join(env.results_path, tracker_name, "%s_%03d" % (tracker_param, int(runid)))


def _read_sequence_result(results_dir, seq):
    seq_name = seq.name
    bbox = _load_bbox(os.path.join(results_dir, seq_name + ".txt"))
    score = _load_vector(os.path.join(results_dir, seq_name + "_max_score.txt"))
    apce = _load_vector(os.path.join(results_dir, seq_name + "_APCE.txt"))
    if bbox is None or score is None or apce is None:
        return None

    gt = np.asarray(seq.ground_truth_rect, dtype=np.float32).reshape(-1, 4)
    visible = np.asarray(getattr(seq, "target_visible", np.ones(len(gt), dtype=bool))).astype(bool)
    length = min(len(bbox), len(score), len(apce), len(gt), len(visible))
    if length < 2:
        return None

    bbox = bbox[:length].astype(np.float32)
    gt = gt[:length].astype(np.float32)
    score = score[:length].astype(np.float32)
    apce = apce[:length].astype(np.float32)
    visible = visible[:length]
    bbox[0] = gt[0]
    valid = np.isfinite(gt).all(axis=1) & (gt[:, 2] > 0) & (gt[:, 3] > 0)
    iou = np.zeros(length, dtype=np.float32)
    if valid.any():
        iou[valid] = _iou_xywh(bbox[valid], gt[valid])
    return {
        "bbox": bbox,
        "score": score,
        "apce": apce,
        "visible": visible,
        "valid": valid,
        "iou": iou,
    }


def _reliability(seq_result, frame_id, max_norm_motion, apce_norm):
    if seq_result is None or frame_id <= 0 or frame_id >= len(seq_result["bbox"]):
        return 0.0
    candidate = {
        "prev_bbox": seq_result["bbox"][frame_id - 1].tolist(),
        "target_bbox": seq_result["bbox"][frame_id].tolist(),
        "max_score": float(seq_result["score"][frame_id]),
        "apce": float(seq_result["apce"][frame_id]),
        "visible": bool(seq_result["visible"][frame_id]),
    }
    return _pcum_motion_reliability(
        candidate,
        max_norm_motion=max_norm_motion,
        apce_norm=apce_norm,
    )


def _is_local_low(score, apce, score_thr, apce_thr, mode):
    low_score = score < score_thr
    low_apce = apce < apce_thr
    mode = str(mode).lower()
    if mode == "and":
        return low_score and low_apce
    if mode == "score":
        return low_score
    if mode == "apce":
        return low_apce
    return low_score or low_apce


def analyze(args):
    dataset = get_dataset(args.dataset_name)
    groups = _group_sequences(dataset)
    results_dir = args.results_dir or _result_dir(args.tracker_name, args.tracker_param, args.runid)

    rows = []
    totals = {
        "frames": 0,
        "valid_frames": 0,
        "local_low": 0,
        "failure": 0,
        "trigger": 0,
        "trigger_failure": 0,
        "visible_trigger": 0,
        "invisible_trigger": 0,
    }
    per_view = OrderedDict((name, dict(totals)) for name in VIEWS.values())

    for target, seq_by_view in groups.items():
        results_by_view = {
            suffix: _read_sequence_result(results_dir, seq_by_view[suffix])
            for suffix in VIEWS
            if suffix in seq_by_view
        }
        if len(results_by_view) < 3:
            continue

        min_len = min(len(result["bbox"]) for result in results_by_view.values() if result is not None)
        for frame_id in range(1, min_len):
            reliable_by_view = {
                suffix: _reliability(
                    result,
                    frame_id,
                    max_norm_motion=args.max_norm_motion,
                    apce_norm=args.apce_norm,
                )
                for suffix, result in results_by_view.items()
            }

            for suffix, result in results_by_view.items():
                view_name = VIEWS[suffix]
                if result is None or not bool(result["valid"][frame_id]):
                    continue
                score = float(result["score"][frame_id])
                apce = float(result["apce"][frame_id])
                visible = bool(result["visible"][frame_id])
                iou = float(result["iou"][frame_id])
                local_low = _is_local_low(
                    score,
                    apce,
                    args.local_low_score,
                    args.local_low_apce,
                    args.local_low_mode,
                )
                failure = iou < args.failure_iou
                peer_scores = [
                    reliable_by_view[peer_suffix]
                    for peer_suffix in VIEWS
                    if peer_suffix != suffix and peer_suffix in reliable_by_view
                ]
                reliable_peers = [value for value in peer_scores if value >= args.min_reliability]
                trigger = local_low and len(reliable_peers) >= args.min_remote

                for bucket in (totals, per_view[view_name]):
                    bucket["frames"] += 1
                    bucket["valid_frames"] += 1
                    bucket["local_low"] += int(local_low)
                    bucket["failure"] += int(failure)
                    bucket["trigger"] += int(trigger)
                    bucket["trigger_failure"] += int(trigger and failure)
                    bucket["visible_trigger"] += int(trigger and visible)
                    bucket["invisible_trigger"] += int(trigger and not visible)

                if trigger:
                    rows.append({
                        "target": target,
                        "view": view_name,
                        "frame": frame_id,
                        "score": score,
                        "apce": apce,
                        "iou": iou,
                        "visible": int(visible),
                        "failure": int(failure),
                        "peer_reliability_mean": float(np.mean(peer_scores)) if peer_scores else 0.0,
                        "peer_reliability_max": float(np.max(peer_scores)) if peer_scores else 0.0,
                    })

    return results_dir, rows, totals, per_view


def _rate(num, den):
    return float(num) / float(den) if den else 0.0


def _summary_row(name, data):
    return {
        "view": name,
        "frames": data["frames"],
        "local_low": data["local_low"],
        "failure": data["failure"],
        "trigger": data["trigger"],
        "trigger_failure": data["trigger_failure"],
        "trigger_rate": _rate(data["trigger"], data["frames"]),
        "failure_coverage": _rate(data["trigger_failure"], data["failure"]),
        "trigger_failure_ratio": _rate(data["trigger_failure"], data["trigger"]),
        "visible_trigger": data["visible_trigger"],
        "invisible_trigger": data["invisible_trigger"],
    }


def write_outputs(args, results_dir, trigger_rows, totals, per_view):
    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = [_summary_row(name, data) for name, data in per_view.items()]
    summary_rows.append(_summary_row("All", totals))

    summary_path = os.path.join(args.output_dir, "motion_redetect_trigger_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    trigger_path = os.path.join(args.output_dir, "motion_redetect_trigger_frames.csv")
    if trigger_rows:
        with open(trigger_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(trigger_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trigger_rows)

    report_path = os.path.join(args.output_dir, "motion_redetect_trigger_report.md")
    with open(report_path, "w") as fh:
        fh.write("# PCUM Motion-Redetect Trigger Diagnosis\n\n")
        fh.write("- Dataset: `%s`\n" % args.dataset_name)
        fh.write("- Tracker param: `%s`\n" % args.tracker_param)
        fh.write("- Run ID: `%s`\n" % args.runid)
        fh.write("- Results directory: `%s`\n" % os.path.relpath(results_dir))
        fh.write("- Local low thresholds: score < %.3f, APCE < %.3f\n" % (
            args.local_low_score, args.local_low_apce))
        fh.write("- Local low mode: `%s`\n" % args.local_low_mode)
        fh.write("- Motion reliability threshold: %.3f\n" % args.min_reliability)
        fh.write("- Failure proxy: IoU < %.2f\n\n" % args.failure_iou)
        fh.write("| View | Frames | Low-conf | Failures | Triggers | Triggered failures | Trigger rate | Failure coverage | Trigger failure ratio |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            fh.write(
                "| {view} | {frames} | {local_low} | {failure} | {trigger} | {trigger_failure} | "
                "{trigger_rate:.4f} | {failure_coverage:.4f} | {trigger_failure_ratio:.4f} |\n".format(**row)
            )
        fh.write("\n")
        fh.write("This is an offline diagnostic. It estimates where the test-time motion-redetect gate would trigger from saved boxes, scores, APCE, and visibility flags. It does not prove final AUC improvement; run `tracking/test.py` with the motion-redetect YAML for that.\n")

    return summary_path, trigger_path if trigger_rows else None, report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PCUM motion-redetect trigger coverage from saved results.")
    parser.add_argument("--dataset_name", default="threemdot_test")
    parser.add_argument("--tracker_name", default="entertrack")
    parser.add_argument("--tracker_param", default="pcum_ablation_current_full_remote")
    parser.add_argument("--runid", type=int, default=201)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--output_dir", default="output/analysis/pcum_motion_redetect")
    parser.add_argument("--local_low_score", type=float, default=0.25)
    parser.add_argument("--local_low_apce", type=float, default=100.0)
    parser.add_argument("--local_low_mode", default="or", choices=("or", "and", "score", "apce"))
    parser.add_argument("--min_remote", type=int, default=1)
    parser.add_argument("--min_reliability", type=float, default=0.12)
    parser.add_argument("--max_norm_motion", type=float, default=2.0)
    parser.add_argument("--apce_norm", type=float, default=200.0)
    parser.add_argument("--failure_iou", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir, trigger_rows, totals, per_view = analyze(args)
    summary_path, trigger_path, report_path = write_outputs(args, results_dir, trigger_rows, totals, per_view)
    print("Summary:", summary_path)
    if trigger_path:
        print("Trigger frames:", trigger_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
