import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import _init_paths
import matplotlib.pyplot as plt
import numpy as np

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings


plt.rcParams["figure.figsize"] = [8, 8]


class ResultTracker:
    def __init__(self, name, parameter_name, dataset_name, run_id=None, display_name=None):
        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = "{}/{}/{}".format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = "{}/{}/{}_{:03d}".format(
                env.results_path, self.name, self.parameter_name, self.run_id
            )


def result_trackerlist(name, parameter_name, dataset_name, run_ids=None, display_name=None):
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [
        ResultTracker(name, parameter_name, dataset_name, run_id, display_name)
        for run_id in run_ids
    ]


def has_bbox_results(tracker):
    if not os.path.isdir(tracker.results_dir):
        return False
    for name in os.listdir(tracker.results_dir):
        if not name.endswith(".txt"):
            continue
        if name.endswith("_time.txt") or name.endswith("_max_score.txt") or name.endswith("_APCE.txt"):
            continue
        return True
    return False


def _load_bbox_file(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",", dtype=float)
    except ValueError:
        data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :4]


def _calc_iou_xywh(pred_bb, anno_bb):
    tl = np.maximum(pred_bb[:, :2], anno_bb[:, :2])
    br = np.minimum(
        pred_bb[:, :2] + pred_bb[:, 2:] - 1.0,
        anno_bb[:, :2] + anno_bb[:, 2:] - 1.0,
    )
    wh = np.maximum(br - tl + 1.0, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    pred_area = np.maximum(pred_bb[:, 2], 0.0) * np.maximum(pred_bb[:, 3], 0.0)
    anno_area = np.maximum(anno_bb[:, 2], 0.0) * np.maximum(anno_bb[:, 3], 0.0)
    union = pred_area + anno_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _sequence_view_name(seq_name):
    if seq_name.endswith("-1"):
        return "Drone A"
    if seq_name.endswith("-2"):
        return "Drone B"
    if seq_name.endswith("-3"):
        return "Drone C"
    return "Unknown"


def _success_auc(overlaps):
    overlaps = np.asarray(overlaps, dtype=float)
    overlaps = overlaps[np.isfinite(overlaps)]
    if overlaps.size == 0:
        return float("nan")
    thresholds = np.linspace(0.0, 1.0, 21)
    return float(np.mean([(overlaps >= thr).mean() for thr in thresholds]))


def _all_frame_rows(trackers, dataset):
    rows = []
    for tracker in trackers:
        by_view = {
            "Drone A": [],
            "Drone B": [],
            "Drone C": [],
        }
        missing = 0
        sequence_count = 0

        for seq in dataset:
            sequence_count += 1
            seq_name = getattr(seq, "name", str(seq))
            view_name = _sequence_view_name(seq_name)
            bbox_path = os.path.join(tracker.results_dir, "{}.txt".format(seq_name))
            pred_bb = _load_bbox_file(bbox_path)
            if pred_bb is None:
                missing += 1
                continue

            anno_bb = np.asarray(seq.ground_truth_rect, dtype=float).reshape(-1, 4)
            length = min(pred_bb.shape[0], anno_bb.shape[0])
            if length <= 0:
                missing += 1
                continue

            pred = pred_bb[:length].copy()
            anno = anno_bb[:length].copy()
            pred[0, :] = anno[0, :]

            valid = np.isfinite(anno).all(axis=1) & (anno[:, 2] > 0) & (anno[:, 3] > 0)
            if valid.any() and view_name in by_view:
                overlaps = _calc_iou_xywh(pred[valid], anno[valid])
                by_view[view_name].extend(overlaps.tolist())

        all_overlaps = by_view["Drone A"] + by_view["Drone B"] + by_view["Drone C"]
        rows.append({
            "display": tracker.display_name,
            "param": tracker.parameter_name,
            "run_id": tracker.run_id,
            "sequences": sequence_count,
            "missing": missing,
            "drone_a_auc": _success_auc(by_view["Drone A"]),
            "drone_b_auc": _success_auc(by_view["Drone B"]),
            "drone_c_auc": _success_auc(by_view["Drone C"]),
            "all_auc": _success_auc(all_overlaps),
            "drone_a_frames": len(by_view["Drone A"]),
            "drone_b_frames": len(by_view["Drone B"]),
            "drone_c_frames": len(by_view["Drone C"]),
            "all_frames": len(all_overlaps),
        })
    return rows


def print_all_frame_results(trackers, dataset, csv_path=None):
    rows = _all_frame_rows(trackers, dataset)
    print("\nAll-frame AUC over valid GT frames, without target_visible filtering")
    print("{:<32s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>7s} | {:>7s}".format(
        "Tracker", "Drone A", "Drone B", "Drone C", "All", "Seq", "Missing"
    ))
    print("-" * 99)
    for row in rows:
        print("{:<32s} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>7d} | {:>7d}".format(
            str(row["display"])[:32],
            row["drone_a_auc"] * 100.0,
            row["drone_b_auc"] * 100.0,
            row["drone_c_auc"] * 100.0,
            row["all_auc"] * 100.0,
            row["sequences"],
            row["missing"],
        ))

    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("\nAll-frame CSV: {}".format(csv_path))


dataset_name = "threemdot_test"
dataset = get_dataset(dataset_name)

trackers = []
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_baseline",
    dataset_name=dataset_name,
    run_ids=200,
    display_name="baseline",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_local_view0",
    dataset_name=dataset_name,
    run_ids=200,
    display_name="local_view0",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_local_allviews",
    dataset_name=dataset_name,
    run_ids=200,
    display_name="local_allviews",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_real_target",
    dataset_name=dataset_name,
    run_ids=200,
    display_name="real_target_localtest",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_allviews_equal",
    dataset_name=dataset_name,
    run_ids=200,
    display_name="allviews_equal_localtest",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_a_weight",
    dataset_name=dataset_name,
    run_ids=200,
    display_name="a_weight_localtest",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_dropout",
    dataset_name=dataset_name,
    run_ids=200,
    display_name="dropout_localtest",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_full",
    dataset_name=dataset_name,
    run_ids=200,
    display_name="full_localtest",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_real_target_remote",
    dataset_name=dataset_name,
    run_ids=201,
    display_name="real_target_remote",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_allviews_equal_remote",
    dataset_name=dataset_name,
    run_ids=201,
    display_name="allviews_equal_remote",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_a_weight_remote",
    dataset_name=dataset_name,
    run_ids=201,
    display_name="a_weight_remote",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_dropout_remote",
    dataset_name=dataset_name,
    run_ids=201,
    display_name="dropout_remote",
))
trackers.extend(result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_full_remote",
    dataset_name=dataset_name,
    run_ids=201,
    display_name="full_remote",
))

motion_redetect_tracker = result_trackerlist(
    name="entertrack",
    parameter_name="pcum_ablation_current_full_remote_motion_redetect",
    dataset_name=dataset_name,
    run_ids=202,
    display_name="full_remote_motion_redetect",
)[0]
if has_bbox_results(motion_redetect_tracker):
    trackers.append(motion_redetect_tracker)
else:
    print("Skip missing tracker results:", motion_redetect_tracker.results_dir)


print_results(
    trackers,
    dataset,
    dataset_name,
    merge_results=False,
    plot_types=("success", "norm_prec", "prec"),
    skip_missing_seq=True,
)
print_all_frame_results(
    trackers,
    dataset,
    csv_path="output/analysis/pcum_current_visual/all_frame_auc_by_view_analysis_results.csv",
)
