import argparse
import csv
import os
from collections import OrderedDict

import numpy as np

import _init_paths  # noqa: F401
from lib.test.evaluation.environment import env_settings


VIEWS = OrderedDict([
    ("-1", "Drone A"),
    ("-2", "Drone B"),
    ("-3", "Drone C"),
])


def _result_dir(tracker_name, tracker_param, runid):
    env = env_settings()
    return os.path.join(env.results_path, tracker_name, "%s_%03d" % (tracker_param, int(runid)))


def _view_name(seq_name):
    for suffix, name in VIEWS.items():
        if seq_name.endswith(suffix):
            return name
    return "Unknown"


def _read_decision(path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _empty_bucket():
    return {
        "frames": 0,
        "redetect": 0,
        "has_remote": 0,
        "select_remote": 0,
        "select_local_default": 0,
        "select_local_redetect": 0,
        "fallback_no_remote": 0,
        "fallback_score": 0,
        "fallback_confidence": 0,
        "local_conf_sum": 0.0,
        "remote_conf_sum": 0.0,
        "remote_conf_count": 0,
    }


def _update_bucket(bucket, data):
    if data.size == 0:
        return
    redetect = data[:, 0] > 0.5
    remote_count = data[:, 1]
    selected = data[:, 2].astype(np.int32)
    fallback = data[:, 3].astype(np.int32)
    local_conf = data[:, 4]
    remote_conf = data[:, 6]

    bucket["frames"] += int(data.shape[0])
    bucket["redetect"] += int(redetect.sum())
    bucket["has_remote"] += int((remote_count > 0).sum())
    bucket["select_remote"] += int((selected == 2).sum())
    bucket["select_local_default"] += int((selected == 0).sum())
    bucket["select_local_redetect"] += int((selected == 1).sum())
    bucket["fallback_no_remote"] += int((fallback == 1).sum())
    bucket["fallback_score"] += int((fallback == 2).sum())
    bucket["fallback_confidence"] += int((fallback == 3).sum())
    bucket["local_conf_sum"] += float(local_conf.sum())
    valid_remote = remote_conf >= 0
    bucket["remote_conf_sum"] += float(remote_conf[valid_remote].sum())
    bucket["remote_conf_count"] += int(valid_remote.sum())


def _rate(num, den):
    return float(num) / float(den) if den else 0.0


def _row(name, bucket):
    frames = bucket["frames"]
    return {
        "view": name,
        "frames": frames,
        "redetect": bucket["redetect"],
        "redetect_rate": _rate(bucket["redetect"], frames),
        "has_remote_rate": _rate(bucket["has_remote"], frames),
        "select_remote_rate": _rate(bucket["select_remote"], frames),
        "select_local_redetect_rate": _rate(bucket["select_local_redetect"], frames),
        "fallback_score": bucket["fallback_score"],
        "fallback_confidence": bucket["fallback_confidence"],
        "fallback_no_remote": bucket["fallback_no_remote"],
        "avg_local_conf": _rate(bucket["local_conf_sum"], frames),
        "avg_remote_conf": _rate(bucket["remote_conf_sum"], bucket["remote_conf_count"]),
    }


def analyze(args):
    results_dir = args.results_dir or _result_dir(args.tracker_name, args.tracker_param, args.runid)
    buckets = OrderedDict((name, _empty_bucket()) for name in list(VIEWS.values()) + ["Unknown"])
    totals = _empty_bucket()

    for name in sorted(os.listdir(results_dir)):
        if not name.endswith("_pcum_decision.txt"):
            continue
        seq_name = name[:-len("_pcum_decision.txt")]
        path = os.path.join(results_dir, name)
        data = _read_decision(path)
        bucket = buckets[_view_name(seq_name)]
        _update_bucket(bucket, data)
        _update_bucket(totals, data)

    rows = [_row(name, bucket) for name, bucket in buckets.items() if bucket["frames"] > 0]
    rows.append(_row("All", totals))
    return results_dir, rows


def write_outputs(args, results_dir, rows):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "pcum_decision_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = os.path.join(args.output_dir, "pcum_decision_summary.md")
    with open(report_path, "w") as fh:
        fh.write("# PCUM Decision Summary\n\n")
        fh.write("- Results directory: `%s`\n\n" % os.path.relpath(results_dir))
        fh.write("| View | Frames | Redetect | Redetect rate | Remote selected | Local-redetect selected | Score fallback | Confidence fallback |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fh.write(
                "| {view} | {frames} | {redetect} | {redetect_rate:.4f} | "
                "{select_remote_rate:.4f} | {select_local_redetect_rate:.4f} | "
                "{fallback_score} | {fallback_confidence} |\n".format(**row)
            )
        fh.write("\n")
        fh.write("Decision columns are saved per frame as: redetect flag, remote count, selected source, fallback reason, local confidence, redetect-local confidence, remote confidence, search factor. Selected source: 0 local, 1 local-redetect, 2 remote. Fallback reason: 0 none, 1 no remote, 2 score drop, 3 confidence drop.\n")
    return csv_path, report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize saved PCUM motion-redetect decision logs.")
    parser.add_argument("--tracker_name", default="entertrack")
    parser.add_argument("--tracker_param", default="pcum_ablation_current_full_remote_motion_redetect")
    parser.add_argument("--runid", type=int, default=202)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--output_dir", default="output/analysis/pcum_motion_redetect/decision_run202")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir, rows = analyze(args)
    if not rows or rows[-1]["frames"] == 0:
        raise RuntimeError("No *_pcum_decision.txt files found in %s" % results_dir)
    csv_path, report_path = write_outputs(args, results_dir, rows)
    print("Summary:", csv_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
