#!/usr/bin/env python3
"""Export frame-level samples for offline PCUM reliability selector training."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.test.evaluation.datasets import get_dataset
from tracking.pcum_selector_utils import (
    FEATURE_COLUMNS,
    enhanced_prediction_feature_values,
    get_feature_columns,
    gt_label_loss,
    load_bbox_file,
    load_remote_weight_file,
    load_vector_file,
    prediction_feature_values,
    result_dir,
    validate_feature_columns,
    view_name,
)


def csv_columns(feature_columns):
    return [
    "sequence",
    "frame",
    "uav",
    "local_bbox",
    "collab_bbox",
    ] + list(feature_columns) + [
    "local_loss",
    "collab_loss",
    "loss_delta",
    "label",
    "ignore",
    ]


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def export_split(args):
    feature_columns = get_feature_columns(args.feature_set)
    validate_feature_columns(feature_columns)
    dataset = get_dataset(args.dataset)
    local_dir = result_dir(args.root, args.local_tracker, args.local_runid)
    collab_dir = result_dir(args.root, args.collab_tracker, args.collab_runid)
    if not local_dir.is_dir():
        raise FileNotFoundError(local_dir)
    if not collab_dir.is_dir():
        raise FileNotFoundError(collab_dir)

    rows = []
    incomplete = []
    for seq in dataset:
        local_bbox_path = require_file(local_dir / "{}.txt".format(seq.name))
        collab_bbox_path = require_file(collab_dir / "{}.txt".format(seq.name))
        local_bbox = load_bbox_file(local_bbox_path)
        collab_bbox = load_bbox_file(collab_bbox_path)
        gt_bbox = np.asarray(seq.ground_truth_rect, dtype=np.float64)
        length = gt_bbox.shape[0]
        if local_bbox.shape[0] != length or collab_bbox.shape[0] != length:
            incomplete.append((seq.name, local_bbox.shape[0], collab_bbox.shape[0], length))
            continue

        local_score = load_vector_file(local_dir / "{}_max_score.txt".format(seq.name), length, default=0.0)
        collab_score = load_vector_file(collab_dir / "{}_max_score.txt".format(seq.name), length, default=0.0)
        local_apce = load_vector_file(local_dir / "{}_APCE.txt".format(seq.name), length, default=0.0)
        collab_apce = load_vector_file(collab_dir / "{}_APCE.txt".format(seq.name), length, default=0.0)
        remote_weights = load_remote_weight_file(collab_dir / "{}_pcum_remote_weights.txt".format(seq.name), length)

        for frame in range(length):
            prev_local = local_bbox[frame - 1] if frame > 0 else None
            if args.feature_set == "enhanced":
                prev_collab = collab_bbox[frame - 1] if frame > 0 else None
                prev2_local = local_bbox[frame - 2] if frame > 1 else None
                prev2_collab = collab_bbox[frame - 2] if frame > 1 else None
                prev_remote = remote_weights[frame - 1] if frame > 0 else None
                features = enhanced_prediction_feature_values(
                    local_bbox[frame],
                    collab_bbox[frame],
                    prev_local,
                    prev_collab,
                    prev2_local,
                    prev2_collab,
                    local_score[frame],
                    collab_score[frame],
                    local_apce[frame],
                    collab_apce[frame],
                    remote_weights[frame],
                    prev_remote,
                )
            else:
                features = prediction_feature_values(
                    local_bbox[frame],
                    collab_bbox[frame],
                    prev_local,
                    local_score[frame],
                    collab_score[frame],
                    local_apce[frame],
                    collab_apce[frame],
                    remote_weights[frame],
                )
            local_loss = gt_label_loss(local_bbox[frame], gt_bbox[frame])
            collab_loss = gt_label_loss(collab_bbox[frame], gt_bbox[frame])
            loss_delta = collab_loss - local_loss
            if collab_loss < local_loss - args.margin:
                label = 1
                ignore = 0
            elif local_loss < collab_loss - args.margin:
                label = 0
                ignore = 0
            else:
                label = -1
                ignore = 1
            row = {
                "sequence": seq.name,
                "frame": frame,
                "uav": view_name(seq.name),
                "local_bbox": json.dumps([float(v) for v in local_bbox[frame]]),
                "collab_bbox": json.dumps([float(v) for v in collab_bbox[frame]]),
                "local_loss": "{:.10f}".format(local_loss),
                "collab_loss": "{:.10f}".format(collab_loss),
                "loss_delta": "{:.10f}".format(loss_delta),
                "label": label,
                "ignore": ignore,
            }
            for key in feature_columns:
                value = features[key]
                row[key] = "{:.10f}".format(float(value))
            rows.append(row)

    if incomplete:
        detail = "; ".join("{} local={} collab={} gt={}".format(*item) for item in incomplete[:10])
        raise RuntimeError("Prediction length mismatch: {}".format(detail))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns(feature_columns))
        writer.writeheader()
        writer.writerows(rows)

    usable = [r for r in rows if int(r["ignore"]) == 0]
    positives = sum(1 for r in usable if int(r["label"]) == 1)
    negatives = sum(1 for r in usable if int(r["label"]) == 0)
    print("Exported {} rows to {}".format(len(rows), output_path))
    print("usable={} positive={} negative={} ignored={}".format(
        len(usable), positives, negatives, len(rows) - len(usable)
    ))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--dataset", required=True, help="threemdot_train or threemdot_val")
    parser.add_argument("--local-tracker", required=True)
    parser.add_argument("--local-runid", required=True, type=int)
    parser.add_argument("--collab-tracker", required=True)
    parser.add_argument("--collab-runid", required=True, type=int)
    parser.add_argument("--margin", default=0.02, type=float)
    parser.add_argument("--feature-set", choices=("base", "enhanced"), default="base")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    export_split(parse_args())


if __name__ == "__main__":
    main()
