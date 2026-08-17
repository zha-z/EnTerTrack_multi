#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np


AUXILIARY_MARKERS = (
    "_time",
    "_max_score",
    "_APCE",
    "_all_boxes",
    "_all_scores",
    "_pcum_",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify that M0 shadow mode does not change bbox or score outputs."
    )
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--shadow-dir", required=True)
    return parser.parse_args()


def relative_files(root):
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def primary_bbox_files(files):
    selected = {}
    for relative, path in files.items():
        if not relative.endswith(".txt"):
            continue
        name = os.path.basename(relative)
        if any(marker in name for marker in AUXILIARY_MARKERS):
            continue
        selected[relative] = path
    return selected


def numeric_diff(first, second):
    first_values = np.loadtxt(str(first), ndmin=2)
    second_values = np.loadtxt(str(second), ndmin=2)
    if first_values.shape != second_values.shape:
        raise AssertionError(
            "shape mismatch: {} {} vs {} {}".format(
                first, first_values.shape, second, second_values.shape
            )
        )
    return float(np.max(np.abs(first_values - second_values))) \
        if first_values.size else 0.0


def main():
    args = parse_args()
    baseline_files = relative_files(args.baseline_dir)
    shadow_files = relative_files(args.shadow_dir)
    baseline_bbox = primary_bbox_files(baseline_files)
    shadow_bbox = primary_bbox_files(shadow_files)
    if set(baseline_bbox) != set(shadow_bbox):
        raise AssertionError("bbox file sets differ")
    if not baseline_bbox:
        raise AssertionError("no bbox files found")

    max_bbox_diff = 0.0
    byte_identical_bbox = 0
    for relative in sorted(baseline_bbox):
        baseline_path = baseline_bbox[relative]
        shadow_path = shadow_bbox[relative]
        if baseline_path.read_bytes() == shadow_path.read_bytes():
            byte_identical_bbox += 1
        max_bbox_diff = max(
            max_bbox_diff,
            numeric_diff(baseline_path, shadow_path),
        )
    if max_bbox_diff != 0.0:
        raise AssertionError("bbox max_abs_diff is {}".format(max_bbox_diff))

    score_relatives = sorted(
        relative for relative in baseline_files if relative.endswith("_max_score.txt")
    )
    if set(score_relatives) != {
        relative for relative in shadow_files if relative.endswith("_max_score.txt")
    }:
        raise AssertionError("max-score file sets differ")
    max_score_diff = 0.0
    for relative in score_relatives:
        max_score_diff = max(
            max_score_diff,
            numeric_diff(baseline_files[relative], shadow_files[relative]),
        )
    if max_score_diff != 0.0:
        raise AssertionError("score max_abs_diff is {}".format(max_score_diff))

    baseline_set = set(baseline_files)
    shadow_set = set(shadow_files)
    missing = sorted(baseline_set - shadow_set)
    extras = sorted(shadow_set - baseline_set)
    invalid_extras = [
        relative for relative in extras
        if not relative.startswith("motion_state_diagnostics/")
    ]
    if missing:
        raise AssertionError("shadow result is missing baseline files: {}".format(missing))
    if invalid_extras:
        raise AssertionError("unexpected non-motion files: {}".format(invalid_extras))
    if not extras:
        raise AssertionError("shadow result contains no new motion diagnostics")

    print("sequence_bbox_files={}".format(len(baseline_bbox)))
    print("byte_identical_bbox_files={}".format(byte_identical_bbox))
    print("bbox_max_abs_diff={:.12g}".format(max_bbox_diff))
    print("score_files={}".format(len(score_relatives)))
    print("score_max_abs_diff={:.12g}".format(max_score_diff))
    print("new_motion_diagnostic_files={}".format(len(extras)))
    print("equivalent=true")


if __name__ == "__main__":
    main()
