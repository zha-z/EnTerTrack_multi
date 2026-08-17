#!/usr/bin/env python3
"""Verify baseline-vs-MCR-shadow bbox and max-score equivalence."""

import argparse
import os
from pathlib import Path

import numpy as np


AUXILIARY_MARKERS = (
    "_time", "_max_score", "_APCE", "_all_boxes", "_all_scores", "_pcum_",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Require byte-identical primary bbox and max-score files in MCR shadow mode.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--shadow-dir", required=True)
    return parser.parse_args()


def files_under(root):
    root = Path(root)
    return {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}


def primary_bbox_files(files):
    return {
        relative: path for relative, path in files.items()
        if relative.endswith(".txt")
        and not any(marker in os.path.basename(relative) for marker in AUXILIARY_MARKERS)
    }


def require_byte_identical(first, second, label):
    if set(first) != set(second):
        raise AssertionError("{} file sets differ".format(label))
    if not first:
        raise AssertionError("no {} files found".format(label))
    for relative in sorted(first):
        if first[relative].read_bytes() != second[relative].read_bytes():
            first_values = np.loadtxt(str(first[relative]), ndmin=2)
            second_values = np.loadtxt(str(second[relative]), ndmin=2)
            if first_values.shape != second_values.shape:
                detail = "shape {} vs {}".format(first_values.shape, second_values.shape)
            else:
                detail = "max_abs_diff={}".format(
                    float(np.max(np.abs(first_values - second_values))))
            raise AssertionError("{} differs: {} ({})".format(label, relative, detail))


def main():
    args = parse_args()
    baseline = files_under(args.baseline_dir)
    shadow = files_under(args.shadow_dir)
    require_byte_identical(primary_bbox_files(baseline), primary_bbox_files(shadow), "bbox")
    baseline_scores = {
        relative: path for relative, path in baseline.items()
        if relative.endswith("_max_score.txt")
    }
    shadow_scores = {
        relative: path for relative, path in shadow.items()
        if relative.endswith("_max_score.txt")
    }
    require_byte_identical(baseline_scores, shadow_scores, "max-score")
    diagnostics = [relative for relative in shadow if relative.startswith("mcr_diagnostics/")]
    if not diagnostics:
        raise AssertionError("shadow result contains no mcr_diagnostics files")
    print("bbox_files={}".format(len(primary_bbox_files(baseline))))
    print("score_files={}".format(len(baseline_scores)))
    print("mcr_diagnostic_files={}".format(len(diagnostics)))
    print("byte_identical=true")


if __name__ == "__main__":
    main()
