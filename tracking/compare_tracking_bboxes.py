import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    reference_dir = Path(args.reference_dir)
    candidate_dir = Path(args.candidate_dir)
    mismatches = []
    for sequence in dataset:
        reference_path = reference_dir / "{}.txt".format(sequence.name)
        candidate_path = candidate_dir / "{}.txt".format(sequence.name)
        if not reference_path.is_file() or not candidate_path.is_file():
            mismatches.append("{}:missing".format(sequence.name))
            continue
        reference = np.asarray(load_text(
            str(reference_path), delimiter=("\t", ","), dtype=np.float64
        ))
        candidate = np.asarray(load_text(
            str(candidate_path), delimiter=("\t", ","), dtype=np.float64
        ))
        if reference.shape != candidate.shape or not np.array_equal(reference, candidate):
            mismatches.append("{}:different".format(sequence.name))

    if mismatches:
        raise RuntimeError(
            "BBox compatibility failed for {} sequences: {}".format(
                len(mismatches), mismatches[:10]
            )
        )
    print("BBox compatibility passed: {}/{} sequences identical".format(
        len(dataset), len(dataset)
    ))


if __name__ == "__main__":
    main()
