#!/usr/bin/env python3
"""Read-only validation of the frozen FCVC receiver manifest."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.train_fcvc_full import read_manifest, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    print("receiver_case_count={}".format(len(rows)))
    print("manifest_sha256={}".format(sha256_file(args.manifest)))


if __name__ == "__main__":
    main()
