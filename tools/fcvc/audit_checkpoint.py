#!/usr/bin/env python3
"""Inspect an FCVC checkpoint schema without loading a tracker or dataset."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lib.models.entertrack.fcvc.checkpoint import normalize_fcvc_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    state = normalize_fcvc_state_dict(
        torch.load(str(args.checkpoint), map_location="cpu"))
    student = sum(value.numel() for key, value in state.items() if not key.startswith("teacher."))
    teacher = sum(value.numel() for key, value in state.items() if key.startswith("teacher."))
    print("student_parameter_count={}".format(student))
    print("teacher_parameter_count={}".format(teacher))


if __name__ == "__main__":
    main()
