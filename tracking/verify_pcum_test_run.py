import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings
from lib.test.utils.load_text import load_text


ERROR_PATTERNS = (
    r"Traceback \(most recent call last\)",
    r"RuntimeError:",
    r"No CUDA GPUs are available",
    r"FileNotFoundError",
    r"Error[^\n]*checkpoint",
    r"Error\(s\) in loading state_dict",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runid", required=True, type=int)
    parser.add_argument("--log", required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--expected-checkpoint")
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--expected-aggregation")
    parser.add_argument("--expected-temperature", type=float)
    parser.add_argument("--require-no-pcum", action="store_true")
    parser.add_argument("--require-suppression", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    text = log_path.read_text(errors="replace")
    errors = [pattern for pattern in ERROR_PATTERNS if re.search(pattern, text, re.I)]
    if errors:
        raise RuntimeError("Test log contains errors: {}".format(errors))
    if "Done" not in text:
        raise RuntimeError("Test log does not contain the Done marker")

    dataset = get_dataset(args.dataset)
    expected_groups = len(dataset) // 3
    expected_line = (
        "[PCUM remote state] source={} uses_gt_visibility=false".format(
            args.expected_source
        )
    )
    state_lines = [
        line for line in text.splitlines()
        if line.startswith("[PCUM remote state]")
    ]
    if len(state_lines) != expected_groups:
        raise RuntimeError(
            "Expected {} remote-state log lines, found {}".format(
                expected_groups, len(state_lines)
            )
        )
    unexpected = [line for line in state_lines if expected_line not in line]
    if unexpected:
        raise RuntimeError("Unexpected remote-state log: {}".format(unexpected[0]))
    if args.expected_aggregation is not None:
        aggregation_lines = [
            line for line in text.splitlines()
            if line.startswith("[PCUM remote aggregation]")
        ]
        if len(aggregation_lines) != expected_groups:
            raise RuntimeError(
                "Expected {} aggregation log lines, found {}".format(
                    expected_groups, len(aggregation_lines)
                )
            )
        expected_mode = "mode={}".format(args.expected_aggregation)
        unexpected = [
            line for line in aggregation_lines if expected_mode not in line
        ]
        if unexpected:
            raise RuntimeError("Unexpected aggregation log: {}".format(unexpected[0]))
        if args.expected_temperature is not None:
            expected_temperature = "temperature={}".format(
                args.expected_temperature
            )
            unexpected = [
                line for line in aggregation_lines
                if expected_temperature not in line
            ]
            if unexpected:
                raise RuntimeError(
                    "Unexpected aggregation temperature: {}".format(unexpected[0])
                )

    results_dir = Path(env_settings().results_path) / "entertrack" / (
        "{}_{:03d}".format(args.config, args.runid)
    )
    missing = []
    length_mismatches = []
    for sequence in dataset:
        result_path = results_dir / "{}.txt".format(sequence.name)
        if not result_path.is_file():
            missing.append(sequence.name)
            continue
        prediction = np.asarray(load_text(
            str(result_path), delimiter=("\t", ","), dtype=np.float64
        ))
        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)
        expected_length = len(sequence.frames)
        if prediction.shape[0] != expected_length:
            length_mismatches.append(
                (sequence.name, prediction.shape[0], expected_length)
            )
        if args.expected_aggregation is not None:
            weight_path = results_dir / "{}_pcum_remote_weights.txt".format(
                sequence.name
            )
            if not weight_path.is_file():
                missing.append(sequence.name + ":remote_weights")
            else:
                weights = np.atleast_2d(np.loadtxt(str(weight_path)))
                if weights.shape != (expected_length, 12):
                    length_mismatches.append(
                        (sequence.name + ":remote_weights", weights.shape, (expected_length, 12))
                    )
        if args.require_suppression:
            suppression_path = results_dir / (
                "{}_pcum_remote_suppression.txt".format(sequence.name))
            if not suppression_path.is_file():
                missing.append(sequence.name + ":remote_suppression")
            else:
                suppression = np.atleast_2d(np.loadtxt(str(suppression_path)))
                if suppression.shape != (expected_length, 5):
                    length_mismatches.append((
                        sequence.name + ":remote_suppression",
                        suppression.shape,
                        (expected_length, 5),
                    ))

    if missing:
        raise RuntimeError(
            "Missing bbox results: {} of {} ({})".format(
                len(missing), len(dataset), missing[:5]
            )
        )
    if length_mismatches:
        raise RuntimeError(
            "BBox length mismatches: {}".format(length_mismatches[:5])
        )

    checkpoint_note = "unchecked"
    if args.expected_checkpoint:
        checkpoint_path = Path(args.expected_checkpoint)
        if not checkpoint_path.is_file():
            raise RuntimeError("Checkpoint does not exist: {}".format(checkpoint_path))
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        stored_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        if args.expected_epoch is not None and stored_epoch != args.expected_epoch:
            raise RuntimeError(
                "Checkpoint epoch mismatch: expected {}, found {}".format(
                    args.expected_epoch, stored_epoch
                )
            )
        state = checkpoint.get("net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        pcum_keys = [key for key in state if str(key).startswith("pcum.")]
        if args.require_no_pcum and pcum_keys:
            raise RuntimeError(
                "Checkpoint unexpectedly contains PCUM parameters: {}".format(
                    pcum_keys[:5]
                )
            )
        checkpoint_note = "epoch={},pcum_keys={}".format(stored_epoch, len(pcum_keys))

    print(
        "[VERIFIED] config={} dataset={} runid={} bbox={}/{} source={} no_gt=true checkpoint={}".format(
            args.config,
            args.dataset,
            args.runid,
            len(dataset) - len(missing),
            len(dataset),
            args.expected_source,
            checkpoint_note,
        )
    )


if __name__ == "__main__":
    main()
