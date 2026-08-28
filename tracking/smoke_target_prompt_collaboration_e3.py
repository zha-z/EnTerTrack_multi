"""Two-frame real-checkpoint smoke for E3 on the shortest Three-MDOT val target.

This script does not call the result writer and never evaluates threemdot_test.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.test.evaluation import get_dataset  # noqa: E402
from lib.test.evaluation.running import three_view_triplets  # noqa: E402
from lib.test.evaluation.target_prompt_collaboration import (  # noqa: E402
    run_target_prompt_frame,
)
from lib.test.evaluation.tracker import Tracker  # noqa: E402


def _same_bbox(first, second, atol=1e-6):
    return bool(np.allclose(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
        rtol=0.0, atol=atol))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    torch.cuda.set_device(args.gpu)

    triplets = three_view_triplets(get_dataset("threemdot_val"))
    sequences = min(
        triplets, key=lambda item: max(len(sequence.frames) for sequence in item))
    target_id = sequences[0].name.rsplit("-", 1)[0]
    wrapper = Tracker(
        "entertrack", "target_prompt_collaboration_e3", "threemdot_val",
        run_id="e3_smoke_no_write",
        checkpoint_override=args.checkpoint,
        no_gt_inference=True)
    trackers = []
    for _ in range(3):
        params = wrapper.get_parameters()
        params.debug = 0
        trackers.append(wrapper.create_tracker(params))

    init_images = tuple(wrapper._read_image(sequence.frames[0])
                        for sequence in sequences)
    init_infos = tuple(sequence.init_info() for sequence in sequences)
    for tracker, image, info in zip(trackers, init_images, init_infos):
        tracker.initialize(image, info)

    images = tuple(wrapper._read_image(sequence.frames[1])
                   for sequence in sequences)
    results = run_target_prompt_frame(
        trackers, images, frame_id=1, target_id=target_id)

    checks = {
        "dataset": "threemdot_val",
        "selected_target": target_id,
        "selected_sequences": [sequence.name for sequence in sequences],
        "source_frame_count": [len(sequence.frames) for sequence in sequences],
        "smoke_frames": 2,
        "uses_gt": False,
        "checkpoint": os.path.abspath(args.checkpoint),
        "views": {},
    }
    for view, tracker, result in zip(("A", "B", "C"), trackers, results):
        local = result["local_candidate"]
        collaborative = result["collaborative_candidate"]
        diagnostic = result["output"][
            "target_prompt_collaboration_diagnostics"]
        audit = getattr(tracker.network, "initialization_audit", {})
        view_checks = {
            "strict_b0_core_load": bool(audit.get("strict_full_load", False)),
            "local_bbox_finite": bool(np.isfinite(local["target_bbox"]).all()),
            "state_equals_local_bbox": _same_bbox(
                tracker.state, local["target_bbox"]),
            "reported_equals_e3_bbox": _same_bbox(
                result["output"]["target_bbox"],
                collaborative["target_bbox"]),
            "collaboration_state_identity": (
                diagnostic["persistent_state_digest_before"]
                == diagnostic["persistent_state_digest_after_collaboration"]),
            "next_crop_is_local_state": (
                diagnostic["next_crop_state_digest"]
                == tracker.fcvc_next_crop_digest()),
            "sender_prompt_source_local": (
                diagnostic["sender_prompt_source"] == "local"),
            "reported_output_source_e3": (
                diagnostic["reported_output_source"]
                == "target_prompt_collaboration_e3"),
            "state_output_source_local": (
                diagnostic["state_output_source"] == "local"),
            "one_local_forward": (
                tracker._target_prompt_local_forward_count == 1),
            "prompt_k": diagnostic["prompt_k"],
            "valid_remote_count": diagnostic["valid_remote_count"],
            "payload_fp32_bytes_per_sender": diagnostic[
                "payload_fp32_bytes_per_sender"],
            "payload_fp16_bytes_per_sender": diagnostic[
                "payload_fp16_bytes_per_sender"],
            "center_output_shape": list(
                collaborative["out_dict"]["score_map"].shape),
        }
        boolean_checks = [value for value in view_checks.values()
                          if isinstance(value, bool)]
        view_checks["pass"] = all(boolean_checks) and (
            view_checks["prompt_k"] == 8
            and view_checks["valid_remote_count"] == 2
            and view_checks["center_output_shape"] == [1, 1, 16, 16])
        checks["views"][view] = view_checks
    checks["pass"] = all(item["pass"] for item in checks["views"].values())
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not checks["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
