"""Prediction-only E3-U1 sender counterfactual rollout on Three-MDOT val.

This runner never reads frame GT after the ordinary first-frame tracker
initialization.  It writes the two frozen prediction artifacts atomically and
then records their SHA256 hashes in a manifest before any post-hoc GT join.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.evaluation import get_dataset  # noqa: E402
from lib.test.evaluation.running import three_view_triplets  # noqa: E402
from lib.test.evaluation.tracker import Tracker  # noqa: E402


BRANCHES = ("local", "sender0_only", "sender1_only", "both")
VIEWS = ("A", "B", "C")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def _write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, lineterminator="\n",
            extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _bbox_values(box):
    return [float(value) for value in box]


def _same_bbox(first, second, atol=1e-6):
    return bool(np.allclose(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64), rtol=0.0, atol=atol))


def _bbox_disagreement(local, branch):
    local = np.asarray(local, dtype=np.float64)
    branch = np.asarray(branch, dtype=np.float64)
    local_center = local[:2] + 0.5 * local[2:]
    branch_center = branch[:2] + 0.5 * branch[2:]
    displacement = float(np.linalg.norm(branch_center - local_center))
    local_area = max(float(local[2] * local[3]), 1e-12)
    branch_area = max(float(branch[2] * branch[3]), 1e-12)
    return displacement, float(math.log(branch_area / local_area))


def _masked_remote_candidates(candidates, keep_index):
    output = []
    for index, candidate in enumerate(candidates):
        cloned = dict(candidate)
        packet = dict(candidate["target_prompt_packet"])
        original = packet["valid"]
        packet["valid"] = torch.full_like(
            original, bool(index == keep_index), dtype=torch.bool)
        cloned["target_prompt_packet"] = packet
        output.append(cloned)
    return tuple(output)


def _prompt_metadata(tracker, candidate):
    feature = candidate["out_dict"]["backbone_feat"]
    score = candidate["out_dict"]["score_map"]
    search = feature[:, -int(tracker.network.feat_len_s):]
    with torch.no_grad():
        result = tracker.network.target_prompt_extractor.extract_with_metadata(
            search.detach(), score.detach())
    packet_prompt = candidate["target_prompt_packet"]["prompt"]
    if not torch.equal(result["prompt"], packet_prompt):
        raise RuntimeError("recomputed prompt does not match local packet")
    return result


def _summary_stats(values):
    values = values.detach().float().reshape(-1)
    return {
        "mean": float(values.mean().cpu().item()),
        "std": float(values.std(unbiased=False).cpu().item()),
        "min": float(values.min().cpu().item()),
        "max": float(values.max().cpu().item()),
    }


def _prompt_stats(metadata):
    prompt = metadata["prompt"][0].detach().float()
    scores = metadata["topk_scores"][0].detach().float()
    score_stats = _summary_stats(scores)
    norms = prompt.norm(dim=1)
    norm_stats = _summary_stats(norms)
    normalized = torch.nn.functional.normalize(prompt, p=2, dim=1, eps=1e-12)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = cosine[~torch.eye(
        cosine.shape[0], device=cosine.device, dtype=torch.bool)]
    cosine_stats = _summary_stats(off_diagonal)
    return {
        "prompt_topk_score_mean": score_stats["mean"],
        "prompt_topk_score_std": score_stats["std"],
        "prompt_topk_score_min": score_stats["min"],
        "prompt_topk_score_max": score_stats["max"],
        "prompt_top1_top8_gap": score_stats["max"] - score_stats["min"],
        "prompt_norm_mean": norm_stats["mean"],
        "prompt_norm_std": norm_stats["std"],
        "prompt_pairwise_cos_mean": cosine_stats["mean"],
        "prompt_pairwise_cos_std": cosine_stats["std"],
        "prompt_pairwise_cos_min": cosine_stats["min"],
        "prompt_pairwise_cos_max": cosine_stats["max"],
    }


def _set_compatibility(receiver_prompt, sender_prompt):
    receiver = torch.nn.functional.normalize(
        receiver_prompt[0].detach().float(), p=2, dim=1, eps=1e-12)
    sender = torch.nn.functional.normalize(
        sender_prompt[0].detach().float(), p=2, dim=1, eps=1e-12)
    matrix = receiver @ sender.transpose(0, 1)
    sender_best = matrix.max(dim=0).values.mean()
    receiver_best = matrix.max(dim=1).values.mean()
    return {
        "set_cos_mean": float(matrix.mean().cpu().item()),
        "set_cos_max": float(matrix.max().cpu().item()),
        "sender_to_receiver_best_mean": float(sender_best.cpu().item()),
        "receiver_to_sender_best_mean": float(receiver_best.cpu().item()),
        "symmetric_best_match": float(
            (0.5 * (sender_best + receiver_best)).cpu().item()),
    }


def _branch_row(sequence_name, target_id, receiver_view, frame_id,
                branch_name, local_candidate, branch_candidate,
                selected_senders, local_forward_delta):
    local_box = _bbox_values(local_candidate["target_bbox"])
    branch_box = _bbox_values(branch_candidate["target_bbox"])
    displacement, scale_difference = _bbox_disagreement(local_box, branch_box)
    diagnostics = branch_candidate.get(
        "target_prompt_collaboration_diagnostics", {})
    collaboration = branch_candidate.get("out_dict", {}).get(
        "target_prompt_collaboration", {})
    residual_norm = _scalar(collaboration.get("residual_norm", 0.0))
    relative_norm = _scalar(
        collaboration.get("relative_residual_norm", 0.0))
    residual_scale = _scalar(collaboration.get("residual_scale", 0.0))
    remote_count = int(_scalar(
        collaboration.get("valid_remote_count", torch.zeros(1))))
    row = {
        "sequence_name": sequence_name,
        "target_id": target_id,
        "receiver_view": receiver_view,
        "frame_id": int(frame_id),
        "branch_name": branch_name,
        "selected_sender_views": "|".join(selected_senders),
        "local_bbox_x": local_box[0],
        "local_bbox_y": local_box[1],
        "local_bbox_w": local_box[2],
        "local_bbox_h": local_box[3],
        "branch_bbox_x": branch_box[0],
        "branch_bbox_y": branch_box[1],
        "branch_bbox_w": branch_box[2],
        "branch_bbox_h": branch_box[3],
        "local_score": _scalar(local_candidate["max_score"]),
        "local_apce": _scalar(local_candidate["apce"]),
        "branch_score": _scalar(branch_candidate["max_score"]),
        "branch_apce": _scalar(branch_candidate["apce"]),
        "local_branch_center_displacement": displacement,
        "local_branch_scale_difference": scale_difference,
        "residual_norm": residual_norm,
        "relative_residual_norm": relative_norm,
        "residual_scale": residual_scale,
        "remote_count": remote_count,
        "used_remote": bool(collaboration.get("used_remote", False)),
        "local_forward_delta": int(local_forward_delta),
        "state_digest_before": diagnostics.get(
            "persistent_state_digest_before", ""),
        "state_digest_after_collaboration": diagnostics.get(
            "persistent_state_digest_after_collaboration", ""),
        "state_identity": (
            diagnostics.get("persistent_state_digest_before", "")
            == diagnostics.get(
                "persistent_state_digest_after_collaboration", "")),
        "sender_prompt_source": diagnostics.get(
            "sender_prompt_source", "local"),
        "state_output_source": diagnostics.get(
            "state_output_source", "local"),
        "uses_gt": False,
    }
    return row


def _local_branch_candidate(local_candidate):
    candidate = dict(local_candidate)
    candidate["target_prompt_collaboration_diagnostics"] = {
        "sender_prompt_source": "local",
        "state_output_source": "local",
    }
    return candidate


def _initial_rows(sequence_name, target_id, receiver_view, bbox):
    base = {
        "sequence_name": sequence_name,
        "target_id": target_id,
        "receiver_view": receiver_view,
        "frame_id": 0,
        "selected_sender_views": "",
        "local_bbox_x": float(bbox[0]),
        "local_bbox_y": float(bbox[1]),
        "local_bbox_w": float(bbox[2]),
        "local_bbox_h": float(bbox[3]),
        "branch_bbox_x": float(bbox[0]),
        "branch_bbox_y": float(bbox[1]),
        "branch_bbox_w": float(bbox[2]),
        "branch_bbox_h": float(bbox[3]),
        "local_score": float("nan"),
        "local_apce": float("nan"),
        "branch_score": float("nan"),
        "branch_apce": float("nan"),
        "local_branch_center_displacement": 0.0,
        "local_branch_scale_difference": 0.0,
        "residual_norm": 0.0,
        "relative_residual_norm": 0.0,
        "residual_scale": 0.0,
        "remote_count": 0,
        "used_remote": False,
        "local_forward_delta": 0,
        "state_digest_before": "",
        "state_digest_after_collaboration": "",
        "state_identity": True,
        "sender_prompt_source": "local",
        "state_output_source": "local",
        "uses_gt": False,
    }
    return [dict(base, branch_name=name) for name in BRANCHES]


def _initial_prompt_rows(sequence_name, target_id, receiver_view, senders):
    fields = (
        "sender_score", "sender_apce", "receiver_score", "receiver_apce",
        "prompt_topk_score_mean", "prompt_topk_score_std",
        "prompt_topk_score_min", "prompt_topk_score_max",
        "prompt_top1_top8_gap", "prompt_norm_mean", "prompt_norm_std",
        "prompt_pairwise_cos_mean", "prompt_pairwise_cos_std",
        "prompt_pairwise_cos_min", "prompt_pairwise_cos_max",
        "set_cos_mean", "set_cos_max", "sender_to_receiver_best_mean",
        "receiver_to_sender_best_mean", "symmetric_best_match",
        "branch_score", "branch_apce", "local_branch_center_displacement",
        "local_branch_scale_difference", "residual_norm",
        "relative_residual_norm", "residual_scale")
    rows = []
    for slot, sender in enumerate(senders):
        row = {
            "sequence_name": sequence_name,
            "target_id": target_id,
            "receiver_view": receiver_view,
            "frame_id": 0,
            "sender_slot": slot,
            "sender_view": sender,
            "remote_count": 0,
            "uses_gt": False,
        }
        row.update({name: float("nan") for name in fields})
        rows.append(row)
    return rows


def _validate_no_forbidden_columns(columns):
    forbidden = []
    for name in columns:
        lowered = name.lower()
        if lowered == "uses_gt":
            continue
        if (lowered.startswith("gt_") or "ground_truth" in lowered
                or lowered in ("iou", "delta_iou", "label", "visibility",
                               "target_visible")):
            forbidden.append(name)
    if forbidden:
        raise RuntimeError("prediction schema contains GT columns: {}".format(
            forbidden))


def run(args):
    if args.dataset != "threemdot_val":
        raise RuntimeError("E3-U1 rollout is restricted to threemdot_val")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != args.expected_sha256:
        raise RuntimeError("checkpoint SHA256 mismatch")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("non-empty output directory: {}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix="e3-u1-prediction-", dir=str(output_dir.parent)))
    try:
        torch.cuda.set_device(args.gpu)
        triplets = list(three_view_triplets(get_dataset(args.dataset)))
        triplets.sort(key=lambda value: value[0].name)
        if args.max_targets > 0:
            triplets = triplets[:args.max_targets]
        branch_rows = []
        feature_rows = []
        target_inventory = []
        audit = {
            "active_receiver_frames": 0,
            "local_forward_mismatch": 0,
            "state_mutation": 0,
            "both_report_mismatch": 0,
            "local_state_mismatch": 0,
            "uses_gt_rows": 0,
        }
        wrapper = Tracker(
            "entertrack", "target_prompt_collaboration_e3", args.dataset,
            run_id=args.runid, checkpoint_override=str(checkpoint),
            no_gt_inference=True)
        for target_index, sequences in enumerate(triplets):
            parsed = tuple(sequence.name.rsplit("-", 1) for sequence in sequences)
            target_ids = {value[0] for value in parsed}
            if len(target_ids) != 1 or tuple(value[1] for value in parsed) != (
                    "1", "2", "3"):
                raise RuntimeError("invalid canonical triplet")
            target_id = parsed[0][0]
            frame_counts = tuple(len(sequence.frames) for sequence in sequences)
            if len(set(frame_counts)) != 1:
                raise RuntimeError("unequal synchronized frame counts")
            frame_limit = frame_counts[0]
            if args.max_frames > 0:
                frame_limit = min(frame_limit, args.max_frames)
            trackers = []
            for _ in range(3):
                params = wrapper.get_parameters()
                params.debug = 0
                trackers.append(wrapper.create_tracker(params))
            init_images = tuple(wrapper._read_image(sequence.frames[0])
                                for sequence in sequences)
            init_infos = tuple(sequence.init_info() for sequence in sequences)
            for view, sequence, tracker, image, info in zip(
                    VIEWS, sequences, trackers, init_images, init_infos):
                tracker.initialize(image, info)
                branch_rows.extend(_initial_rows(
                    sequence.name, target_id, view, tracker.state))
                senders = tuple(item for item in VIEWS if item != view)
                feature_rows.extend(_initial_prompt_rows(
                    sequence.name, target_id, view, senders))

            for frame_id in range(1, frame_limit):
                images = tuple(wrapper._read_image(sequence.frames[frame_id])
                               for sequence in sequences)
                counts_before = tuple(
                    tracker._target_prompt_local_forward_count
                    for tracker in trackers)
                local_candidates = tuple(
                    tracker.target_prompt_local_candidate(image)
                    for tracker, image in zip(trackers, images))
                counts_after = tuple(
                    tracker._target_prompt_local_forward_count
                    for tracker in trackers)
                deltas = tuple(end - start for start, end in zip(
                    counts_before, counts_after))
                if deltas != (1, 1, 1):
                    audit["local_forward_mismatch"] += 1

                metadata = tuple(_prompt_metadata(tracker, candidate)
                                 for tracker, candidate in zip(
                                     trackers, local_candidates))
                for receiver, (view, sequence, tracker, local) in enumerate(zip(
                        VIEWS, sequences, trackers, local_candidates)):
                    sender_indices = tuple(
                        index for index in range(3) if index != receiver)
                    sender_views = tuple(VIEWS[index] for index in sender_indices)
                    remotes = tuple(local_candidates[index]
                                    for index in sender_indices)
                    state_before = tracker.fcvc_persistent_state_digest()
                    sender0 = tracker.target_prompt_candidate(
                        local, _masked_remote_candidates(remotes, 0), view,
                        sender_views, frame_id, target_id=target_id)
                    sender1 = tracker.target_prompt_candidate(
                        local, _masked_remote_candidates(remotes, 1), view,
                        sender_views, frame_id, target_id=target_id)
                    both = tracker.target_prompt_candidate(
                        local, remotes, view, sender_views, frame_id,
                        target_id=target_id)
                    if tracker.fcvc_persistent_state_digest() != state_before:
                        audit["state_mutation"] += 1
                    candidates = {
                        "local": _local_branch_candidate(local),
                        "sender0_only": sender0,
                        "sender1_only": sender1,
                        "both": both,
                    }
                    selections = {
                        "local": (),
                        "sender0_only": (sender_views[0],),
                        "sender1_only": (sender_views[1],),
                        "both": sender_views,
                    }
                    for name in BRANCHES:
                        branch_rows.append(_branch_row(
                            sequence.name, target_id, view, frame_id, name,
                            local, candidates[name], selections[name],
                            deltas[receiver]))

                    receiver_prompt = metadata[receiver]["prompt"]
                    for slot, sender_index in enumerate(sender_indices):
                        single = sender0 if slot == 0 else sender1
                        diagnostics = single["out_dict"][
                            "target_prompt_collaboration"]
                        sender_meta = metadata[sender_index]
                        row = {
                            "sequence_name": sequence.name,
                            "target_id": target_id,
                            "receiver_view": view,
                            "frame_id": frame_id,
                            "sender_slot": slot,
                            "sender_view": VIEWS[sender_index],
                            "sender_score": _scalar(
                                local_candidates[sender_index]["max_score"]),
                            "sender_apce": _scalar(
                                local_candidates[sender_index]["apce"]),
                            "receiver_score": _scalar(local["max_score"]),
                            "receiver_apce": _scalar(local["apce"]),
                            **_prompt_stats(sender_meta),
                            **_set_compatibility(
                                receiver_prompt, sender_meta["prompt"]),
                            "branch_score": _scalar(single["max_score"]),
                            "branch_apce": _scalar(single["apce"]),
                            "local_branch_center_displacement": (
                                _bbox_disagreement(
                                    local["target_bbox"],
                                    single["target_bbox"])[0]),
                            "local_branch_scale_difference": (
                                _bbox_disagreement(
                                    local["target_bbox"],
                                    single["target_bbox"])[1]),
                            "residual_norm": _scalar(
                                diagnostics["residual_norm"]),
                            "relative_residual_norm": _scalar(
                                diagnostics["relative_residual_norm"]),
                            "residual_scale": _scalar(
                                diagnostics["residual_scale"]),
                            "remote_count": int(_scalar(
                                diagnostics["valid_remote_count"])),
                            "uses_gt": False,
                        }
                        feature_rows.append(row)

                    output, _, _ = tracker.target_prompt_finalize_frame(
                        local, both, info=None,
                        debug_name="e3-u1-{}".format(view.lower()))
                    if not _same_bbox(output["target_bbox"], both["target_bbox"]):
                        audit["both_report_mismatch"] += 1
                    if not _same_bbox(tracker.state, local["target_bbox"]):
                        audit["local_state_mismatch"] += 1
                    audit["active_receiver_frames"] += 1
                if frame_id % 250 == 0:
                    print("E3-U1 target={} frame={}/{}".format(
                        target_id, frame_id, frame_limit - 1), flush=True)
            target_inventory.append({
                "target_id": target_id,
                "sequences": [sequence.name for sequence in sequences],
                "source_frame_count": list(frame_counts),
                "processed_frame_count": frame_limit,
            })
            print("E3-U1 completed target {}/{}: {}".format(
                target_index + 1, len(triplets), target_id), flush=True)

        if any(audit[name] for name in (
                "local_forward_mismatch", "state_mutation",
                "both_report_mismatch", "local_state_mismatch",
                "uses_gt_rows")):
            raise RuntimeError("E3-U1 runtime audit failed: {}".format(audit))
        branch_columns = tuple(branch_rows[0])
        feature_columns = tuple(feature_rows[0])
        _validate_no_forbidden_columns(branch_columns)
        _validate_no_forbidden_columns(feature_columns)
        branch_path = staging / "prediction_only_e3_sender_counterfactual.csv"
        feature_path = staging / "prediction_only_e3_prompt_features.csv"
        _write_csv(branch_path, branch_rows, branch_columns)
        _write_csv(feature_path, feature_rows, feature_columns)
        manifest = {
            "schema_version": 1,
            "phase": "prediction_freeze_before_gt_join",
            "dataset": args.dataset,
            "runid": str(args.runid),
            "uses_gt": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "targets": target_inventory,
            "target_count": len(target_inventory),
            "sequence_count": 3 * len(target_inventory),
            "branch_file": branch_path.name,
            "branch_rows": len(branch_rows),
            "branch_columns": list(branch_columns),
            "branch_sha256": _sha256(branch_path),
            "prompt_feature_file": feature_path.name,
            "prompt_feature_rows": len(feature_rows),
            "prompt_feature_columns": list(feature_columns),
            "prompt_feature_sha256": _sha256(feature_path),
            "branches": list(BRANCHES),
            "runtime_audit": audit,
        }
        manifest_path = staging / "prediction_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=False)
        for path in staging.iterdir():
            shutil.move(str(path), str(output_dir / path.name))
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    finally:
        shutil.rmtree(str(staging), ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="threemdot_val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runid", default="e3_u1_val")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
