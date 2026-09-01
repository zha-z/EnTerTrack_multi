#!/usr/bin/env python3
"""Extract frozen B0 representations for preregistered D2-P2 sources."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
P1_DIR = ROOT / (
    "docs/results/target_prompt_d2_p1_"
    "synthetic_natural_representation_audit_20260830")
P1_INVENTORY_SHA256 = (
    "ca20bc2b0482dde2088bff5e8a2ff2e2c545cceb4ac6302de3e6ad68fe17b911")
P1_FEATURE_SHA256 = (
    "f8c42d9ae52fb03a260292cfddfee00018b7986b9b3ee42afeb3e6c08cf36863")
P1_PROMPT_SHA256 = (
    "ab667bf09b08208c33b41cd4d356e8dfc0c8ff0b09d55fcf4b4fd5b5a3d367ec")

from lib.train.admin import env_settings  # noqa: E402
from lib.train.target_prompt_asymmetric_degradation import (  # noqa: E402
    _normalized_box_to_pixels,
)
from tracking.run_target_prompt_d2_p1_representation import (  # noqa: E402
    load_frozen_model,
    prepare_sample,
    prompt_statistics,
    read_csv,
    score_entropy,
    sha256_file,
    write_csv,
)
from tracking.target_prompt_d2_p2_partial_degradation import (  # noqa: E402
    CANDIDATE_COVERAGE,
    FILL_VALUE_NORMALIZED,
    ORIENTATION_NAMESPACE,
    apply_partial_occlusion,
    orientation_for_sample,
)


PRIMARY_FEATURES = (
    "max_score", "apce", "score_entropy", "prompt_topk_score_mean",
    "prompt_top1_top8_gap", "prompt_norm_mean",
    "prompt_pairwise_cos_mean",
)
PROMPT_STAT_FEATURES = (
    "prompt_topk_score_mean", "prompt_topk_score_std",
    "prompt_topk_score_min", "prompt_topk_score_max",
    "prompt_top1_top8_gap", "prompt_norm_mean", "prompt_norm_std",
    "prompt_pairwise_cos_mean", "prompt_pairwise_cos_std",
    "prompt_pairwise_cos_min", "prompt_pairwise_cos_max",
)
REPRESENTATION_FEATURES = (
    "max_score", "apce", "score_map_max", "score_map_mean",
    "score_map_std", "score_top1_top2_gap", "score_entropy",
    "pred_bbox_x", "pred_bbox_y", "pred_bbox_w", "pred_bbox_h",
    "prompt_valid",
) + PROMPT_STAT_FEATURES
OUTPUT_COLUMNS = (
    "sample_id", "source_sample_id", "split", "target_id", "view",
    "template_frame_id", "search_frame_id", "group", "candidate",
    "coverage", "orientation", "pair_id", "balanced_primary",
) + REPRESENTATION_FEATURES


def _read_p1_reference(split):
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    inventory_path = P1_DIR / "sample_inventory.csv"
    feature_path = P1_DIR / "representation_features.csv"
    prompt_path = P1_DIR / "prompt_features.npz"
    actual = {
        "inventory": sha256_file(inventory_path),
        "feature": sha256_file(feature_path),
        "prompt": sha256_file(prompt_path),
    }
    expected = {
        "inventory": P1_INVENTORY_SHA256,
        "feature": P1_FEATURE_SHA256,
        "prompt": P1_PROMPT_SHA256,
    }
    if actual != expected:
        raise RuntimeError("D2-P1 frozen reference SHA256 mismatch")
    inventory = read_csv(inventory_path)
    features = read_csv(feature_path)
    if len(inventory) != len(features):
        raise RuntimeError("D2-P1 inventory/feature row mismatch")
    if [row["sample_id"] for row in inventory] != [
            row["sample_id"] for row in features]:
        raise RuntimeError("D2-P1 inventory/feature order mismatch")
    inventory_by_id = {row["sample_id"]: row for row in inventory}
    selected = [row for row in features if row["split"] == split]
    clean = [row for row in selected if row["group"] == "clean"]
    natural = [row for row in selected if row["group"] == "natural"
               and str(row["balanced_primary"]).lower() == "true"]
    p100 = [row for row in selected if row["group"] == "synthetic"]
    if not (len(clean) == len(natural) == len(p100) and clean):
        raise RuntimeError("D2-P1 balanced reference counts differ")
    clean_inventory = [inventory_by_id[row["sample_id"]] for row in clean]
    return {
        "clean_features": clean,
        "natural_features": natural,
        "p100_features": p100,
        "clean_inventory": clean_inventory,
        "hashes": actual,
    }


def _reference_record(row, candidate, source_sample_id=""):
    orientation = orientation_for_sample(row["sample_id"]) \
        if candidate == "Clean" else ""
    output = {
        "sample_id": row["sample_id"],
        "source_sample_id": source_sample_id,
        "split": row["split"],
        "target_id": row["target_id"],
        "view": row["view"],
        "template_frame_id": int(row["template_frame_id"]),
        "search_frame_id": int(row["search_frame_id"]),
        "group": candidate.lower(),
        "candidate": candidate,
        "coverage": 0.0 if candidate == "Clean" else "",
        "orientation": orientation,
        "pair_id": row["pair_id"],
        "balanced_primary": True,
    }
    for name in REPRESENTATION_FEATURES:
        output[name] = row[name]
    return output


def _candidate_record(row, source_sample_id, candidate, audit, values,
                      prompt_stats, index):
    output = {
        "sample_id": "d2p2-{}-{}-{:06d}-{}".format(
            row["split"], row["sequence_name"],
            int(row["search_frame_id"]), candidate.lower()),
        "source_sample_id": source_sample_id,
        "split": row["split"],
        "target_id": row["target_id"],
        "view": row["view"],
        "template_frame_id": int(row["template_frame_id"]),
        "search_frame_id": int(row["search_frame_id"]),
        "group": candidate.lower(),
        "candidate": candidate,
        "coverage": CANDIDATE_COVERAGE[candidate],
        "orientation": audit["orientation"],
        "pair_id": row["pair_id"],
        "balanced_primary": True,
        "max_score": float(values["maximum"][index].item()),
        "apce": float(values["apce"][index].item()),
        "score_map_max": float(values["maximum"][index].item()),
        "score_map_mean": float(values["flat"][index].mean().item()),
        "score_map_std": float(values["flat"][index].std(
            unbiased=False).item()),
        "score_top1_top2_gap": float(
            values["sorted_score"][index, 0].item()
            - values["sorted_score"][index, 1].item()),
        "score_entropy": float(values["entropy"][index].item()),
        "pred_bbox_x": float(values["boxes"][index, 0].item()),
        "pred_bbox_y": float(values["boxes"][index, 1].item()),
        "pred_bbox_w": float(values["boxes"][index, 2].item()),
        "pred_bbox_h": float(values["boxes"][index, 3].item()),
        "prompt_valid": bool(values["valid"][index].item()),
    }
    for name, vector in prompt_stats.items():
        output[name] = float(vector[index].item())
    return output


def _extract_candidates(model, clean_inventory, candidates, dataset_root,
                        device, batch_size, adapter_calls):
    prepared = []
    box_cache = {}
    input_identity_mismatch = 0
    realized = {candidate: [] for candidate in candidates}
    for row in clean_inventory:
        base_row = dict(row)
        base_row["group"] = "clean"
        template, search, normalized_box = prepare_sample(
            base_row, dataset_root, box_cache)
        source_sample_id = row["sample_id"]
        full_box, _ = _normalized_box_to_pixels(normalized_box, 256, 256)
        expected_p100 = search.clone()
        expected_p100[:, full_box[1]:full_box[3],
                      full_box[0]:full_box[2]] = FILL_VALUE_NORMALIZED
        for candidate in candidates:
            degraded, audit = apply_partial_occlusion(
                search, normalized_box, candidate, source_sample_id)
            if candidate == "P100" and not torch.equal(
                    degraded, expected_p100):
                input_identity_mismatch += 1
            realized[candidate].append(audit["realized_coverage"])
            prepared.append({
                "row": row,
                "source_sample_id": source_sample_id,
                "candidate": candidate,
                "audit": audit,
                "template": template,
                "search": degraded,
            })
    if input_identity_mismatch:
        raise RuntimeError("P100 input transform differs from D2-P1/D1")

    records = []
    prompts = []
    prompt_ids = []
    for start in range(0, len(prepared), batch_size):
        current = prepared[start:start + batch_size]
        # D2-P1 was extracted with full batches.  Duplicate-only padding keeps
        # the final CUDA kernel batch shape identical; padding outputs are
        # discarded and never enter any artifact or statistic.
        forward_items = list(current)
        if len(forward_items) < batch_size:
            forward_items.extend(
                [forward_items[-1]] * (batch_size - len(forward_items)))
        template = torch.stack(
            [item["template"] for item in forward_items]).to(device)
        search = torch.stack(
            [item["search"] for item in forward_items]).to(device)
        with torch.inference_mode():
            output = model(
                template=template, search=search, ce_template_mask=None,
                ce_keep_rate=None, temperature=100.0,
                return_last_attn=False, return_atp=True, training=False)
            score = output["score_map"].float()
            flat = score.flatten(1)
            sorted_score = torch.topk(flat, k=2, dim=1).values
            maximum = flat.max(dim=1).values
            minimum = flat.min(dim=1).values
            apce = ((maximum - minimum) ** 2) / (
                ((flat - minimum[:, None]) ** 2).mean(dim=1) + 1e-8)
            tokens = output["backbone_feat"][:, -model.feat_len_s:]
            extraction = model.target_prompt_extractor.extract_with_metadata(
                tokens, score)
            prompt = extraction["prompt"].float()
            prompt_stats = prompt_statistics(
                prompt, extraction["topk_scores"].float())
            values = {
                "maximum": maximum,
                "apce": apce,
                "flat": flat,
                "sorted_score": sorted_score,
                "entropy": score_entropy(score),
                "boxes": output["pred_boxes"].view(
                    len(forward_items), -1, 4).mean(1),
                "valid": extraction["valid"],
            }
        prompt_cpu = prompt[:len(current)].detach().cpu().numpy().astype(
            np.float16)
        prompts.append(prompt_cpu)
        for index, item in enumerate(current):
            record = _candidate_record(
                item["row"], item["source_sample_id"], item["candidate"],
                item["audit"], values, prompt_stats, index)
            records.append(record)
            prompt_ids.append(record["sample_id"])
        print("D2-P2 representation {}/{}".format(
            min(start + batch_size, len(prepared)), len(prepared)), flush=True)
    if adapter_calls["count"] != 0:
        raise RuntimeError("collaboration adapter was invoked")
    prompt_array = np.concatenate(prompts, axis=0)
    if prompt_array.shape != (len(records), 8, 192):
        raise RuntimeError("unexpected D2-P2 prompt shape")
    coverage_summary = {
        candidate: {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }
        for candidate, values in realized.items()
    }
    return records, prompt_ids, prompt_array, coverage_summary


def _verify_p100(records, prompts, reference):
    reference_by_pair = {row["pair_id"]: row
                         for row in reference["p100_features"]}
    p1_prompt = np.load(str(P1_DIR / "prompt_features.npz"), allow_pickle=False)
    prompt_by_id = {sample_id: prompt for sample_id, prompt in zip(
        p1_prompt["sample_id"].astype(str).tolist(), p1_prompt["prompt"])}
    maximum_metric_difference = 0.0
    maximum_prompt_difference = 0.0
    mismatches = 0
    for index, row in enumerate(records):
        if row["candidate"] != "P100":
            continue
        old = reference_by_pair.get(row["pair_id"])
        if old is None:
            raise RuntimeError("P100 pair missing from D2-P1")
        for name in PRIMARY_FEATURES + PROMPT_STAT_FEATURES:
            difference = abs(float(row[name]) - float(old[name]))
            maximum_metric_difference = max(maximum_metric_difference, difference)
            if difference > 1e-6:
                mismatches += 1
        old_prompt = prompt_by_id[old["sample_id"]].astype(np.float32)
        new_prompt = prompts[index].astype(np.float32)
        maximum_prompt_difference = max(
            maximum_prompt_difference,
            float(np.max(np.abs(new_prompt - old_prompt))))
        if not np.array_equal(prompts[index], prompt_by_id[old["sample_id"]]):
            mismatches += 1
    if mismatches:
        raise RuntimeError(
            "P100 representation differs from D2-P1: count={}, metric={}, "
            "prompt={}".format(mismatches, maximum_metric_difference,
                               maximum_prompt_difference))
    return {
        "rows_compared": len(reference_by_pair),
        "metric_tolerance": 1e-6,
        "maximum_metric_abs_difference": maximum_metric_difference,
        "prompt_fp16_exact": maximum_prompt_difference == 0.0,
        "maximum_prompt_abs_difference": maximum_prompt_difference,
        "mismatch_count": mismatches,
    }


def _selected_source(output_dir):
    selected_path = output_dir / "selected_source.json"
    manifest_path = output_dir / "selected_source_manifest.json"
    if not selected_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("VAL requires frozen selected_source artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(selected_path) != manifest["selected_source_sha256"]:
        raise RuntimeError("selected_source SHA256 mismatch")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not selected["train_acceptance"]["pass"]:
        raise RuntimeError("Train gate failed; VAL is forbidden")
    candidate = selected["candidate"]
    if candidate not in ("P25", "P50", "P75"):
        raise RuntimeError("invalid frozen selected candidate")
    return candidate, manifest["selected_source_sha256"]


def freeze_representation(output_dir, split, dataset_root, device, batch_size):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if "test" in str(output_dir).lower():
        raise RuntimeError("test-named output path is forbidden")
    reference = _read_p1_reference(split)
    if split == "train":
        candidates = ("P25", "P50", "P75", "P100")
        selection_sha = None
    else:
        selected, selection_sha = _selected_source(output_dir)
        candidates = (selected, "P100")
    prefix = split
    destinations = [
        output_dir / (prefix + "_representation_features.csv"),
        output_dir / (prefix + "_candidate_prompts.npz"),
        output_dir / (prefix + "_representation_manifest.json"),
    ]
    if any(path.exists() for path in destinations):
        raise RuntimeError("{} representation destination exists".format(split))
    model, model_audit = load_frozen_model(device)
    adapter_calls = {"count": 0}

    def adapter_hook(module, inputs):
        adapter_calls["count"] += 1

    hook = model.target_prompt_collaboration.register_forward_pre_hook(
        adapter_hook)
    try:
        candidate_rows, prompt_ids, prompt_array, coverage_summary = (
            _extract_candidates(
                model, reference["clean_inventory"], candidates,
                dataset_root, device, batch_size, adapter_calls))
    finally:
        hook.remove()
    p100_identity = _verify_p100(candidate_rows, prompt_array, reference)
    reference_rows = [
        _reference_record(row, "Clean")
        for row in reference["clean_features"]
    ] + [
        _reference_record(row, "Natural")
        for row in reference["natural_features"]
    ]
    rows = reference_rows + candidate_rows
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate D2-P2 sample id")
    staging = Path(tempfile.mkdtemp(
        prefix="d2-p2-{}-representation-".format(split),
        dir=str(output_dir)))
    try:
        feature_path = staging / destinations[0].name
        prompt_path = staging / destinations[1].name
        manifest_path = staging / destinations[2].name
        write_csv(feature_path, [
            {name: row[name] for name in OUTPUT_COLUMNS} for row in rows])
        np.savez_compressed(
            str(prompt_path), sample_id=np.asarray(prompt_ids, dtype="U80"),
            prompt=prompt_array)
        manifest = {
            "phase": "{}_representation_freeze".format(split),
            "split": split,
            "candidates_run": list(candidates),
            "selected_source_sha256_verified": selection_sha,
            "d2_p1_reference_hashes_verified": reference["hashes"],
            "reference_clean_rows": len(reference["clean_features"]),
            "reference_natural_rows": len(reference["natural_features"]),
            "candidate_rows": len(candidate_rows),
            "feature_file": destinations[0].name,
            "feature_rows": len(rows),
            "feature_sha256": sha256_file(feature_path),
            "prompt_file": destinations[1].name,
            "prompt_shape": list(prompt_array.shape),
            "prompt_dtype": str(prompt_array.dtype),
            "prompt_sha256": sha256_file(prompt_path),
            "orientation_namespace": ORIENTATION_NAMESPACE,
            "fill_value_normalized": FILL_VALUE_NORMALIZED,
            "realized_coverage": coverage_summary,
            "p100_identity": p100_identity,
            "model": model_audit,
            "runtime_audit": {
                "model_eval": not model.training,
                "inference_mode": True,
                "adapter_forward_calls": adapter_calls["count"],
                "collaboration_used": False,
                "gt_passed_to_model": False,
                "gt_used_for_deterministic_crop": True,
                "training_run": False,
                "backward_run": False,
                "official_test_accessed": False,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        for source, destination in zip(
                (feature_path, prompt_path, manifest_path), destinations):
            os.replace(str(source), str(destination))
    finally:
        try:
            staging.rmdir()
        except OSError:
            pass
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("train", "val"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    dataset_root = Path(
        args.dataset_root or env_settings().threemdot_dir).resolve()
    torch.cuda.set_device(args.gpu)
    freeze_representation(
        args.output_dir, args.phase, dataset_root,
        torch.device("cuda", args.gpu), args.batch_size)


if __name__ == "__main__":
    main()
