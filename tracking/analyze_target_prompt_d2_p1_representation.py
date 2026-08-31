#!/usr/bin/env python3
"""Descriptive-only analysis of frozen D2-P1 representations."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


PRIMARY_FEATURES = (
    "max_score",
    "apce",
    "score_entropy",
    "prompt_topk_score_mean",
    "prompt_top1_top8_gap",
    "prompt_norm_mean",
    "prompt_pairwise_cos_mean",
)
PROMPT_FEATURES = (
    "prompt_topk_score_mean",
    "prompt_top1_top8_gap",
    "prompt_norm_mean",
    "prompt_pairwise_cos_mean",
)
SUMMARY_FEATURES = (
    "max_score", "apce", "score_map_mean", "score_map_std",
    "score_top1_top2_gap", "score_entropy",
    "prompt_topk_score_mean", "prompt_topk_score_std",
    "prompt_topk_score_min", "prompt_topk_score_max",
    "prompt_top1_top8_gap", "prompt_norm_mean", "prompt_norm_std",
    "prompt_pairwise_cos_mean", "prompt_pairwise_cos_std",
    "prompt_pairwise_cos_min", "prompt_pairwise_cos_max",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("refusing to write empty CSV: {}".format(path))
    columns = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def finite_values(rows, feature):
    values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {name: float("nan") for name in (
            "mean", "std", "median", "p10", "p25", "p75", "p90")}
    quantiles = np.quantile(
        values, (0.10, 0.25, 0.50, 0.75, 0.90), method="linear")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(quantiles[2]),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }


def wasserstein_equal(values_a, values_b):
    a = np.sort(np.asarray(values_a, dtype=np.float64))
    b = np.sort(np.asarray(values_b, dtype=np.float64))
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("balanced W1 requires equal nonzero sample counts")
    return float(np.mean(np.abs(a - b)))


def ks_statistic(values_a, values_b):
    a = np.sort(np.asarray(values_a, dtype=np.float64))
    b = np.sort(np.asarray(values_b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    support = np.sort(np.unique(np.concatenate((a, b))))
    cdf_a = np.searchsorted(a, support, side="right") / float(len(a))
    cdf_b = np.searchsorted(b, support, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def standardized_mean_difference(values_a, values_b):
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_variance = (((len(a) - 1) * np.var(a, ddof=1)
                        + (len(b) - 1) * np.var(b, ddof=1))
                       / (len(a) + len(b) - 2))
    if not math.isfinite(pooled_variance) or pooled_variance < 1e-24:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled_variance))


def normalize_rows(tokens):
    norms = np.linalg.norm(tokens, axis=-1, keepdims=True)
    return tokens / np.maximum(norms, 1e-12)


def paired_prompt_metrics(clean, synthetic):
    c = normalize_rows(np.asarray(clean, dtype=np.float32))
    s = normalize_rows(np.asarray(synthetic, dtype=np.float32))
    similarity = np.clip(np.matmul(c, s.T), -1.0, 1.0)
    c_to_s = float(np.mean(np.max(similarity, axis=1)))
    s_to_c = float(np.mean(np.max(similarity, axis=0)))
    centroid_c = np.mean(np.asarray(clean, dtype=np.float32), axis=0)
    centroid_s = np.mean(np.asarray(synthetic, dtype=np.float32), axis=0)
    centroid_cos = float(np.clip(
        np.dot(centroid_c, centroid_s) / max(
            np.linalg.norm(centroid_c) * np.linalg.norm(centroid_s), 1e-12),
        -1.0, 1.0))
    return {
        "c_to_s_nearest_cos_mean": c_to_s,
        "s_to_c_nearest_cos_mean": s_to_c,
        "symmetric_best_match_cos": 0.5 * (c_to_s + s_to_c),
        "cross_token_cos_mean_control": float(np.mean(similarity)),
        "centroid_cos_control": centroid_cos,
    }


def verify_freeze(output_dir):
    inventory_manifest = json.loads(
        (output_dir / "sample_inventory_manifest.json").read_text(
            encoding="utf-8"))
    representation_manifest = json.loads(
        (output_dir / "representation_manifest.json").read_text(
            encoding="utf-8"))
    inventory_path = output_dir / inventory_manifest["inventory_file"]
    feature_path = output_dir / representation_manifest["feature_file"]
    prompt_path = output_dir / representation_manifest["prompt_file"]
    if sha256_file(inventory_path) != inventory_manifest["inventory_sha256"]:
        raise RuntimeError("inventory freeze mismatch")
    if sha256_file(feature_path) != representation_manifest["feature_sha256"]:
        raise RuntimeError("feature freeze mismatch")
    if sha256_file(prompt_path) != representation_manifest["prompt_sha256"]:
        raise RuntimeError("prompt freeze mismatch")
    if representation_manifest["runtime_audit"]["adapter_forward_calls"] != 0:
        raise RuntimeError("collaboration entered primary representation")
    return (inventory_manifest, representation_manifest,
            inventory_path, feature_path, prompt_path)


def _balanced(rows, split, group):
    return [row for row in rows
            if row["split"] == split and row["group"] == group
            and as_bool(row["balanced_primary"])]


def _prompt_centroid_nearest(features, prompt_by_id):
    output = {}
    for split in ("train", "val"):
        for view in ("A", "B", "C"):
            natural = [row for row in features
                       if row["split"] == split and row["view"] == view
                       and row["group"] == "natural"
                       and as_bool(row["balanced_primary"])]
            synthetic = [row for row in features
                         if row["split"] == split and row["view"] == view
                         and row["group"] == "synthetic"]
            if not natural:
                continue
            natural_centroids = np.stack([
                prompt_by_id[row["sample_id"]].astype(np.float32).mean(axis=0)
                for row in natural])
            natural_centroids = normalize_rows(natural_centroids)
            for start in range(0, len(synthetic), 512):
                current = synthetic[start:start + 512]
                centroids = np.stack([
                    prompt_by_id[row["sample_id"]].astype(np.float32).mean(axis=0)
                    for row in current])
                centroids = normalize_rows(centroids)
                nearest = np.max(np.matmul(centroids, natural_centroids.T), axis=1)
                for row, value in zip(current, nearest):
                    output[row["sample_id"]] = float(value)
    return output


def analyze(output_dir):
    output_dir = Path(output_dir).resolve()
    (inventory_manifest, representation_manifest, inventory_path,
     feature_path, prompt_path) = verify_freeze(output_dir)
    inventory = read_csv(inventory_path)
    features = read_csv(feature_path)
    if len(inventory) != len(features):
        raise RuntimeError("inventory/feature row mismatch")
    if [row["sample_id"] for row in inventory] != [
            row["sample_id"] for row in features]:
        raise RuntimeError("inventory/feature sample order mismatch")
    prompt_artifact = np.load(str(prompt_path), allow_pickle=False)
    prompt_ids = prompt_artifact["sample_id"].astype(str).tolist()
    prompts = prompt_artifact["prompt"]
    if prompt_ids != [row["sample_id"] for row in features]:
        raise RuntimeError("prompt sample order mismatch")
    prompt_by_id = {sample_id: prompt
                    for sample_id, prompt in zip(prompt_ids, prompts)}

    # Strong paired causal analysis.
    pair_groups = defaultdict(dict)
    for row in features:
        if row["pair_id"]:
            pair_groups[row["pair_id"]][row["group"]] = row
    natural_nearest = _prompt_centroid_nearest(features, prompt_by_id)
    paired_rows = []
    for pair_id in sorted(pair_groups):
        pair = pair_groups[pair_id]
        if set(pair) != {"clean", "synthetic"}:
            raise RuntimeError("incomplete C/S pair: {}".format(pair_id))
        clean = pair["clean"]
        synthetic = pair["synthetic"]
        if any(clean[name] != synthetic[name] for name in (
                "split", "target_id", "view", "template_frame_id",
                "search_frame_id")):
            raise RuntimeError("C/S identity mismatch: {}".format(pair_id))
        c_box = np.asarray([float(clean[name]) for name in (
            "pred_bbox_x", "pred_bbox_y", "pred_bbox_w", "pred_bbox_h")])
        s_box = np.asarray([float(synthetic[name]) for name in (
            "pred_bbox_x", "pred_bbox_y", "pred_bbox_w", "pred_bbox_h")])
        c_center = c_box[:2] + 0.5 * c_box[2:]
        s_center = s_box[:2] + 0.5 * s_box[2:]
        record = {
            "pair_id": pair_id,
            "split": clean["split"],
            "target_id": clean["target_id"],
            "view": clean["view"],
            "template_frame_id": int(clean["template_frame_id"]),
            "search_frame_id": int(clean["search_frame_id"]),
            "delta_max_score_s_minus_c": (
                float(synthetic["max_score"]) - float(clean["max_score"])),
            "delta_apce_s_minus_c": (
                float(synthetic["apce"]) - float(clean["apce"])),
            "delta_score_entropy_s_minus_c": (
                float(synthetic["score_entropy"])
                - float(clean["score_entropy"])),
            "bbox_center_displacement_normalized": float(
                np.linalg.norm(s_center - c_center)),
            "bbox_center_displacement_pixels_256": float(
                256.0 * np.linalg.norm(s_center - c_center)),
            "bbox_log_scale_l1": float(np.abs(np.log(
                np.maximum(s_box[2:], 1e-12)
                / np.maximum(c_box[2:], 1e-12))).sum()),
            "synthetic_same_view_natural_centroid_nearest_cos": (
                natural_nearest.get(synthetic["sample_id"], float("nan"))),
        }
        record.update(paired_prompt_metrics(
            prompt_by_id[clean["sample_id"]],
            prompt_by_id[synthetic["sample_id"]]))
        paired_rows.append(record)

    # Scalar distributions: balanced primary plus raw Natural.
    distribution_rows = []
    distance_rows = []
    comparison_pairs = (
        ("synthetic_vs_natural", "synthetic", "natural"),
        ("clean_vs_natural", "clean", "natural"),
        ("synthetic_vs_clean", "synthetic", "clean"),
    )
    for split in ("train", "val"):
        grouped = {group: _balanced(features, split, group)
                   for group in ("clean", "synthetic", "natural")}
        counts = {len(value) for value in grouped.values()}
        if len(counts) != 1 or next(iter(counts)) == 0:
            raise RuntimeError("balanced group counts differ for {}".format(split))
        for group, selected in grouped.items():
            for feature in SUMMARY_FEATURES:
                values = finite_values(selected, feature)
                distribution_rows.append({
                    "split": split,
                    "analysis_scope": "balanced_primary",
                    "group": group,
                    "feature": feature,
                    "sample_count": len(selected),
                    "finite_count": len(values),
                    **distribution_summary(values),
                })
        raw_natural = [row for row in features
                       if row["split"] == split and row["group"] == "natural"]
        for feature in SUMMARY_FEATURES:
            values = finite_values(raw_natural, feature)
            distribution_rows.append({
                "split": split,
                "analysis_scope": "natural_raw",
                "group": "natural",
                "feature": feature,
                "sample_count": len(raw_natural),
                "finite_count": len(values),
                **distribution_summary(values),
            })
        for comparison, first, second in comparison_pairs:
            for feature in PRIMARY_FEATURES:
                a = finite_values(grouped[first], feature)
                b = finite_values(grouped[second], feature)
                if len(a) != len(grouped[first]) or len(b) != len(grouped[second]):
                    raise RuntimeError("nonfinite primary feature: {}".format(feature))
                distance_rows.append({
                    "split": split,
                    "analysis_scope": "balanced_primary",
                    "comparison": comparison,
                    "first_group": first,
                    "second_group": second,
                    "feature": feature,
                    "sample_count_each": len(a),
                    "wasserstein_1": wasserstein_equal(a, b),
                    "ks_statistic": ks_statistic(a, b),
                    "standardized_mean_difference_first_minus_second": (
                        standardized_mean_difference(a, b)),
                })

    # Fixed Natural P10/P90 prompt anomaly description.
    anomaly_rows = []
    for split in ("train", "val"):
        for view in ("A", "B", "C"):
            natural = [row for row in features
                       if row["split"] == split and row["view"] == view
                       and row["group"] == "natural"
                       and as_bool(row["balanced_primary"])]
            for feature in PROMPT_FEATURES:
                natural_values = finite_values(natural, feature)
                if len(natural_values) == 0:
                    lower = upper = float("nan")
                else:
                    lower, upper = np.quantile(
                        natural_values, (0.10, 0.90), method="linear")
                for group in ("clean", "synthetic"):
                    selected = [row for row in features
                                if row["split"] == split
                                and row["view"] == view
                                and row["group"] == group]
                    values = finite_values(selected, feature)
                    outside = ((values < lower) | (values > upper)) \
                        if len(values) else np.asarray([], dtype=bool)
                    anomaly_rows.append({
                        "split": split,
                        "view": view,
                        "feature": feature,
                        "group": group,
                        "natural_p10": float(lower),
                        "natural_p90": float(upper),
                        "sample_count": len(values),
                        "outside_natural_p10_p90_count": int(outside.sum()),
                        "outside_natural_p10_p90_ratio": (
                            float(outside.mean()) if len(outside) else float("nan")),
                    })

    # Same-view prompt nearest similarity summary.
    prompt_similarity_rows = []
    for split in ("train", "val"):
        for view in ("A", "B", "C"):
            ids = [row["sample_id"] for row in features
                   if row["split"] == split and row["view"] == view
                   and row["group"] == "synthetic"]
            values = np.asarray([natural_nearest[item] for item in ids],
                                dtype=np.float64)
            prompt_similarity_rows.append({
                "split": split,
                "view": view,
                "comparison": "synthetic_to_same_view_balanced_natural",
                "sample_count": len(values),
                **distribution_summary(values),
            })

    # Per-target and per-view counts and primary means.
    per_target_rows = []
    per_view_rows = []
    for split in ("train", "val"):
        targets = sorted({row["target_id"] for row in features
                          if row["split"] == split})
        for target in targets:
            for group in ("clean", "synthetic", "natural"):
                selected = [row for row in features
                            if row["split"] == split
                            and row["target_id"] == target
                            and row["group"] == group]
                balanced = [row for row in selected
                            if as_bool(row["balanced_primary"])]
                per_target_rows.append({
                    "split": split,
                    "target_id": target,
                    "group": group,
                    "raw_sample_count": len(selected),
                    "balanced_sample_count": len(balanced),
                    **{"balanced_{}_mean".format(feature): (
                        float(np.mean(finite_values(balanced, feature)))
                        if balanced else float("nan"))
                       for feature in PRIMARY_FEATURES},
                })
        for view in ("A", "B", "C"):
            for group in ("clean", "synthetic", "natural"):
                selected = [row for row in features
                            if row["split"] == split and row["view"] == view
                            and row["group"] == group]
                balanced = [row for row in selected
                            if as_bool(row["balanced_primary"])]
                per_view_rows.append({
                    "split": split,
                    "view": view,
                    "group": group,
                    "raw_sample_count": len(selected),
                    "balanced_sample_count": len(balanced),
                    **{"balanced_{}_mean".format(feature): (
                        float(np.mean(finite_values(balanced, feature)))
                        if balanced else float("nan"))
                       for feature in PRIMARY_FEATURES},
                })

    # Preregistered case selection.
    split_decisions = {}
    for split in ("train", "val"):
        natural = _balanced(features, split, "natural")
        target_count = len({row["target_id"] for row in natural})
        view_counts = {view: sum(row["view"] == view for row in natural)
                       for view in ("A", "B", "C")}
        adequate = target_count >= 3 and min(view_counts.values()) >= 30
        rows_by_key = {(row["comparison"], row["feature"]): row
                       for row in distance_rows if row["split"] == split}
        large_shift = sum(abs(float(rows_by_key[
            ("synthetic_vs_clean", feature)][
                "standardized_mean_difference_first_minus_second"])) >= 0.8
                          for feature in PRIMARY_FEATURES)
        closer = sum(float(rows_by_key[("synthetic_vs_natural", feature)][
            "wasserstein_1"]) < float(rows_by_key[
                ("clean_vs_natural", feature)]["wasserstein_1"])
                     for feature in PRIMARY_FEATURES)
        split_decisions[split] = {
            "adequate": adequate,
            "balanced_natural_target_count": target_count,
            "balanced_natural_view_counts": view_counts,
            "large_shift_count_of_7": large_shift,
            "synthetic_closer_count_of_7": closer,
        }
    if not all(value["adequate"] for value in split_decisions.values()):
        frozen_case = "D"
    elif all(value["large_shift_count_of_7"] >= 4
             and value["synthetic_closer_count_of_7"] <= 3
             for value in split_decisions.values()):
        frozen_case = "A"
    elif all(value["synthetic_closer_count_of_7"] >= 4
             for value in split_decisions.values()):
        frozen_case = "B"
    else:
        frozen_case = "C"

    artifacts = {
        "clean_synthetic_paired_analysis.csv": paired_rows,
        "group_distribution_summary.csv": distribution_rows,
        "distribution_distance.csv": distance_rows,
        "per_target_summary.csv": per_target_rows,
        "per_view_summary.csv": per_view_rows,
        "prompt_anomaly_summary.csv": anomaly_rows,
        "synthetic_natural_prompt_similarity.csv": prompt_similarity_rows,
    }
    destinations = [output_dir / name for name in artifacts]
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    if any(path.exists() for path in destinations + [analysis_manifest_path]):
        raise RuntimeError("analysis destination already exists")
    staging = Path(tempfile.mkdtemp(prefix="d2-p1-analysis-",
                                    dir=str(output_dir)))
    try:
        artifact_manifest = {}
        for name, rows in artifacts.items():
            path = staging / name
            write_csv(path, rows)
            artifact_manifest[name] = {
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        manifest = {
            "phase": "post_freeze_descriptive_analysis",
            "inventory_sha256_verified": inventory_manifest["inventory_sha256"],
            "representation_feature_sha256_verified": (
                representation_manifest["feature_sha256"]),
            "prompt_sha256_verified": representation_manifest["prompt_sha256"],
            "primary_features": list(PRIMARY_FEATURES),
            "split_decisions": split_decisions,
            "frozen_case": frozen_case,
            "classifier_run": False,
            "selector_run": False,
            "learned_metric_run": False,
            "official_test_accessed": False,
            "artifacts": artifact_manifest,
        }
        staged_manifest = staging / analysis_manifest_path.name
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        for name in artifacts:
            os.replace(str(staging / name), str(output_dir / name))
        os.replace(str(staged_manifest), str(analysis_manifest_path))
    finally:
        try:
            staging.rmdir()
        except OSError:
            pass
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if "test" in str(args.output_dir).lower():
        raise RuntimeError("test-named output path is forbidden")
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
