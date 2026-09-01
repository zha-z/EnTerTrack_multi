#!/usr/bin/env python3
"""Train-only selection and frozen VAL holdout analysis for D2-P2."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P1_DIR = ROOT / (
    "docs/results/target_prompt_d2_p1_"
    "synthetic_natural_representation_audit_20260830")

from tracking.analyze_target_prompt_d2_p1_representation import (  # noqa: E402
    paired_prompt_metrics,
    standardized_mean_difference,
    wasserstein_equal,
)
from tracking.run_target_prompt_d2_p1_representation import (  # noqa: E402
    read_csv,
    sha256_file,
    write_csv,
)
from tracking.run_target_prompt_d2_p2_representation import (  # noqa: E402
    P1_FEATURE_SHA256,
    P1_INVENTORY_SHA256,
    P1_PROMPT_SHA256,
    PRIMARY_FEATURES,
)
from tracking.target_prompt_d2_p2_partial_degradation import (  # noqa: E402
    CANDIDATE_COVERAGE,
    FILL_VALUE_NORMALIZED,
    ORIENTATION_NAMESPACE,
)


TRAIN_CANDIDATES = ("P25", "P50", "P75", "P100")
SELECTABLE = ("P25", "P50", "P75")
EPSILON = 1e-12
LARGE_SMD = 0.8


def _finite(rows, feature):
    values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise RuntimeError("non-finite primary feature: {}".format(feature))
    return values


def _rows(rows, candidate, **filters):
    return [row for row in rows if row["candidate"] == candidate
            and all(row[name] == value for name, value in filters.items())]


def _verify_representation(output_dir, split):
    manifest_path = output_dir / (split + "_representation_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_path = output_dir / manifest["feature_file"]
    prompt_path = output_dir / manifest["prompt_file"]
    if sha256_file(feature_path) != manifest["feature_sha256"]:
        raise RuntimeError("{} feature freeze mismatch".format(split))
    if sha256_file(prompt_path) != manifest["prompt_sha256"]:
        raise RuntimeError("{} prompt freeze mismatch".format(split))
    if manifest["d2_p1_reference_hashes_verified"] != {
            "inventory": P1_INVENTORY_SHA256,
            "feature": P1_FEATURE_SHA256,
            "prompt": P1_PROMPT_SHA256}:
        raise RuntimeError("D2-P1 reference provenance mismatch")
    if manifest["p100_identity"]["mismatch_count"] != 0:
        raise RuntimeError("P100 identity failed")
    if manifest["runtime_audit"]["adapter_forward_calls"] != 0:
        raise RuntimeError("adapter entered representation path")
    rows = read_csv(feature_path)
    if len(rows) != manifest["feature_rows"]:
        raise RuntimeError("{} feature row mismatch".format(split))
    return manifest, rows, feature_path, prompt_path


def candidate_distances(rows, candidates):
    natural = _rows(rows, "Natural")
    clean = _rows(rows, "Clean")
    if len(natural) != len(clean) or not clean:
        raise RuntimeError("balanced Clean/Natural counts differ")
    output = []
    clean_w1 = {}
    for feature in PRIMARY_FEATURES:
        clean_w1[feature] = wasserstein_equal(
            _finite(clean, feature), _finite(natural, feature))
    for candidate in ("Clean",) + tuple(candidates):
        selected = _rows(rows, candidate)
        if len(selected) != len(natural):
            raise RuntimeError("candidate count mismatch: {}".format(candidate))
        for feature in PRIMARY_FEATURES:
            candidate_values = _finite(selected, feature)
            natural_values = _finite(natural, feature)
            clean_values = _finite(clean, feature)
            w1 = wasserstein_equal(candidate_values, natural_values)
            smd = 0.0 if candidate == "Clean" else (
                standardized_mean_difference(candidate_values, clean_values))
            output.append({
                "candidate": candidate,
                "coverage": 0.0 if candidate == "Clean"
                else CANDIDATE_COVERAGE[candidate],
                "feature": feature,
                "sample_count_each": len(selected),
                "wasserstein_candidate_natural": w1,
                "wasserstein_clean_natural": clean_w1[feature],
                "normalized_distance": w1 / (clean_w1[feature] + EPSILON),
                "candidate_closer_than_clean": w1 < clean_w1[feature],
                "smd_candidate_minus_clean": smd,
                "abs_smd_candidate_clean": abs(smd),
            })
    return output


def candidate_summary(distance_rows, candidates):
    output = []
    for candidate in candidates:
        selected = [row for row in distance_rows
                    if row["candidate"] == candidate]
        output.append({
            "candidate": candidate,
            "coverage": CANDIDATE_COVERAGE[candidate],
            "d_source": float(np.mean([
                float(row["normalized_distance"]) for row in selected])),
            "synthetic_closer_count_of_7": sum(
                str(row["candidate_closer_than_clean"]).lower() == "true"
                or row["candidate_closer_than_clean"] is True
                for row in selected),
            "large_shift_count_of_7": sum(
                float(row["abs_smd_candidate_clean"]) >= LARGE_SMD
                for row in selected),
            "mean_abs_smd_vs_clean": float(np.mean([
                float(row["abs_smd_candidate_clean"]) for row in selected])),
        })
    return output


def _load_candidate_prompts(prompt_path):
    artifact = np.load(str(prompt_path), allow_pickle=False)
    return {sample_id: prompt for sample_id, prompt in zip(
        artifact["sample_id"].astype(str).tolist(), artifact["prompt"])}


def paired_severity(rows, prompt_path, split, candidates):
    new_prompts = _load_candidate_prompts(prompt_path)
    old_prompt_artifact = np.load(
        str(P1_DIR / "prompt_features.npz"), allow_pickle=False)
    old_prompts = {sample_id: prompt for sample_id, prompt in zip(
        old_prompt_artifact["sample_id"].astype(str).tolist(),
        old_prompt_artifact["prompt"])}
    clean_by_id = {row["sample_id"]: row for row in _rows(rows, "Clean")}
    aggregates = []
    for candidate in candidates:
        values = {
            "delta_max_score": [], "delta_apce": [],
            "delta_score_entropy": [],
            "bbox_center_displacement_normalized": [],
            "bbox_center_displacement_pixels_256": [],
            "prompt_symmetric_best_match_cos": [],
        }
        selected = _rows(rows, candidate)
        for row in selected:
            clean = clean_by_id[row["source_sample_id"]]
            values["delta_max_score"].append(
                float(row["max_score"]) - float(clean["max_score"]))
            values["delta_apce"].append(
                float(row["apce"]) - float(clean["apce"]))
            values["delta_score_entropy"].append(
                float(row["score_entropy"]) - float(clean["score_entropy"]))
            clean_box = np.asarray([float(clean[name]) for name in (
                "pred_bbox_x", "pred_bbox_y", "pred_bbox_w", "pred_bbox_h")])
            candidate_box = np.asarray([float(row[name]) for name in (
                "pred_bbox_x", "pred_bbox_y", "pred_bbox_w", "pred_bbox_h")])
            clean_center = clean_box[:2] + 0.5 * clean_box[2:]
            candidate_center = candidate_box[:2] + 0.5 * candidate_box[2:]
            displacement = float(np.linalg.norm(candidate_center - clean_center))
            values["bbox_center_displacement_normalized"].append(displacement)
            values["bbox_center_displacement_pixels_256"].append(
                displacement * 256.0)
            metrics = paired_prompt_metrics(
                old_prompts[clean["sample_id"]], new_prompts[row["sample_id"]])
            values["prompt_symmetric_best_match_cos"].append(
                metrics["symmetric_best_match_cos"])
        record = {
            "split": split,
            "candidate": candidate,
            "coverage": CANDIDATE_COVERAGE[candidate],
            "pair_count": len(selected),
        }
        for name, vector in values.items():
            array = np.asarray(vector, dtype=np.float64)
            record[name + "_mean"] = float(np.mean(array))
            record[name + "_median"] = float(np.median(array))
            record[name + "_p10"] = float(np.quantile(array, 0.10))
            record[name + "_p90"] = float(np.quantile(array, 0.90))
        aggregates.append(record)
    return aggregates


def _atomic_artifacts(output_dir, artifacts, manifest_name=None,
                      manifest=None):
    destinations = [output_dir / name for name in artifacts]
    if manifest_name:
        destinations.append(output_dir / manifest_name)
    if any(path.exists() for path in destinations):
        raise RuntimeError("analysis destination already exists")
    staging = Path(tempfile.mkdtemp(prefix="d2-p2-analysis-", dir=str(output_dir)))
    try:
        hashes = {}
        for name, rows in artifacts.items():
            path = staging / name
            write_csv(path, rows)
            hashes[name] = {"rows": len(rows), "sha256": sha256_file(path)}
        if manifest_name:
            manifest = dict(manifest or {})
            manifest["artifacts"] = hashes
            path = staging / manifest_name
            path.write_text(json.dumps(
                manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name in artifacts:
            os.replace(str(staging / name), str(output_dir / name))
        if manifest_name:
            os.replace(str(staging / manifest_name),
                       str(output_dir / manifest_name))
    finally:
        try:
            staging.rmdir()
        except OSError:
            pass
    return hashes


def _git_source_commit():
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(ROOT), text=True)
    if status.strip():
        raise RuntimeError(
            "tracked source tree must be clean before Train selection")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def train_select(output_dir):
    output_dir = Path(output_dir).resolve()
    if any((output_dir / name).exists() for name in (
            "val_representation_features.csv",
            "val_representation_manifest.json",
            "val_candidate_prompts.npz")):
        raise RuntimeError("VAL artifact exists before Train selection freeze")
    manifest, rows, feature_path, prompt_path = _verify_representation(
        output_dir, "train")
    if set(manifest["candidates_run"]) != set(TRAIN_CANDIDATES):
        raise RuntimeError("Train candidate set differs from preregistration")
    distance_rows = candidate_distances(rows, TRAIN_CANDIDATES)
    summaries = candidate_summary(distance_rows, TRAIN_CANDIDATES)
    by_candidate = {row["candidate"]: row for row in summaries}
    selected_name = min(
        SELECTABLE,
        key=lambda name: (by_candidate[name]["d_source"],
                          CANDIDATE_COVERAGE[name]))
    selected = by_candidate[selected_name]
    p100 = by_candidate["P100"]
    selected["selected_train_only"] = True
    for row in summaries:
        row.setdefault("selected_train_only", False)
        row["mean_abs_smd_ratio_to_p100"] = (
            row["mean_abs_smd_vs_clean"]
            / max(p100["mean_abs_smd_vs_clean"], EPSILON))
    gate = {
        "closer_at_least_5_of_7": (
            selected["synthetic_closer_count_of_7"] >= 5),
        "d_source_below_p100": selected["d_source"] < p100["d_source"],
        "large_shift_count_below_p100": (
            selected["large_shift_count_of_7"]
            < p100["large_shift_count_of_7"]),
        "mean_abs_smd_below_p100": (
            selected["mean_abs_smd_vs_clean"]
            < p100["mean_abs_smd_vs_clean"]),
    }
    gate["pass"] = all(gate.values())
    paired = paired_severity(
        rows, prompt_path, "train", TRAIN_CANDIDATES)
    artifacts = {
        "train_distribution_distance.csv": distance_rows,
        "train_candidate_summary.csv": summaries,
        "train_paired_severity_analysis.csv": paired,
    }
    hashes = _atomic_artifacts(output_dir, artifacts)
    source_commit = _git_source_commit()
    selected_distance = [row for row in distance_rows
                         if row["candidate"] == selected_name]
    source = {
        "phase": "train_only_candidate_freeze_before_val",
        "candidate": selected_name,
        "coverage": CANDIDATE_COVERAGE[selected_name],
        "orientation_rule": {
            "namespace": ORIENTATION_NAMESPACE,
            "hash": "sha256(namespace + NUL + D2-P1 clean sample_id)",
            "mapping": ["left", "right", "top", "bottom"],
            "same_orientation_across_severities": True,
        },
        "fill_rule": {
            "value_normalized": FILL_VALUE_NORMALIZED,
            "mechanism": "single_contiguous_edge_anchored_block",
        },
        "train_d_source": selected["d_source"],
        "train_p100_d_source": p100["d_source"],
        "synthetic_closer_count_of_7": (
            selected["synthetic_closer_count_of_7"]),
        "large_shift_count_of_7": selected["large_shift_count_of_7"],
        "mean_abs_smd_vs_clean": selected["mean_abs_smd_vs_clean"],
        "mean_abs_smd_p100_vs_clean": p100["mean_abs_smd_vs_clean"],
        "feature_distances": selected_distance,
        "train_acceptance": gate,
        "source_code_commit": source_commit,
        "source_code_sha256": {
            "partial_helper": sha256_file(ROOT / (
                "tracking/target_prompt_d2_p2_partial_degradation.py")),
            "representation_runner": sha256_file(ROOT / (
                "tracking/run_target_prompt_d2_p2_representation.py")),
            "analysis_runner": sha256_file(ROOT / (
                "tracking/analyze_target_prompt_d2_p2_calibration.py")),
        },
        "train_artifact_sha256": {
            manifest["feature_file"]: manifest["feature_sha256"],
            manifest["prompt_file"]: manifest["prompt_sha256"],
            **{name: item["sha256"] for name, item in hashes.items()},
        },
        "val_accessed_for_selection": False,
        "p100_selectable": False,
    }
    selected_path = output_dir / "selected_source.json"
    selected_manifest_path = output_dir / "selected_source_manifest.json"
    if selected_path.exists() or selected_manifest_path.exists():
        raise RuntimeError("selected source destination exists")
    selected_path.write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_manifest = {
        "selected_source_file": selected_path.name,
        "selected_source_sha256": sha256_file(selected_path),
        "candidate": selected_name,
        "train_gate_pass": gate["pass"],
        "val_accessed": False,
    }
    selected_manifest_path.write_text(
        json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if not gate["pass"]:
        final_path = output_dir / "paired_severity_analysis.csv"
        if final_path.exists():
            raise RuntimeError("paired severity destination exists")
        write_csv(final_path, paired)
    print(json.dumps(source, indent=2, sort_keys=True))


def _selected_freeze(output_dir):
    source_path = output_dir / "selected_source.json"
    manifest_path = output_dir / "selected_source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(source_path) != manifest["selected_source_sha256"]:
        raise RuntimeError("selected source freeze mismatch")
    if not source["train_acceptance"]["pass"]:
        raise RuntimeError("Train gate failed; VAL analysis forbidden")
    for name, expected in source["train_artifact_sha256"].items():
        if sha256_file(output_dir / name) != expected:
            raise RuntimeError("Train artifact changed after selection: {}".format(name))
    return source, manifest


def _stratum_distance(rows, selected_name, field, value):
    groups = {}
    for candidate in ("Clean", "Natural", selected_name, "P100"):
        groups[candidate] = _rows(rows, candidate, **{field: value})
    counts = {len(group) for group in groups.values()}
    if len(counts) != 1 or next(iter(counts)) == 0:
        raise RuntimeError("unbalanced {} stratum {}".format(field, value))
    records = []
    ratios = []
    p100_ratios = []
    for feature in PRIMARY_FEATURES:
        natural = _finite(groups["Natural"], feature)
        clean = _finite(groups["Clean"], feature)
        selected = _finite(groups[selected_name], feature)
        p100 = _finite(groups["P100"], feature)
        clean_w1 = wasserstein_equal(clean, natural)
        selected_w1 = wasserstein_equal(selected, natural)
        p100_w1 = wasserstein_equal(p100, natural)
        ratio = selected_w1 / (clean_w1 + EPSILON)
        p100_ratio = p100_w1 / (clean_w1 + EPSILON)
        ratios.append(ratio)
        p100_ratios.append(p100_ratio)
        records.append({
            field: value,
            "selected_candidate": selected_name,
            "feature": feature,
            "sample_count_each": len(clean),
            "wasserstein_selected_natural": selected_w1,
            "wasserstein_clean_natural": clean_w1,
            "wasserstein_p100_natural": p100_w1,
            "normalized_distance_selected": ratio,
            "normalized_distance_p100": p100_ratio,
            "selected_closer_than_clean": selected_w1 < clean_w1,
        })
    d_source = float(np.mean(ratios))
    p100_d_source = float(np.mean(p100_ratios))
    for record in records:
        record["d_source_selected"] = d_source
        record["d_source_p100"] = p100_d_source
    return records, d_source


def val_holdout(output_dir):
    output_dir = Path(output_dir).resolve()
    source, selected_manifest = _selected_freeze(output_dir)
    selected_name = source["candidate"]
    train_manifest, train_rows, _, train_prompt_path = _verify_representation(
        output_dir, "train")
    val_manifest, val_rows, _, val_prompt_path = _verify_representation(
        output_dir, "val")
    if val_manifest["selected_source_sha256_verified"] != (
            selected_manifest["selected_source_sha256"]):
        raise RuntimeError("VAL did not bind frozen selected source")
    if set(val_manifest["candidates_run"]) != {selected_name, "P100"}:
        raise RuntimeError("VAL contains an unselected partial candidate")
    distance_rows = candidate_distances(val_rows, (selected_name, "P100"))
    summaries = candidate_summary(distance_rows, (selected_name, "P100"))
    summary_by_name = {row["candidate"]: row for row in summaries}
    selected = summary_by_name[selected_name]
    p100 = summary_by_name["P100"]
    val_gate = {
        "closer_at_least_4_of_7": (
            selected["synthetic_closer_count_of_7"] >= 4),
        "d_source_below_p100": selected["d_source"] < p100["d_source"],
    }
    val_gate["pass"] = all(val_gate.values())
    for row in summaries:
        row["frozen_selected_candidate"] = row["candidate"] == selected_name
        row["val_acceptance_pass"] = val_gate["pass"] \
            if row["candidate"] == selected_name else ""
    per_view = []
    per_target = []
    extreme = []
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for view in ("A", "B", "C"):
            records, d_source = _stratum_distance(
                rows, selected_name, "view", view)
            for record in records:
                record["split"] = split
            per_view.extend(records)
            if d_source >= 2.0:
                extreme.append({"split": split, "kind": "view",
                                "name": view, "d_source": d_source})
        targets = sorted({row["target_id"] for row in rows
                          if row["candidate"] == "Clean"})
        for target in targets:
            records, d_source = _stratum_distance(
                rows, selected_name, "target_id", target)
            for record in records:
                record["split"] = split
            per_target.extend(records)
            count = int(records[0]["sample_count_each"])
            if count >= 30 and d_source >= 2.0:
                extreme.append({"split": split, "kind": "target",
                                "name": target, "sample_count": count,
                                "d_source": d_source})
    train_paired = read_csv(output_dir / "train_paired_severity_analysis.csv")
    val_paired = paired_severity(
        val_rows, val_prompt_path, "val", (selected_name, "P100"))
    paired = train_paired + val_paired
    if val_gate["pass"]:
        frozen_case = "A"
        next_action = "A_D2-S1_source_only_training_requires_new_preregistration"
    else:
        frozen_case = "B"
        next_action = "B_redesign_degradation_mechanism"
    holdout_rows = []
    for row in summaries:
        holdout_rows.append({
            **row,
            "train_gate_pass": source["train_acceptance"]["pass"],
            "val_gate_pass": val_gate["pass"],
            "frozen_case": frozen_case,
            "robustness_extreme_warning": bool(extreme),
            "next_action": next_action,
        })
    artifacts = {
        "val_distribution_distance.csv": distance_rows,
        "val_holdout_summary.csv": holdout_rows,
        "paired_severity_analysis.csv": paired,
        "per_view_summary.csv": per_view,
        "per_target_summary.csv": per_target,
    }
    final_manifest = {
        "phase": "frozen_selected_val_holdout",
        "selected_source_sha256_verified": (
            selected_manifest["selected_source_sha256"]),
        "selected_candidate": selected_name,
        "train_gate_pass": source["train_acceptance"]["pass"],
        "val_gate": val_gate,
        "frozen_case": frozen_case,
        "next_action": next_action,
        "robustness_extreme_definition": (
            "view D_source>=2.0 or target with n>=30 and D_source>=2.0"),
        "robustness_extreme_warnings": extreme,
        "val_candidates_read": ["Clean", "Natural", selected_name, "P100"],
        "unselected_val_candidates_read": False,
        "official_test_accessed": False,
        "training_run": False,
    }
    _atomic_artifacts(
        output_dir, artifacts, manifest_name="analysis_manifest.json",
        manifest=final_manifest)
    print(json.dumps(final_manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=("train-select", "val-holdout"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if "test" in str(args.output_dir).lower():
        raise RuntimeError("test-named output path is forbidden")
    if args.phase == "train-select":
        train_select(args.output_dir)
    else:
        val_holdout(args.output_dir)


if __name__ == "__main__":
    main()
