"""Prediction-freeze and post-hoc GT analysis for Plain Collaboration D0.

The ``freeze`` command never loads a dataset or ground truth.  The ``join``
command refuses to run unless the frozen prediction CSV still matches the
SHA256 recorded by ``freeze``.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FORBIDDEN_DATASETS = {"threemdot_test", "threemdot", "three_mdot_test"}
IDENTITY_COLUMNS = ("sequence_name", "target_id", "receiver_view", "frame_id")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        if not rows:
            raise ValueError("cannot infer columns from empty rows")
        columns = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_predictions(results_dir, output_dir):
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    sources = sorted(results_dir.glob(
        "*_plain_collaboration_counterfactual.csv"))
    if not sources:
        raise FileNotFoundError(
            "no prediction-only counterfactual logs in {}".format(results_dir))
    rows = []
    inventory = []
    seen = set()
    suffix = "_plain_collaboration_counterfactual.csv"
    for source in sources:
        sequence_name = source.name[:-len(suffix)]
        with source.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        for row in source_rows:
            if str(row.get("uses_gt", "")).lower() not in ("false", "0"):
                raise RuntimeError("runtime log contains uses_gt=true")
            forbidden_prediction_columns = {
                key for key in row
                if key.lower() != "uses_gt"
                and (key.lower().startswith("gt_")
                     or "ground_truth" in key.lower()
                     or key.lower() in {
                         "target_visible", "local_iou", "collaborative_iou",
                         "delta_iou", "label"})
            }
            if forbidden_prediction_columns:
                raise RuntimeError(
                    "prediction-only schema contains post-hoc columns: {}"
                    .format(sorted(forbidden_prediction_columns)))
            merged = {"sequence_name": sequence_name, **row}
            key = tuple(merged[name] for name in IDENTITY_COLUMNS)
            if key in seen:
                raise RuntimeError("duplicate prediction row {}".format(key))
            seen.add(key)
            rows.append(merged)
        inventory.append({
            "path": str(source),
            "rows": len(source_rows),
            "sha256": sha256_file(source),
        })
    rows.sort(key=lambda row: (
        row["target_id"], row["receiver_view"], int(row["frame_id"])))
    prediction_path = output_dir / "prediction_only_features.csv"
    write_csv(prediction_path, rows)
    manifest = {
        "schema_version": 1,
        "phase": "prediction_freeze_before_gt_join",
        "uses_gt": False,
        "results_dir": str(results_dir),
        "prediction_file": str(prediction_path),
        "prediction_rows": len(rows),
        "prediction_sha256": sha256_file(prediction_path),
        "source_files": inventory,
    }
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def iou_xywh(first, second):
    ax, ay, aw, ah = [float(value) for value in first]
    bx, by, bw, bh = [float(value) for value in second]
    if not all(math.isfinite(value) for value in (ax, ay, aw, ah, bx, by, bw, bh)):
        return float("nan")
    if min(aw, ah, bw, bh) <= 0:
        return float("nan")
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else float("nan")


def as_float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def roc_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = 0.5 * ((index + 1) + end)
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives)


def average_precision(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    true_positive = 0
    total = 0
    value = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        group_positive = sum(label for _, label in pairs[index:end])
        true_positive += group_positive
        total += end - index
        value += (group_positive / positives) * (true_positive / total)
        index = end
    return value


def feature_analysis(joined_rows):
    definitions = (
        ("local_max_score", "local_max_score", 1.0),
        ("local_apce", "local_apce", 1.0),
        ("local_entropy", "local_entropy", 1.0),
        ("receiver_need_by_score", "local_max_score", -1.0),
        ("receiver_need_by_apce", "local_apce", -1.0),
        ("receiver_need_by_entropy", "local_entropy", 1.0),
        ("sender_mean_score", "sender_mean_score", 1.0),
        ("sender_max_score", "sender_max_score", 1.0),
        ("sender_mean_apce", "sender_mean_apce", 1.0),
        ("sender_max_apce", "sender_max_apce", 1.0),
        ("sender_reliability_by_entropy", "sender_mean_entropy", -1.0),
        ("remote_minus_local_score", "remote_minus_local_score", 1.0),
        ("remote_minus_local_apce", "remote_minus_local_apce", 1.0),
        ("local_collab_center_displacement",
         "local_collab_center_displacement", 1.0),
        ("relative_residual_norm", "relative_residual_norm", 1.0),
    )
    output = []
    eligible = [row for row in joined_rows if row["label"] in ("helpful", "harmful")]
    valid_rows = [row for row in joined_rows if row["valid_for_analysis"]]
    for name, source, direction in definitions:
        values = [(direction * as_float(row, source), row["label"] == "helpful")
                  for row in eligible]
        values = [(score, int(label)) for score, label in values if math.isfinite(score)]
        scores = [item[0] for item in values]
        labels = [item[1] for item in values]
        helpful_values = [score for score, label in values if label]
        harmful_values = [score for score, label in values if not label]
        quantile_values = [
            (direction * as_float(row, source), row["label"])
            for row in valid_rows
            if math.isfinite(as_float(row, source))]
        q1 = float(np.quantile([item[0] for item in quantile_values], 0.25)) \
            if quantile_values else float("nan")
        q4 = float(np.quantile([item[0] for item in quantile_values], 0.75)) \
            if quantile_values else float("nan")
        low_group = [label for score, label in quantile_values if score <= q1]
        high_group = [label for score, label in quantile_values if score >= q4]

        def ratio(group, label):
            return sum(value == label for value in group) / len(group) \
                if group else float("nan")

        output.append({
            "feature": name,
            "source_column": source,
            "direction_multiplier": direction,
            "eligible_rows": len(values),
            "helpful_rows": sum(labels),
            "harmful_rows": len(labels) - sum(labels),
            "roc_auc_helpful_vs_harmful": roc_auc(scores, labels),
            "pr_auc_helpful_vs_harmful": average_precision(scores, labels),
            "mean_helpful": float(np.mean(helpful_values)) if helpful_values else float("nan"),
            "mean_harmful": float(np.mean(harmful_values)) if harmful_values else float("nan"),
            "q1_threshold_directed": q1,
            "q1_rows": len(low_group),
            "q1_helpful_ratio": ratio(low_group, "helpful"),
            "q1_harmful_ratio": ratio(low_group, "harmful"),
            "q4_threshold_directed": q4,
            "q4_rows": len(high_group),
            "q4_helpful_ratio": ratio(high_group, "helpful"),
            "q4_harmful_ratio": ratio(high_group, "harmful"),
        })
    return output


def summarize_labels(rows, group_name):
    output = []
    for value in sorted({row[group_name] for row in rows}):
        group = [row for row in rows if row[group_name] == value and row["valid_for_analysis"]]
        counts = {label: sum(row["label"] == label for row in group)
                  for label in ("helpful", "harmful", "tie")}
        output.append({
            group_name: value,
            "frame_count": len(group),
            **{"{}_count".format(key): item for key, item in counts.items()},
            **{"{}_ratio".format(key): item / len(group) if group else float("nan")
               for key, item in counts.items()},
            "mean_local_iou": float(np.mean([row["local_iou"] for row in group]))
                if group else float("nan"),
            "mean_collaborative_iou": float(np.mean(
                [row["collaborative_iou"] for row in group])) if group else float("nan"),
            "mean_delta_iou": float(np.mean([row["delta_iou"] for row in group]))
                if group else float("nan"),
        })
    return output


def join_ground_truth(output_dir, dataset_name):
    if dataset_name.lower() in FORBIDDEN_DATASETS or "test" in dataset_name.lower():
        raise RuntimeError("official/outer test GT join is forbidden")
    from lib.test.evaluation import get_dataset

    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = Path(manifest["prediction_file"])
    actual_sha = sha256_file(prediction_path)
    if actual_sha != manifest["prediction_sha256"]:
        raise RuntimeError("frozen prediction SHA256 mismatch; GT join refused")
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    sequence_map = {sequence.name: sequence for sequence in get_dataset(dataset_name)}
    labels = []
    joined = []
    for row in predictions:
        sequence = sequence_map.get(row["sequence_name"])
        if sequence is None:
            raise KeyError("sequence not in {}: {}".format(
                dataset_name, row["sequence_name"]))
        frame_id = int(row["frame_id"])
        ground_truth = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        if frame_id >= len(ground_truth):
            raise IndexError("frame outside GT: {} {}".format(sequence.name, frame_id))
        visible_data = getattr(sequence, "target_visible", None)
        visible = True if visible_data is None else bool(np.asarray(
            visible_data).reshape(-1)[frame_id])
        gt = ground_truth[frame_id]
        local_bbox = [as_float(row, "local_bbox_{}".format(key))
                      for key in ("x", "y", "w", "h")]
        collaborative_bbox = [as_float(row, "collaborative_bbox_{}".format(key))
                              for key in ("x", "y", "w", "h")]
        local_iou = iou_xywh(local_bbox, gt)
        collaborative_iou = iou_xywh(collaborative_bbox, gt)
        valid = bool(visible and math.isfinite(local_iou)
                     and math.isfinite(collaborative_iou))
        delta = collaborative_iou - local_iou if valid else float("nan")
        if not valid:
            label = "invalid"
        elif delta > 0.02:
            label = "helpful"
        elif delta < -0.02:
            label = "harmful"
        else:
            label = "tie"
        sender_scores = [as_float(row, "sender_{}_max_score".format(index))
                         for index in (0, 1)]
        sender_apces = [as_float(row, "sender_{}_apce".format(index))
                       for index in (0, 1)]
        sender_entropies = [as_float(row, "sender_{}_entropy".format(index))
                           for index in (0, 1)]
        def finite_mean(values):
            values = [value for value in values if math.isfinite(value)]
            return float(np.mean(values)) if values else float("nan")

        def finite_max(values):
            values = [value for value in values if math.isfinite(value)]
            return float(max(values)) if values else float("nan")

        derived = {
            "sender_mean_score": finite_mean(sender_scores),
            "sender_max_score": finite_max(sender_scores),
            "sender_mean_apce": finite_mean(sender_apces),
            "sender_max_apce": finite_max(sender_apces),
            "sender_mean_entropy": finite_mean(sender_entropies),
            "remote_minus_local_score": finite_mean(sender_scores)
                - as_float(row, "local_max_score"),
            "remote_minus_local_apce": finite_mean(sender_apces)
                - as_float(row, "local_apce"),
        }
        label_row = {
            **{key: row[key] for key in IDENTITY_COLUMNS},
            "target_visible": visible,
            "valid_for_analysis": valid,
            "local_iou": local_iou,
            "collaborative_iou": collaborative_iou,
            "delta_iou": delta,
            "label": label,
            "prediction_sha256": actual_sha,
        }
        labels.append(label_row)
        joined.append({**row, **derived, **label_row})
    write_csv(output_dir / "posthoc_gt_labels.csv", labels)
    write_csv(output_dir / "counterfactual_frame_metrics.csv", joined)
    reliability = feature_analysis(joined)
    write_csv(output_dir / "reliability_feature_analysis.csv", reliability)
    view_rows = summarize_labels(joined, "receiver_view")
    target_rows = summarize_labels(joined, "target_id")
    write_csv(output_dir / "counterfactual_per_view.csv", view_rows)
    write_csv(output_dir / "counterfactual_per_target.csv", target_rows)
    join_manifest = {
        "phase": "posthoc_gt_join",
        "dataset": dataset_name,
        "prediction_sha256_verified": actual_sha,
        "prediction_rows": len(predictions),
        "label_counts": {
            label: sum(row["label"] == label for row in labels)
            for label in ("helpful", "harmful", "tie", "invalid")
        },
        "helpful_threshold": 0.02,
        "harmful_threshold": -0.02,
    }
    (output_dir / "posthoc_join_manifest.json").write_text(
        json.dumps(join_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(join_manifest, indent=2, sort_keys=True))


def compare_summaries(output_dir, variants):
    output_dir = Path(output_dir)
    parsed = []
    for item in variants:
        name, path = item.split("=", 1)
        parsed.append((name, json.loads(Path(path).read_text(encoding="utf-8"))))
    comparison = []
    per_view = []
    per_target = []
    for variant, summary in parsed:
        comparison.append({"variant": variant, "scope": "overall", "group": "all",
                           **summary["overall"]})
        for view, metrics in summary["per_view"].items():
            row = {"variant": variant, "view": view, **metrics}
            per_view.append(row)
            comparison.append({"variant": variant, "scope": "view", "group": view,
                               **metrics})
        for target, metrics in summary["per_target"].items():
            row = {"variant": variant, "target_id": target, **metrics}
            per_target.append(row)
    write_csv(output_dir / "safe_commit_comparison.csv", comparison)
    write_csv(output_dir / "per_view_summary.csv", per_view)
    write_csv(output_dir / "per_target_summary.csv", per_target)


def audit_safe_commit(output_dir, local_results_dir, safe_results_dir):
    output_dir = Path(output_dir)
    prediction_path = output_dir / "prediction_only_features.csv"
    manifest = json.loads((output_dir / "prediction_manifest.json").read_text(
        encoding="utf-8"))
    actual_sha = sha256_file(prediction_path)
    if actual_sha != manifest["prediction_sha256"]:
        raise RuntimeError("frozen prediction SHA256 mismatch")
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row["sequence_name"], []).append(row)
    local_saved_mismatches = 0
    reported_saved_mismatches = 0
    state_local_mismatches = 0
    report_collab_mismatches = 0
    digest_mismatches = 0
    active_rows = 0
    for sequence_name, sequence_rows in by_sequence.items():
        sequence_rows.sort(key=lambda row: int(row["frame_id"]))
        local_saved = np.loadtxt(
            str(Path(local_results_dir) / (sequence_name + ".txt")),
            dtype=float).reshape(-1, 4)
        safe_saved = np.loadtxt(
            str(Path(safe_results_dir) / (sequence_name + ".txt")),
            dtype=float).reshape(-1, 4)
        if len(local_saved) != len(sequence_rows) or len(safe_saved) != len(sequence_rows):
            raise RuntimeError("bbox/log row mismatch for {}".format(sequence_name))
        for index, row in enumerate(sequence_rows):
            local = np.asarray([as_float(row, "local_bbox_{}".format(key))
                                for key in ("x", "y", "w", "h")])
            collaborative = np.asarray([
                as_float(row, "collaborative_bbox_{}".format(key))
                for key in ("x", "y", "w", "h")])
            state = np.asarray([as_float(row, "state_output_bbox_{}".format(key))
                                for key in ("x", "y", "w", "h")])
            reported = np.asarray([
                as_float(row, "reported_output_bbox_{}".format(key))
                for key in ("x", "y", "w", "h")])
            local_saved_mismatches += int(not np.array_equal(
                local.astype(int), local_saved[index].astype(int)))
            reported_saved_mismatches += int(not np.array_equal(
                collaborative.astype(int), safe_saved[index].astype(int)))
            state_local_mismatches += int(not np.allclose(state, local, atol=1e-9))
            report_collab_mismatches += int(not np.allclose(
                reported, collaborative, atol=1e-9))
            if int(row["frame_id"]) > 0:
                active_rows += 1
                digest_mismatches += int(
                    row["persistent_state_digest_before"]
                    != row["persistent_state_digest_after"])
    audit = {
        "uses_gt": False,
        "prediction_sha256_verified": actual_sha,
        "row_count": len(rows),
        "active_row_count": active_rows,
        "sequence_count": len(by_sequence),
        "persistent_mutation_during_collaboration_count": digest_mismatches,
        "state_output_not_local_count": state_local_mismatches,
        "reported_output_not_collaborative_count": report_collab_mismatches,
        "local_rollout_vs_logged_local_integer_mismatch_count": local_saved_mismatches,
        "safe_result_vs_logged_collaborative_integer_mismatch_count":
            reported_saved_mismatches,
        "all_active_search_tokens_256": all(
            int(row["search_token_count"]) == 256
            for row in rows if int(row["frame_id"]) > 0),
        "all_active_remote_count_2": all(
            int(row["valid_remote_count"]) == 2
            for row in rows if int(row["frame_id"]) > 0),
        "all_rows_uses_gt_false": all(
            str(row["uses_gt"]).lower() in ("false", "0") for row in rows),
        "all_rows_safe_commit_true": all(
            str(row["safe_commit"]).lower() in ("true", "1") for row in rows),
    }
    (output_dir / "safe_commit_integrity.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--results-dir", required=True)
    freeze.add_argument("--output-dir", required=True)
    join = commands.add_parser("join")
    join.add_argument("--output-dir", required=True)
    join.add_argument("--dataset", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--variant", action="append", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--local-results-dir", required=True)
    audit.add_argument("--safe-results-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_predictions(args.results_dir, args.output_dir)
    elif args.command == "join":
        join_ground_truth(args.output_dir, args.dataset)
    elif args.command == "compare":
        compare_summaries(args.output_dir, args.variant)
    else:
        audit_safe_commit(
            args.output_dir, args.local_results_dir, args.safe_results_dir)


if __name__ == "__main__":
    main()
