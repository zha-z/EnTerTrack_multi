"""Freeze and analyze E1.5 per-sender Plain Collaboration counterfactuals."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tracking.analyze_plain_collaboration_safe_commit import (
    average_precision,
    iou_xywh,
    roc_auc,
    sha256_file,
    write_csv,
)


BRANCHES = ("local", "sender0_only", "sender1_only", "both")
FORBIDDEN_DATASETS = {"threemdot", "threemdot_test", "three_mdot_test"}
IDENTITY = ("sequence_name", "target_id", "receiver_view", "frame_id")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _manifest_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _resolve_manifest_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _as_float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _bbox(row, prefix="branch", integer=False):
    values = np.asarray([
        _as_float(row, "{}_bbox_{}".format(prefix, key))
        for key in ("x", "y", "w", "h")
    ], dtype=float)
    return values.astype(int).astype(float) if integer else values


def _label(delta, valid):
    if not valid:
        return "invalid"
    if delta > 0.02:
        return "helpful"
    if delta < -0.02:
        return "harmful"
    return "tie"


def freeze_predictions(results_dir, output_dir):
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    suffix = "_plain_collaboration_sender_counterfactual.csv"
    sources = sorted(results_dir.glob("*" + suffix))
    if not sources:
        raise FileNotFoundError("no sender counterfactual logs")
    rows = []
    inventory = []
    seen = set()
    groups = defaultdict(set)
    for source in sources:
        sequence_name = source.name[:-len(suffix)]
        with source.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        for row in source_rows:
            if str(row.get("uses_gt", "")).lower() not in ("false", "0"):
                raise RuntimeError("runtime sender log contains uses_gt=true")
            forbidden = {
                key for key in row
                if key.lower() != "uses_gt"
                and (key.lower().startswith("gt_")
                     or "ground_truth" in key.lower()
                     or key.lower() in {
                         "target_visible", "iou", "delta_iou", "label"})
            }
            if forbidden:
                raise RuntimeError(
                    "prediction schema contains post-hoc columns: {}".format(
                        sorted(forbidden)))
            merged = {"sequence_name": sequence_name, **row}
            key = tuple(merged[name] for name in IDENTITY) + (
                merged["branch_name"],)
            if key in seen:
                raise RuntimeError("duplicate branch row {}".format(key))
            seen.add(key)
            group_key = tuple(merged[name] for name in IDENTITY)
            groups[group_key].add(merged["branch_name"])
            rows.append(merged)
        inventory.append({
            "path": _manifest_path(source),
            "rows": len(source_rows),
            "sha256": sha256_file(source),
        })
    incomplete = [key for key, value in groups.items() if value != set(BRANCHES)]
    if incomplete:
        raise RuntimeError("incomplete four-branch groups: {}".format(
            incomplete[:5]))
    rows.sort(key=lambda row: (
        row["target_id"], row["receiver_view"], int(row["frame_id"]),
        BRANCHES.index(row["branch_name"])))
    prediction_path = output_dir / "prediction_only_sender_counterfactual.csv"
    write_csv(prediction_path, rows)
    manifest = {
        "schema_version": 1,
        "phase": "prediction_freeze_before_gt_join",
        "uses_gt": False,
        "results_dir": _manifest_path(results_dir),
        "prediction_file": _manifest_path(prediction_path),
        "prediction_rows": len(rows),
        "receiver_frame_groups": len(groups),
        "branches_per_group": 4,
        "prediction_sha256": sha256_file(prediction_path),
        "source_files": inventory,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _group_prediction_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[name] for name in IDENTITY)
        grouped[key][row["branch_name"]] = row
    for key, branches in grouped.items():
        if set(branches) != set(BRANCHES):
            raise RuntimeError("branch group changed after freeze: {}".format(key))
    return grouped


def _metrics_by_sequence(grouped, sequence_map):
    from lib.test.analysis.fcvc_results import _curves

    by_sequence = defaultdict(list)
    for key, branches in grouped.items():
        by_sequence[key[0]].append((int(key[3]), branches))
    variants = (
        "local", "sender0_only", "sender1_only", "both",
        "oracle_single", "oracle_4")
    sequence_rows = []
    oracle_boxes = {}
    for sequence_name, frame_groups in by_sequence.items():
        sequence = sequence_map[sequence_name]
        target = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        visible_data = getattr(sequence, "target_visible", None)
        visible = (None if visible_data is None else
                   np.asarray(visible_data).reshape(-1).astype(bool))
        frame_groups.sort(key=lambda item: item[0])
        predictions = {name: [] for name in variants}
        for frame_id, branches in frame_groups:
            candidates = {
                name: _bbox(branches[name], integer=True)
                for name in ("local", "sender0_only", "sender1_only", "both")
            }
            is_visible = True if visible is None else bool(visible[frame_id])
            if is_visible:
                ious = {name: iou_xywh(box, target[frame_id])
                        for name, box in candidates.items()}
                single_name = max(
                    ("local", "sender0_only", "sender1_only"),
                    key=lambda name: ious[name])
                oracle_name = max(candidates, key=lambda name: ious[name])
            else:
                single_name = "local"
                oracle_name = "local"
            predictions["local"].append(candidates["local"])
            predictions["sender0_only"].append(candidates["sender0_only"])
            predictions["sender1_only"].append(candidates["sender1_only"])
            predictions["both"].append(candidates["both"])
            predictions["oracle_single"].append(candidates[single_name])
            predictions["oracle_4"].append(candidates[oracle_name])
        oracle_boxes[sequence_name] = predictions
        for variant in variants:
            metrics = _curves(
                np.asarray(predictions[variant]),
                target,
                target_visible=visible_data,
                dataset=sequence.dataset,
            )
            sequence_rows.append({
                "sequence_name": sequence_name,
                "target_id": sequence_name.rsplit("-", 1)[0],
                "receiver_view": {"1": "A", "2": "B", "3": "C"}[
                    sequence_name.rsplit("-", 1)[1]],
                "variant": variant,
                **metrics,
            })
    return sequence_rows, oracle_boxes


def _macro_summary(sequence_rows, group_field=None):
    variants = (
        "local", "sender0_only", "sender1_only", "both",
        "oracle_single", "oracle_4")
    groups = ["all"] if group_field is None else sorted({
        row[group_field] for row in sequence_rows})
    output = []
    for group in groups:
        for variant in variants:
            selected = [row for row in sequence_rows
                        if row["variant"] == variant
                        and (group_field is None or row[group_field] == group)]
            output.append({
                (group_field or "scope"): group,
                "variant": variant,
                "sequence_count": len(selected),
                **{name: float(np.mean([row[name] for row in selected]))
                   for name in ("auc", "precision", "normalized_precision",
                                "mean_iou")},
            })
    return output


def _sender_records(grouped, labels_by_key):
    records = []
    for key, branches in grouped.items():
        labels = labels_by_key[key]
        for index, branch_name in enumerate(("sender0_only", "sender1_only")):
            row = branches[branch_name]
            sender_prefix = "sender_0"
            records.append({
                **row,
                "sender_slot": index,
                "sender_view": row[sender_prefix + "_view"],
                "sender_max_score": _as_float(row, sender_prefix + "_max_score"),
                "sender_apce": _as_float(row, sender_prefix + "_apce"),
                "sender_entropy": _as_float(row, sender_prefix + "_entropy"),
                "sender_bbox_motion": _as_float(
                    row, sender_prefix + "_bbox_motion"),
                "sender_scale_change": _as_float(
                    row, sender_prefix + "_scale_change"),
                "sender_minus_receiver_score": _as_float(
                    row, sender_prefix + "_max_score")
                    - _as_float(row, "local_max_score"),
                "sender_minus_receiver_apce": _as_float(
                    row, sender_prefix + "_apce")
                    - _as_float(row, "local_apce"),
                "delta_iou": labels[
                    "delta_iou_sender{}".format(index)],
                "label": labels["label_sender{}".format(index)],
                "valid_for_analysis": labels["valid_for_analysis"],
            })
    return records


def _helpfulness_summary(sender_records):
    specs = [("overall", lambda row: "all")]
    specs += [
        ("receiver", lambda row: row["receiver_view"]),
        ("sender", lambda row: row["sender_view"]),
        ("receiver_sender", lambda row: "{}<-{}".format(
            row["receiver_view"], row["sender_view"])),
    ]
    output = []
    for group_type, key_fn in specs:
        keys = sorted({key_fn(row) for row in sender_records})
        for group in keys:
            rows = [row for row in sender_records
                    if key_fn(row) == group and row["valid_for_analysis"]]
            counts = {label: sum(row["label"] == label for row in rows)
                      for label in ("helpful", "harmful", "tie")}
            output.append({
                "group_type": group_type,
                "group": group,
                "valid_rows": len(rows),
                **{"{}_count".format(name): value
                   for name, value in counts.items()},
                **{"{}_ratio".format(name): value / len(rows)
                   if rows else float("nan")
                   for name, value in counts.items()},
                "mean_delta_iou": float(np.mean(
                    [row["delta_iou"] for row in rows]))
                    if rows else float("nan"),
            })
    return output


def _reliability_analysis(sender_records):
    features = (
        ("sender_score", "sender_max_score", 1.0),
        ("sender_apce", "sender_apce", 1.0),
        ("sender_low_entropy", "sender_entropy", -1.0),
        ("receiver_score", "local_max_score", 1.0),
        ("receiver_apce", "local_apce", 1.0),
        ("receiver_low_entropy", "local_entropy", -1.0),
        ("sender_minus_receiver_score", "sender_minus_receiver_score", 1.0),
        ("sender_minus_receiver_apce", "sender_minus_receiver_apce", 1.0),
    )
    valid = [row for row in sender_records
             if row["valid_for_analysis"] and int(row["frame_id"]) > 0]
    eligible = [row for row in valid if row["label"] in ("helpful", "harmful")]
    output = []
    for name, source, direction in features:
        values = [(direction * _as_float(row, source),
                   int(row["label"] == "helpful")) for row in eligible]
        values = [item for item in values if math.isfinite(item[0])]
        scores = [item[0] for item in values]
        labels = [item[1] for item in values]
        all_values = [(direction * _as_float(row, source), row["label"])
                      for row in valid]
        all_values = [item for item in all_values if math.isfinite(item[0])]
        threshold = float(np.quantile(
            [item[0] for item in all_values], 0.75)) if all_values else float("nan")
        high = [label for value, label in all_values if value >= threshold]
        output.append({
            "feature": name,
            "source_column": source,
            "reliability_direction": direction,
            "eligible_non_tie_rows": len(values),
            "helpful_rows": sum(labels),
            "harmful_rows": len(labels) - sum(labels),
            "roc_auc_helpful_vs_harmful": roc_auc(scores, labels),
            "pr_auc_helpful_vs_harmful": average_precision(scores, labels),
            "positive_class_prior": sum(labels) / len(labels)
                if labels else float("nan"),
            "high_reliability_threshold_directed": threshold,
            "high_reliability_rows": len(high),
            "p_helpful_given_high": sum(label == "helpful" for label in high)
                / len(high) if high else float("nan"),
            "p_harmful_given_high": sum(label == "harmful" for label in high)
                / len(high) if high else float("nan"),
        })
    return output


def _aggregation_analysis(grouped, labels_by_key):
    cases = (
        ("sender0_helpful_sender1_harmful", "helpful", "harmful"),
        ("sender0_harmful_sender1_helpful", "harmful", "helpful"),
        ("both_senders_helpful", "helpful", "helpful"),
        ("both_senders_harmful", "harmful", "harmful"),
    )
    output = []
    scopes = [("overall", "all")]
    scopes += [("receiver", view) for view in ("A", "B", "C")]
    for scope_type, scope in scopes:
        selected_keys = [key for key in grouped
                         if scope_type == "overall" or key[2] == scope]
        for case_name, first, second in cases:
            keys = [key for key in selected_keys
                    if labels_by_key[key]["label_sender0"] == first
                    and labels_by_key[key]["label_sender1"] == second]
            counts = {label: sum(
                labels_by_key[key]["label_both"] == label for key in keys)
                for label in ("helpful", "harmful", "tie")}
            output.append({
                "scope_type": scope_type,
                "scope": scope,
                "case": case_name,
                "frame_count": len(keys),
                **{"both_{}_count".format(name): value
                   for name, value in counts.items()},
                **{"both_{}_ratio".format(name): value / len(keys)
                   if keys else float("nan")
                   for name, value in counts.items()},
            })
        for branch_name, label_name in (
                ("sender0_only", "label_sender0"),
                ("sender1_only", "label_sender1"),
                ("both", "label_both")):
            keys = [key for key in selected_keys
                    if labels_by_key[key]["valid_for_analysis"]]
            counts = {label: sum(
                labels_by_key[key][label_name] == label for key in keys)
                for label in ("helpful", "harmful", "tie")}
            output.append({
                "scope_type": scope_type,
                "scope": scope,
                "case": "branch_summary_{}".format(branch_name),
                "frame_count": len(keys),
                **{"both_{}_count".format(name): value
                   for name, value in counts.items()},
                **{"both_{}_ratio".format(name): value / len(keys)
                   if keys else float("nan")
                   for name, value in counts.items()},
            })
    return output


def _selector_group_cv(sender_records):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feature_sets = (
        ("sender_only", (
            "sender_max_score", "sender_apce", "sender_entropy",
            "sender_bbox_motion", "sender_scale_change")),
        ("receiver_plus_sender", (
            "sender_max_score", "sender_apce", "sender_entropy",
            "sender_bbox_motion", "sender_scale_change",
            "local_max_score", "local_apce", "local_entropy",
            "sender_minus_receiver_score", "sender_minus_receiver_apce")),
        ("plus_post_fusion_disagreement", (
            "sender_max_score", "sender_apce", "sender_entropy",
            "sender_bbox_motion", "sender_scale_change",
            "local_max_score", "local_apce", "local_entropy",
            "sender_minus_receiver_score", "sender_minus_receiver_apce",
            "branch_max_score", "branch_apce", "branch_entropy",
            "center_displacement", "scale_difference", "score_delta",
            "apce_delta", "residual_norm", "relative_residual_norm")),
    )
    active = [row for row in sender_records
              if row["valid_for_analysis"] and int(row["frame_id"]) > 0]
    output = []
    for task in ("A_helpful_vs_harmful", "B_replace_local"):
        task_rows = ([row for row in active
                      if row["label"] in ("helpful", "harmful")]
                     if task.startswith("A_") else active)
        y = np.asarray([int(row["label"] == "helpful")
                        for row in task_rows], dtype=int)
        groups = np.asarray([row["target_id"] for row in task_rows])
        for set_name, names in feature_sets:
            x = np.asarray([[_as_float(row, name) for name in names]
                            for row in task_rows], dtype=float)
            oof = np.full(len(task_rows), np.nan, dtype=float)
            for target in sorted(set(groups)):
                test_mask = groups == target
                train_mask = ~test_mask
                if len(set(y[train_mask])) < 2:
                    continue
                model = Pipeline((
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("classifier", LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=1000,
                        random_state=42, solver="liblinear")),
                ))
                model.fit(x[train_mask], y[train_mask])
                oof[test_mask] = model.predict_proba(x[test_mask])[:, 1]
            valid_oof = np.isfinite(oof)
            labels = y[valid_oof]
            scores = oof[valid_oof]
            predicted = scores >= 0.5
            output.append({
                "task": task,
                "feature_set": set_name,
                "feature_count": len(names),
                "cv": "leave_one_target_out",
                "target_folds": len(set(groups)),
                "eligible_rows": len(task_rows),
                "oof_rows": int(valid_oof.sum()),
                "label_coverage": len(task_rows) / len(active)
                    if active else float("nan"),
                "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
                "roc_auc": float(roc_auc_score(labels, scores))
                    if len(set(labels)) == 2 else float("nan"),
                "pr_auc": float(average_precision_score(labels, scores))
                    if len(set(labels)) == 2 else float("nan"),
                "selection_coverage_at_0_5": float(predicted.mean())
                    if len(predicted) else float("nan"),
                "precision_at_0_5": float(precision_score(
                    labels, predicted, zero_division=0)),
                "recall_at_0_5": float(recall_score(
                    labels, predicted, zero_division=0)),
                "uses_gt_at_runtime": False,
                "posthoc_diagnostic_only": True,
            })
    return output


def join_and_analyze(output_dir, dataset_name):
    if dataset_name.lower() in FORBIDDEN_DATASETS or "test" in dataset_name.lower():
        raise RuntimeError("official/outer test GT join is forbidden")
    from lib.test.evaluation import get_dataset

    output_dir = Path(output_dir).resolve()
    manifest = json.loads((output_dir / "prediction_manifest.json").read_text(
        encoding="utf-8"))
    prediction_path = _resolve_manifest_path(manifest["prediction_file"])
    prediction_sha = sha256_file(prediction_path)
    if prediction_sha != manifest["prediction_sha256"]:
        raise RuntimeError("prediction SHA256 mismatch; GT join refused")
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    grouped = _group_prediction_rows(predictions)
    dataset = get_dataset(dataset_name)
    sequence_map = {sequence.name: sequence for sequence in dataset}
    labels = []
    labels_by_key = {}
    for key, branches in grouped.items():
        sequence_name, target_id, receiver_view, frame_text = key
        sequence = sequence_map[sequence_name]
        frame_id = int(frame_text)
        target = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        visible_data = getattr(sequence, "target_visible", None)
        visible = True if visible_data is None else bool(np.asarray(
            visible_data).reshape(-1)[frame_id])
        boxes = {name: _bbox(branches[name]) for name in BRANCHES}
        local_repeats_match = all(np.allclose(
            _bbox(branches[name], prefix="local"), boxes["local"], atol=1e-9)
            for name in BRANCHES)
        if not local_repeats_match:
            raise RuntimeError("branches do not share one local candidate")
        ious = {name: iou_xywh(box, target[frame_id])
                for name, box in boxes.items()}
        valid = bool(visible and all(math.isfinite(value) for value in ious.values()))
        deltas = {name: ious[name] - ious["local"] if valid else float("nan")
                  for name in ("sender0_only", "sender1_only", "both")}
        row = {
            "sequence_name": sequence_name,
            "target_id": target_id,
            "receiver_view": receiver_view,
            "frame_id": frame_id,
            "sender0_view": branches["sender0_only"]["sender_0_view"],
            "sender1_view": branches["sender1_only"]["sender_0_view"],
            "target_visible": visible,
            "valid_for_analysis": valid,
            "iou_local": ious["local"],
            "iou_sender0": ious["sender0_only"],
            "iou_sender1": ious["sender1_only"],
            "iou_both": ious["both"],
            "delta_iou_sender0": deltas["sender0_only"],
            "delta_iou_sender1": deltas["sender1_only"],
            "delta_iou_both": deltas["both"],
            "label_sender0": _label(deltas["sender0_only"], valid),
            "label_sender1": _label(deltas["sender1_only"], valid),
            "label_both": _label(deltas["both"], valid),
            "remote_help_available": bool(valid and max(deltas.values()) > 0.02),
            "prediction_sha256": prediction_sha,
        }
        labels.append(row)
        labels_by_key[key] = row
    write_csv(output_dir / "posthoc_sender_labels.csv", labels)

    sender_records = _sender_records(grouped, labels_by_key)
    write_csv(output_dir / "sender_helpfulness_summary.csv",
              _helpfulness_summary(sender_records))
    write_csv(output_dir / "sender_reliability_analysis.csv",
              _reliability_analysis(sender_records))
    write_csv(output_dir / "aggregation_interaction_analysis.csv",
              _aggregation_analysis(grouped, labels_by_key))

    sequence_rows, _ = _metrics_by_sequence(grouped, sequence_map)
    overall = _macro_summary(sequence_rows)
    local_auc = next(row["auc"] for row in overall if row["variant"] == "local")
    valid_labels = [row for row in labels if row["valid_for_analysis"]]
    remote_ratio = sum(row["remote_help_available"] for row in valid_labels) \
        / len(valid_labels)
    for row in overall:
        row["auc_delta_vs_local"] = row["auc"] - local_auc
        row["remote_help_available_ratio"] = remote_ratio
        row["gt_oracle"] = row["variant"].startswith("oracle")
    write_csv(output_dir / "oracle_headroom_summary.csv", overall)

    per_view = _macro_summary(sequence_rows, "receiver_view")
    per_target = _macro_summary(sequence_rows, "target_id")
    for rows, field in ((per_view, "receiver_view"),
                        (per_target, "target_id")):
        for row in rows:
            group_labels = [item for item in valid_labels
                            if item[field] == row[field]]
            row["remote_help_available_ratio"] = (
                sum(item["remote_help_available"] for item in group_labels)
                / len(group_labels) if group_labels else float("nan"))
            row["gt_oracle"] = row["variant"].startswith("oracle")
    write_csv(output_dir / "oracle_per_view.csv", per_view)
    write_csv(output_dir / "oracle_per_target.csv", per_target)
    write_csv(output_dir / "selector_group_cv.csv",
              _selector_group_cv(sender_records))

    join_manifest = {
        "phase": "posthoc_gt_join_and_analysis",
        "dataset": dataset_name,
        "prediction_sha256_verified": prediction_sha,
        "prediction_rows": len(predictions),
        "receiver_frame_groups": len(grouped),
        "valid_receiver_frames": len(valid_labels),
        "remote_help_available_ratio": remote_ratio,
        "helpful_threshold": 0.02,
        "harmful_threshold": -0.02,
        "oracle_is_gt_upper_bound_only": True,
    }
    (output_dir / "posthoc_analysis_manifest.json").write_text(
        json.dumps(join_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(join_manifest, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--results-dir", required=True)
    freeze.add_argument("--output-dir", required=True)
    join = commands.add_parser("join")
    join.add_argument("--output-dir", required=True)
    join.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_predictions(args.results_dir, args.output_dir)
    else:
        join_and_analyze(args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
