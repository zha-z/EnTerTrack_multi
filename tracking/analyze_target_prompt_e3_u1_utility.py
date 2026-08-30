"""Post-hoc GT join and target-LOTO analysis for frozen E3-U1 artifacts."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.analysis.fcvc_results import _curves  # noqa: E402
from lib.test.evaluation import get_dataset  # noqa: E402
from tracking.analyze_plain_collaboration_safe_commit import (  # noqa: E402
    average_precision,
    iou_xywh,
    roc_auc,
    write_csv,
)


BRANCHES = ("local", "sender0_only", "sender1_only", "both")
IDENTITY = ("sequence_name", "target_id", "receiver_view", "frame_id")
SENDER_IDENTITY = IDENTITY + ("sender_slot",)
FORBIDDEN_DATASETS = {"threemdot", "threemdot_test", "three_mdot_test"}

P0 = ("sender_score", "sender_apce")
P1 = P0 + (
    "prompt_topk_score_mean", "prompt_topk_score_std",
    "prompt_topk_score_min", "prompt_topk_score_max",
    "prompt_top1_top8_gap")
P2 = P1 + (
    "prompt_norm_mean", "prompt_norm_std",
    "prompt_pairwise_cos_mean", "prompt_pairwise_cos_std",
    "prompt_pairwise_cos_min", "prompt_pairwise_cos_max")
P3 = P2 + (
    "set_cos_mean", "set_cos_max", "sender_to_receiver_best_mean",
    "receiver_to_sender_best_mean", "symmetric_best_match")
P4 = P3 + (
    "branch_score", "branch_apce", "local_branch_center_displacement",
    "local_branch_scale_difference", "residual_norm",
    "relative_residual_norm", "residual_scale")
FEATURE_GROUPS = (("P0", P0), ("P1", P1), ("P2", P2),
                  ("P3", P3), ("P4", P4))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _bool(row, name):
    return str(row.get(name, "")).strip().lower() in ("1", "true", "yes")


def _bbox(row, prefix="branch", integer=False):
    box = np.asarray([_float(row, "{}_bbox_{}".format(prefix, name))
                      for name in ("x", "y", "w", "h")], dtype=float)
    return box.astype(int).astype(float) if integer else box


def _label(delta, valid):
    if not valid or not math.isfinite(delta):
        return "invalid"
    if delta > 0.02:
        return "helpful"
    if delta < -0.02:
        return "harmful"
    return "tie"


def _group(rows, identity, value_field):
    output = defaultdict(dict)
    for row in rows:
        key = tuple(row[name] for name in identity)
        value = row[value_field]
        if value in output[key]:
            raise RuntimeError("duplicate row {} {}".format(key, value))
        output[key][value] = row
    return output


def _verify_prediction_freeze(output_dir):
    manifest_path = output_dir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "threemdot_val" or manifest.get("uses_gt"):
        raise RuntimeError("invalid prediction manifest scope")
    branch_path = output_dir / manifest["branch_file"]
    feature_path = output_dir / manifest["prompt_feature_file"]
    if _sha256(branch_path) != manifest["branch_sha256"]:
        raise RuntimeError("frozen branch SHA256 mismatch")
    if _sha256(feature_path) != manifest["prompt_feature_sha256"]:
        raise RuntimeError("frozen prompt-feature SHA256 mismatch")
    failure_fields = (
        "local_forward_mismatch", "state_mutation",
        "both_report_mismatch", "local_state_mismatch", "uses_gt_rows")
    if any(int(manifest["runtime_audit"].get(name, 0))
           for name in failure_fields):
        raise RuntimeError("runtime audit is not clean")
    return manifest, branch_path, feature_path


def _join_labels(branch_groups, sequence_map, prediction_sha):
    rows = []
    labels = {}
    for key, branches in branch_groups.items():
        if set(branches) != set(BRANCHES):
            raise RuntimeError("incomplete four-branch group {}".format(key))
        sequence_name, target_id, receiver_view, frame_text = key
        sequence = sequence_map[sequence_name]
        frame_id = int(frame_text)
        target = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        visibility = getattr(sequence, "target_visible", None)
        visible = (True if visibility is None else bool(
            np.asarray(visibility).reshape(-1)[frame_id]))
        local = _bbox(branches["local"])
        if not all(np.allclose(
                _bbox(branches[name], "local"), local,
                rtol=0.0, atol=1e-9) for name in BRANCHES):
            raise RuntimeError("branch local candidates differ")
        boxes = {name: _bbox(branches[name]) for name in BRANCHES}
        ious = {name: iou_xywh(box, target[frame_id])
                for name, box in boxes.items()}
        valid = bool(visible and all(math.isfinite(value)
                                    for value in ious.values()))
        deltas = {name: (ious[name] - ious["local"] if valid
                         else float("nan"))
                  for name in ("sender0_only", "sender1_only", "both")}
        sender_views = {
            0: branches["sender0_only"]["selected_sender_views"],
            1: branches["sender1_only"]["selected_sender_views"],
        }
        row = {
            "sequence_name": sequence_name,
            "target_id": target_id,
            "receiver_view": receiver_view,
            "frame_id": frame_id,
            "sender0_view": sender_views[0],
            "sender1_view": sender_views[1],
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
            "remote_help_available": bool(
                valid and max(deltas["sender0_only"],
                              deltas["sender1_only"]) > 0.02),
            "prediction_branch_sha256": prediction_sha,
        }
        rows.append(row)
        labels[key] = row
    rows.sort(key=lambda row: (
        row["target_id"], row["receiver_view"], row["frame_id"]))
    return rows, labels


def _join_sender_features(feature_rows, labels):
    output = []
    seen = set()
    for row in feature_rows:
        identity = tuple(row[name] for name in IDENTITY)
        slot = int(row["sender_slot"])
        key = identity + (str(slot),)
        if key in seen:
            raise RuntimeError("duplicate prompt feature {}".format(key))
        seen.add(key)
        label_row = labels[identity]
        merged = dict(row)
        merged.update({
            "delta_iou": label_row[
                "delta_iou_sender{}".format(slot)],
            "label": label_row["label_sender{}".format(slot)],
            "valid_for_analysis": label_row["valid_for_analysis"],
        })
        output.append(merged)
    return output


def _fit_oof(records, feature_names, task):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score, precision_score, recall_score, roc_auc_score)

    active = [row for row in records
              if _bool(row, "valid_for_analysis")
              and int(row["frame_id"]) > 0]
    selected = ([row for row in active
                 if row["label"] in ("helpful", "harmful")]
                if task == "A_helpful_vs_harmful" else active)
    x = np.asarray([[_float(row, name) for name in feature_names]
                    for row in selected], dtype=float)
    y = np.asarray([int(row["label"] == "helpful")
                    for row in selected], dtype=int)
    groups = np.asarray([row["target_id"] for row in selected])
    oof = np.full(len(selected), np.nan, dtype=float)
    fold_rows = []
    for target in sorted(set(groups)):
        test = groups == target
        train = ~test
        if len(set(y[train])) < 2:
            continue
        train_x = x[train].copy()
        test_x = x[test].copy()
        with np.errstate(all="ignore"):
            medians = np.nanmedian(train_x, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        train_x = np.where(np.isfinite(train_x), train_x, medians)
        test_x = np.where(np.isfinite(test_x), test_x, medians)
        means = train_x.mean(axis=0)
        scales = train_x.std(axis=0)
        scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
        train_x = (train_x - means) / scales
        test_x = (test_x - means) / scales
        model = LogisticRegression(
            max_iter=2000, class_weight=None, random_state=42)
        model.fit(train_x, y[train])
        oof[test] = model.predict_proba(test_x)[:, 1]
        fold_rows.append({
            "held_out_target": target,
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "train_positive_prior": float(y[train].mean()),
            "all_missing_train_columns": int((~np.isfinite(
                x[train])).all(axis=0).sum()),
        })
    valid = np.isfinite(oof)
    labels_oof = y[valid]
    scores = oof[valid]
    predictions = scores >= 0.5
    summary = {
        "task": task,
        "feature_group": "",
        "feature_count": len(feature_names),
        "feature_names": "|".join(feature_names),
        "cv": "leave_one_target_out",
        "target_folds": len(set(groups)),
        "eligible_rows": len(selected),
        "oof_rows": int(valid.sum()),
        "positive_prior": float(labels_oof.mean()) if len(labels_oof) else float("nan"),
        "roc_auc": (float(roc_auc_score(labels_oof, scores))
                    if len(set(labels_oof)) == 2 else float("nan")),
        "pr_auc": (float(average_precision_score(labels_oof, scores))
                   if len(set(labels_oof)) == 2 else float("nan")),
        "precision_at_0_5": float(precision_score(
            labels_oof, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(
            labels_oof, predictions, zero_division=0)),
        "selection_rate_at_0_5": (float(predictions.mean())
                                    if len(predictions) else float("nan")),
        "uses_gt_at_runtime": False,
    }
    probability = {}
    for row, value in zip(selected, oof):
        key = tuple(row[name] for name in SENDER_IDENTITY)
        probability[key] = float(value)
    return summary, probability, fold_rows


def _feature_group_cv(sender_records):
    summaries = []
    probabilities = {}
    folds = []
    for group_name, names in FEATURE_GROUPS:
        for task in ("A_helpful_vs_harmful", "B_replace_local"):
            summary, current, fold_rows = _fit_oof(
                sender_records, names, task)
            summary["feature_group"] = group_name
            summaries.append(summary)
            if group_name == "P4" and task == "B_replace_local":
                probabilities = current
            for row in fold_rows:
                folds.append({
                    "feature_group": group_name,
                    "task": task,
                    **row,
                })
    return summaries, probabilities, folds


def _univariate_feature_analysis(sender_records):
    active = [row for row in sender_records
              if _bool(row, "valid_for_analysis")
              and int(row["frame_id"]) > 0]
    output = []
    for task in ("A_helpful_vs_harmful", "B_replace_local"):
        selected = ([row for row in active
                     if row["label"] in ("helpful", "harmful")]
                    if task.startswith("A_") else active)
        labels = [int(row["label"] == "helpful") for row in selected]
        for name in P4:
            pairs = [(_float(row, name), label)
                     for row, label in zip(selected, labels)]
            pairs = [value for value in pairs if math.isfinite(value[0])]
            scores = [value[0] for value in pairs]
            current_labels = [value[1] for value in pairs]
            output.append({
                "task": task,
                "feature": name,
                "rows": len(pairs),
                "positive_prior": (sum(current_labels) / len(current_labels)
                                   if current_labels else float("nan")),
                "roc_auc_raw_direction": roc_auc(scores, current_labels),
                "pr_auc_raw_direction": average_precision(
                    scores, current_labels),
            })
    return output


def _oof_policy(branch_groups, probability):
    rows = []
    boxes = {}
    counts = Counter()
    for key, branches in branch_groups.items():
        sequence_name, target_id, receiver_view, frame_id = key
        if int(frame_id) == 0:
            selected = "local"
            p0 = p1 = float("nan")
        else:
            p0 = probability.get(key + ("0",), float("nan"))
            p1 = probability.get(key + ("1",), float("nan"))
            if (not math.isfinite(p0) or not math.isfinite(p1)
                    or max(p0, p1) < 0.5 or p0 == p1):
                selected = "local"
            else:
                selected = "sender0_only" if p0 > p1 else "sender1_only"
        box = _bbox(branches[selected])
        boxes[key] = box
        counts[selected] += 1
        rows.append({
            "sequence_name": sequence_name,
            "target_id": target_id,
            "receiver_view": receiver_view,
            "frame_id": int(frame_id),
            "sender0_helpful_probability": p0,
            "sender1_helpful_probability": p1,
            "selected_branch": selected,
            "reported_bbox_x": box[0],
            "reported_bbox_y": box[1],
            "reported_bbox_w": box[2],
            "reported_bbox_h": box[3],
            "state_output_source": "local",
            "uses_gt_at_runtime": False,
        })
    return rows, boxes, counts


def _sequence_metrics(branch_groups, sequence_map, oof_boxes=None):
    by_sequence = defaultdict(list)
    for key, branches in branch_groups.items():
        by_sequence[key[0]].append((int(key[3]), key, branches))
    variants = [
        "local", "sender0_only", "sender1_only", "both",
        "oracle_single", "oracle_4"]
    if oof_boxes is not None:
        variants.insert(4, "oof_selector")
    variants = tuple(variants)
    output = []
    for sequence_name, frames in by_sequence.items():
        frames.sort(key=lambda item: item[0])
        sequence = sequence_map[sequence_name]
        target = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        visibility = getattr(sequence, "target_visible", None)
        visible = (None if visibility is None else
                   np.asarray(visibility).reshape(-1).astype(bool))
        prediction = {name: [] for name in variants}
        for frame_id, key, branches in frames:
            candidates = {name: _bbox(branches[name], integer=True)
                          for name in BRANCHES}
            prediction["local"].append(candidates["local"])
            prediction["sender0_only"].append(candidates["sender0_only"])
            prediction["sender1_only"].append(candidates["sender1_only"])
            prediction["both"].append(candidates["both"])
            if oof_boxes is not None:
                prediction["oof_selector"].append(
                    np.asarray(oof_boxes[key]).astype(int).astype(float))
            is_visible = True if visible is None else bool(visible[frame_id])
            if is_visible:
                ious = {name: iou_xywh(box, target[frame_id])
                        for name, box in candidates.items()}
                single = max(("local", "sender0_only", "sender1_only"),
                             key=lambda name: ious[name])
                oracle4 = max(BRANCHES, key=lambda name: ious[name])
            else:
                single = oracle4 = "local"
            prediction["oracle_single"].append(candidates[single])
            prediction["oracle_4"].append(candidates[oracle4])
        for variant in variants:
            metrics = _curves(
                np.asarray(prediction[variant]), target[:len(frames)],
                target_visible=(None if visibility is None else
                                np.asarray(visibility)[:len(frames)]),
                dataset=sequence.dataset)
            output.append({
                "sequence_name": sequence_name,
                "target_id": sequence_name.rsplit("-", 1)[0],
                "receiver_view": {"1": "A", "2": "B", "3": "C"}[
                    sequence_name.rsplit("-", 1)[1]],
                "variant": variant,
                **metrics,
            })
    return output


def _macro(sequence_rows, group_field=None):
    variants = tuple(sorted({row["variant"] for row in sequence_rows}))
    groups = ["all"] if group_field is None else sorted({
        row[group_field] for row in sequence_rows})
    output = []
    for group in groups:
        for variant in variants:
            selected = [row for row in sequence_rows
                        if row["variant"] == variant
                        and (group_field is None or row[group_field] == group)]
            output.append({
                "group_type": group_field or "overall",
                "group": group,
                "variant": variant,
                "sequence_count": len(selected),
                **{name: float(np.mean([row[name] for row in selected]))
                   for name in ("auc", "precision", "normalized_precision",
                                "mean_iou")},
            })
    return output


def _sender_summary(sender_records, sequence_rows):
    active = [row for row in sender_records
              if _bool(row, "valid_for_analysis")]
    local_auc_by_view = {
        view: np.mean([row["auc"] for row in sequence_rows
                       if row["receiver_view"] == view
                       and row["variant"] == "local"])
        for view in ("A", "B", "C")}
    output = []
    for receiver in ("A", "B", "C"):
        for sender in (view for view in ("A", "B", "C")
                       if view != receiver):
            rows = [row for row in active
                    if row["receiver_view"] == receiver
                    and row["sender_view"] == sender]
            slot = int(rows[0]["sender_slot"])
            variant = "sender{}_only".format(slot)
            sender_auc = np.mean([
                row["auc"] for row in sequence_rows
                if row["receiver_view"] == receiver
                and row["variant"] == variant])
            counts = Counter(row["label"] for row in rows)
            output.append({
                "receiver_sender": "{}<-{}".format(receiver, sender),
                "receiver_view": receiver,
                "sender_view": sender,
                "sender_slot": slot,
                "valid_rows": len(rows),
                "helpful_count": counts["helpful"],
                "harmful_count": counts["harmful"],
                "tie_count": counts["tie"],
                "helpful_ratio": counts["helpful"] / len(rows),
                "harmful_ratio": counts["harmful"] / len(rows),
                "tie_ratio": counts["tie"] / len(rows),
                "mean_delta_iou": float(np.mean([
                    float(row["delta_iou"]) for row in rows])),
                "local_auc": local_auc_by_view[receiver],
                "sender_auc": sender_auc,
                "auc_delta": sender_auc - local_auc_by_view[receiver],
            })
    return output


def _aggregation(labels):
    valid = [row for row in labels if _bool(row, "valid_for_analysis")]
    output = []
    for first in ("helpful", "harmful", "tie"):
        for second in ("helpful", "harmful", "tie"):
            rows = [row for row in valid
                    if row["label_sender0"] == first
                    and row["label_sender1"] == second]
            counts = Counter(row["label_both"] for row in rows)
            output.append({
                "sender0_label": first,
                "sender1_label": second,
                "frame_count": len(rows),
                "frame_ratio": len(rows) / len(valid),
                "both_helpful_count": counts["helpful"],
                "both_harmful_count": counts["harmful"],
                "both_tie_count": counts["tie"],
                "both_helpful_ratio": (counts["helpful"] / len(rows)
                                       if rows else float("nan")),
                "both_harmful_ratio": (counts["harmful"] / len(rows)
                                       if rows else float("nan")),
                "both_tie_ratio": (counts["tie"] / len(rows)
                                   if rows else float("nan")),
            })
    return output


def _active_branch_utility(branch_groups, label_map):
    output = []
    for key, branches in branch_groups.items():
        frame_id = int(key[3])
        label_row = label_map[key]
        if frame_id <= 0 or not _bool(label_row, "valid_for_analysis"):
            continue
        for branch, slot in (("sender0_only", 0),
                             ("sender1_only", 1), ("both", None)):
            prediction = branches[branch]
            delta_name = ("delta_iou_both" if slot is None else
                          "delta_iou_sender{}".format(slot))
            label_name = ("label_both" if slot is None else
                          "label_sender{}".format(slot))
            output.append({
                "sequence_name": key[0],
                "target_id": key[1],
                "receiver_view": key[2],
                "frame_id": frame_id,
                "branch": branch,
                "relative_residual_norm": _float(
                    prediction, "relative_residual_norm"),
                "delta_iou": _float(label_row, delta_name),
                "label": label_row[label_name],
                "used_remote": _bool(prediction, "used_remote"),
            })
    return output


def _utility_stats(records):
    counts = Counter(row["label"] for row in records)
    deltas = [float(row["delta_iou"]) for row in records]
    count = len(records)
    return {
        "frame_count": count,
        "helpful_count": counts["helpful"],
        "harmful_count": counts["harmful"],
        "tie_count": counts["tie"],
        "helpful_ratio": counts["helpful"] / count if count else float("nan"),
        "harmful_ratio": counts["harmful"] / count if count else float("nan"),
        "tie_ratio": counts["tie"] / count if count else float("nan"),
        "mean_delta_iou": float(np.mean(deltas)) if deltas else float("nan"),
        "mean_absolute_delta_iou": (float(np.mean(np.abs(deltas)))
                                    if deltas else float("nan")),
        "mean_relative_residual_norm": (float(np.mean([
            float(row["relative_residual_norm"]) for row in records]))
            if records else float("nan")),
    }


def _residual_quantiles(records):
    output = []
    for branch in ("sender0_only", "sender1_only", "both"):
        selected = [row for row in records if row["branch"] == branch]
        values = np.asarray([
            float(row["relative_residual_norm"]) for row in selected],
            dtype=float)
        cuts = np.quantile(values, (0.25, 0.50, 0.75), method="linear")
        bins = [[] for _ in range(4)]
        for row in selected:
            index = int(np.searchsorted(
                cuts, float(row["relative_residual_norm"]), side="left"))
            bins[index].append(row)
        for index, current in enumerate(bins):
            output.append({
                "branch": branch,
                "quantile": "Q{}".format(index + 1),
                "q25_cut": float(cuts[0]),
                "q50_cut": float(cuts[1]),
                "q75_cut": float(cuts[2]),
                **_utility_stats(current),
            })
    return output


def _cap_saturation(records, cap):
    output = []
    branch_records = {
        branch: [row for row in records if row["branch"] == branch]
        for branch in ("sender0_only", "sender1_only", "both")}
    branch_records["single_sender"] = (
        branch_records["sender0_only"] + branch_records["sender1_only"])
    for branch in ("sender0_only", "sender1_only", "single_sender", "both"):
        selected = branch_records[branch]
        for cap_hit in (False, True):
            current = [
                row for row in selected
                if (float(row["relative_residual_norm"]) >= cap - 1e-6)
                == cap_hit]
            output.append({
                "branch": branch,
                "cap_status": "cap_hit" if cap_hit else "non_cap_hit",
                "actual_relative_norm_cap": cap,
                "cap_hit_rule": "relative_residual_norm >= cap - 1e-6",
                "branch_frame_count": len(selected),
                "frame_ratio": (len(current) / len(selected)
                                if selected else float("nan")),
                **_utility_stats(current),
            })
    return output


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(records):
    pairs = [
        (float(row["relative_residual_norm"]), float(row["delta_iou"]))
        for row in records
        if math.isfinite(float(row["relative_residual_norm"]))
        and math.isfinite(float(row["delta_iou"]))]
    if len(pairs) < 2:
        return float("nan")
    x_rank = _average_ranks([item[0] for item in pairs])
    y_rank = _average_ranks([item[1] for item in pairs])
    if np.std(x_rank) < 1e-12 or np.std(y_rank) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _residual_correlations(records):
    groups = {
        "sender0_only": [row for row in records
                         if row["branch"] == "sender0_only"],
        "sender1_only": [row for row in records
                         if row["branch"] == "sender1_only"],
        "single_sender": [row for row in records
                          if row["branch"] in
                          ("sender0_only", "sender1_only")],
        "both": [row for row in records if row["branch"] == "both"],
    }
    return [{
        "branch": branch,
        "frame_count": len(selected),
        "spearman_relative_residual_norm_vs_delta_iou": _spearman(selected),
        "interpretation": "posthoc_descriptive_not_runtime_selector",
    } for branch, selected in groups.items()]


def _sender_rates(labels):
    valid = [row for row in labels if _bool(row, "valid_for_analysis")]
    sender_labels = [
        row[name] for row in valid
        for name in ("label_sender0", "label_sender1")]
    counts = Counter(sender_labels)
    delta_values = [
        _float(row, name) for row in valid
        for name in ("delta_iou_sender0", "delta_iou_sender1")]
    return {
        "valid_receiver_frames": len(valid),
        "valid_sender_rows": len(sender_labels),
        "remote_help_available_count": sum(
            _bool(row, "remote_help_available") for row in valid),
        "remote_help_available_ratio": (sum(
            _bool(row, "remote_help_available") for row in valid) / len(valid)),
        "helpful_count": counts["helpful"],
        "harmful_count": counts["harmful"],
        "tie_count": counts["tie"],
        "helpful_ratio": counts["helpful"] / len(sender_labels),
        "harmful_ratio": counts["harmful"] / len(sender_labels),
        "tie_ratio": counts["tie"] / len(sender_labels),
        "mean_delta_iou": float(np.mean(delta_values)),
    }


def _index_metrics(rows, group_type, group):
    return {
        row["variant"]: row for row in rows
        if row["group_type"] == group_type and row["group"] == group
    }


def _comparison_rows(e3_overall, d1_overall, e3_rates, d1_rates,
                     e3_utility, d1_utility):
    output = []

    def add(category, metric, e3, d1, unit):
        output.append({
            "category": category,
            "metric": metric,
            "e3_value": e3,
            "e3_d1_value": d1,
            "d1_minus_e3": d1 - e3,
            "unit": unit,
        })

    for variant in ("local", "sender0_only", "sender1_only", "both",
                    "oracle_single", "oracle_4"):
        add("tracking_auc", variant, float(e3_overall[variant]["auc"]),
            float(d1_overall[variant]["auc"]), "auc_fraction")
    add("tracking_auc", "oracle_single_headroom_vs_local",
        float(e3_overall["oracle_single"]["auc"])
        - float(e3_overall["local"]["auc"]),
        float(d1_overall["oracle_single"]["auc"])
        - float(d1_overall["local"]["auc"]), "auc_fraction")
    for name in ("remote_help_available_ratio", "helpful_ratio",
                 "harmful_ratio", "tie_ratio", "mean_delta_iou"):
        add("sender_utility", name, float(e3_rates[name]),
            float(d1_rates[name]), "fraction")
    for branch in ("sender0_only", "sender1_only", "both"):
        e3_selected = [row for row in e3_utility if row["branch"] == branch]
        d1_selected = [row for row in d1_utility if row["branch"] == branch]
        add("residual", "{}_mean_relative_residual_norm".format(branch),
            _utility_stats(e3_selected)["mean_relative_residual_norm"],
            _utility_stats(d1_selected)["mean_relative_residual_norm"],
            "fraction")
    return output


def _per_target_comparison(d1_rows, e3_rows):
    targets = sorted({row["group"] for row in d1_rows})
    output = []
    for target in targets:
        d1 = _index_metrics(d1_rows, "target_id", target)
        e3 = _index_metrics(e3_rows, "target_id", target)
        output.append({
            "target_id": target,
            "local_auc": float(d1["local"]["auc"]),
            "both_d1_auc": float(d1["both"]["auc"]),
            "oracle_single_d1_auc": float(d1["oracle_single"]["auc"]),
            "both_d1_minus_local": (float(d1["both"]["auc"])
                                     - float(d1["local"]["auc"])),
            "oracle_single_d1_minus_local": (
                float(d1["oracle_single"]["auc"])
                - float(d1["local"]["auc"])),
            "both_e3_auc": float(e3["both"]["auc"]),
            "oracle_single_e3_auc": float(e3["oracle_single"]["auc"]),
            "both_e3_minus_local": (float(e3["both"]["auc"])
                                    - float(e3["local"]["auc"])),
            "oracle_single_e3_minus_local": (
                float(e3["oracle_single"]["auc"])
                - float(e3["local"]["auc"])),
            "d1_minus_e3_both_auc": (float(d1["both"]["auc"])
                                      - float(e3["both"]["auc"])),
            "d1_minus_e3_oracle_single_auc": (
                float(d1["oracle_single"]["auc"])
                - float(e3["oracle_single"]["auc"])),
        })
    return output


def analyze_d1_descriptive(output_dir, dataset_name, e3_reference_dir):
    if dataset_name.lower() in FORBIDDEN_DATASETS or "test" in dataset_name.lower():
        raise RuntimeError("official test GT join is forbidden")
    output_dir = Path(output_dir).resolve()
    e3_reference_dir = Path(e3_reference_dir).resolve()

    manifest, branch_path, feature_path = _verify_prediction_freeze(output_dir)
    if manifest.get("tracker_param") != "target_prompt_collaboration_e3_d1":
        raise RuntimeError("D1 descriptive profile requires the frozen D1 tracker")
    cap = float(manifest["relative_norm_cap"])
    branch_rows = _read_csv(branch_path)
    feature_rows = _read_csv(feature_path)
    if len(branch_rows) != manifest["branch_rows"]:
        raise RuntimeError("branch row count mismatch")
    if len(feature_rows) != manifest["prompt_feature_rows"]:
        raise RuntimeError("prompt feature row count mismatch")
    branch_groups = _group(branch_rows, IDENTITY, "branch_name")

    # This is the first point at which GT is loaded. Prediction artifacts and
    # hashes have already been independently frozen and verified above.
    dataset = get_dataset(dataset_name)
    sequence_map = {sequence.name: sequence for sequence in dataset}
    labels, label_map = _join_labels(
        branch_groups, sequence_map, manifest["branch_sha256"])
    write_csv(output_dir / "posthoc_d1_sender_labels.csv", labels)
    sender_records = _join_sender_features(feature_rows, label_map)

    sequence_rows = _sequence_metrics(branch_groups, sequence_map)
    overall = _macro(sequence_rows)
    per_view = _macro(sequence_rows, "receiver_view")
    per_target = _macro(sequence_rows, "target_id")
    for rows in (overall, per_view, per_target):
        local_by_group = {row["group"]: row["auc"] for row in rows
                          if row["variant"] == "local"}
        for row in rows:
            row["auc_delta_vs_local"] = (
                row["auc"] - local_by_group[row["group"]])
            row["gt_oracle"] = row["variant"].startswith("oracle")
    d1_rates = _sender_rates(labels)
    for row in overall:
        row["remote_help_available_ratio"] = d1_rates[
            "remote_help_available_ratio"]
    write_csv(output_dir / "oracle_headroom_summary.csv", overall)
    write_csv(output_dir / "per_sender_summary.csv",
              _sender_summary(sender_records, sequence_rows))
    write_csv(output_dir / "aggregation_interaction_analysis.csv",
              _aggregation(labels))

    utility_records = _active_branch_utility(branch_groups, label_map)
    write_csv(output_dir / "residual_utility_quantiles.csv",
              _residual_quantiles(utility_records))
    write_csv(output_dir / "cap_saturation_utility.csv",
              _cap_saturation(utility_records, cap))
    write_csv(output_dir / "residual_delta_iou_correlation.csv",
              _residual_correlations(utility_records))

    e3_manifest, e3_branch_path, _ = _verify_prediction_freeze(
        e3_reference_dir)
    e3_overall_rows = _read_csv(
        e3_reference_dir / "oracle_headroom_summary.csv")
    e3_view_rows = _read_csv(e3_reference_dir / "oracle_per_view.csv")
    e3_target_rows = _read_csv(e3_reference_dir / "oracle_per_target.csv")
    e3_label_rows = _read_csv(
        e3_reference_dir / "posthoc_e3_sender_labels.csv")
    e3_rates = _sender_rates(e3_label_rows)
    e3_branch_rows = _read_csv(e3_branch_path)
    e3_branch_groups = _group(e3_branch_rows, IDENTITY, "branch_name")
    e3_label_map = {
        tuple(row[name] for name in IDENTITY): row
        for row in e3_label_rows}
    e3_utility_records = _active_branch_utility(
        e3_branch_groups, e3_label_map)

    d1_overall = _index_metrics(overall, "overall", "all")
    e3_overall = _index_metrics(e3_overall_rows, "overall", "all")
    comparison = _comparison_rows(
        e3_overall, d1_overall, e3_rates, d1_rates,
        e3_utility_records, utility_records)
    write_csv(output_dir / "e3_vs_d1_utility_summary.csv", comparison)

    variants = {
        "local", "sender0_only", "sender1_only", "both",
        "oracle_single", "oracle_4"}
    combined_view = []
    for family, rows in (("E3", e3_view_rows), ("E3-D1", per_view)):
        for row in rows:
            if row["variant"] not in variants:
                continue
            combined_view.append({"checkpoint_family": family, **row})
    write_csv(output_dir / "per_view_summary.csv", combined_view)
    write_csv(output_dir / "per_target_summary.csv",
              _per_target_comparison(per_target, e3_target_rows))

    produced = (
        "posthoc_d1_sender_labels.csv", "e3_vs_d1_utility_summary.csv",
        "per_sender_summary.csv", "per_view_summary.csv",
        "per_target_summary.csv", "oracle_headroom_summary.csv",
        "aggregation_interaction_analysis.csv",
        "residual_utility_quantiles.csv", "cap_saturation_utility.csv",
        "residual_delta_iou_correlation.csv")
    analysis_manifest = {
        "phase": "posthoc_gt_join_descriptive_only",
        "dataset": dataset_name,
        "profile": "d1_descriptive_no_selector",
        "prediction_branch_sha256_verified": manifest["branch_sha256"],
        "prediction_prompt_feature_sha256_verified": manifest[
            "prompt_feature_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "e3_reference_prediction_sha256_verified": e3_manifest[
            "branch_sha256"],
        "receiver_frame_groups": len(branch_groups),
        "valid_receiver_frames": d1_rates["valid_receiver_frames"],
        "valid_sender_rows": d1_rates["valid_sender_rows"],
        "remote_help_available_count": d1_rates[
            "remote_help_available_count"],
        "remote_help_available_ratio": d1_rates[
            "remote_help_available_ratio"],
        "helpful_threshold": 0.02,
        "harmful_threshold": -0.02,
        "actual_relative_norm_cap": cap,
        "cap_hit_tolerance": 1e-6,
        "logistic_regression_run": False,
        "selector_run": False,
        "uses_gt_at_runtime": False,
        "oracle_uses_gt_posthoc": True,
        "artifacts": {
            name: {"sha256": _sha256(output_dir / name),
                   "rows": len(_read_csv(output_dir / name))}
            for name in produced},
    }
    (output_dir / "posthoc_analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(analysis_manifest, indent=2, sort_keys=True))


def analyze(output_dir, dataset_name):
    if dataset_name.lower() in FORBIDDEN_DATASETS or "test" in dataset_name.lower():
        raise RuntimeError("official test GT join is forbidden")
    output_dir = Path(output_dir).resolve()
    manifest, branch_path, feature_path = _verify_prediction_freeze(output_dir)
    branch_rows = _read_csv(branch_path)
    feature_rows = _read_csv(feature_path)
    if len(branch_rows) != manifest["branch_rows"]:
        raise RuntimeError("branch row count mismatch")
    if len(feature_rows) != manifest["prompt_feature_rows"]:
        raise RuntimeError("prompt feature row count mismatch")
    branch_groups = _group(branch_rows, IDENTITY, "branch_name")
    dataset = get_dataset(dataset_name)
    sequence_map = {sequence.name: sequence for sequence in dataset}
    labels, label_map = _join_labels(
        branch_groups, sequence_map, manifest["branch_sha256"])
    write_csv(output_dir / "posthoc_e3_sender_labels.csv", labels)
    sender_records = _join_sender_features(feature_rows, label_map)

    cv_rows, p4_probability, fold_rows = _feature_group_cv(sender_records)
    write_csv(output_dir / "prompt_feature_group_cv.csv", cv_rows)
    write_csv(output_dir / "prompt_feature_analysis.csv",
              _univariate_feature_analysis(sender_records))
    write_csv(
        output_dir / "prompt_feature_loto_folds.csv", fold_rows,
        columns=("feature_group", "task", "held_out_target", "train_rows",
                 "test_rows", "train_positive_prior",
                 "all_missing_train_columns"))

    policy_rows, oof_boxes, selection_counts = _oof_policy(
        branch_groups, p4_probability)
    write_csv(output_dir / "oof_policy_predictions.csv", policy_rows)
    sequence_rows = _sequence_metrics(branch_groups, sequence_map, oof_boxes)
    overall = _macro(sequence_rows)
    per_view = _macro(sequence_rows, "receiver_view")
    per_target = _macro(sequence_rows, "target_id")
    local_auc = next(row["auc"] for row in overall
                     if row["variant"] == "local")
    for rows in (overall, per_view, per_target):
        local_by_group = {row["group"]: row["auc"] for row in rows
                          if row["variant"] == "local"}
        for row in rows:
            row["auc_delta_vs_local"] = (
                row["auc"] - local_by_group[row["group"]])
            row["gt_oracle"] = row["variant"].startswith("oracle")
    valid_labels = [row for row in labels if _bool(row, "valid_for_analysis")]
    remote_help = (sum(_bool(row, "remote_help_available")
                       for row in valid_labels) / len(valid_labels))
    for row in overall:
        row["remote_help_available_ratio"] = remote_help
    write_csv(output_dir / "oracle_headroom_summary.csv", overall)
    write_csv(output_dir / "oracle_per_view.csv", per_view)
    write_csv(output_dir / "oracle_per_target.csv", per_target)
    write_csv(output_dir / "oof_tracking_summary.csv",
              overall + per_view + per_target)
    write_csv(output_dir / "per_sender_summary.csv",
              _sender_summary(sender_records, sequence_rows))
    write_csv(output_dir / "aggregation_interaction_analysis.csv",
              _aggregation(labels))

    analysis_manifest = {
        "phase": "posthoc_gt_join_and_target_loto",
        "dataset": dataset_name,
        "prediction_branch_sha256_verified": manifest["branch_sha256"],
        "prediction_prompt_feature_sha256_verified": manifest[
            "prompt_feature_sha256"],
        "receiver_frame_groups": len(branch_groups),
        "valid_receiver_frames": len(valid_labels),
        "remote_help_available_ratio": remote_help,
        "helpful_threshold": 0.02,
        "harmful_threshold": -0.02,
        "oof_policy_feature_group": "P4",
        "oof_policy_threshold": 0.5,
        "oof_selection_counts": dict(selection_counts),
        "local_auc": local_auc,
        "uses_gt_at_runtime": False,
        "oracle_uses_gt_posthoc": True,
    }
    (output_dir / "posthoc_analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(analysis_manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="threemdot_val")
    parser.add_argument(
        "--profile", choices=("e3_selector", "d1_descriptive"),
        default="e3_selector",
        help="Default preserves the original E3-U1 selector analysis")
    parser.add_argument(
        "--e3-reference-dir", default=(
            "docs/results/"
            "target_prompt_collaboration_e3_u1_utility_audit_20260829"))
    args = parser.parse_args()
    if args.profile == "d1_descriptive":
        analyze_d1_descriptive(
            args.output_dir, args.dataset, args.e3_reference_dir)
    else:
        analyze(args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
