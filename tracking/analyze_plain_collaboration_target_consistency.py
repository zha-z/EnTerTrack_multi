"""E2B prediction freeze and post-hoc target-consistency audit.

``freeze`` never opens a dataset or annotation.  It validates E2B shadow
identity against the frozen E1.5 rollout, derives strictly causal compact
prototype features, then writes their SHA256 manifest.  ``analyze`` verifies
that freeze before joining the already-frozen E1.5 labels and inner-val GT.
"""

import argparse
import csv
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

from tracking.analyze_plain_collaboration_safe_commit import iou_xywh, write_csv
from tracking.analyze_plain_collaboration_temporal_reliability import (
    _bool,
    _float,
    _macro_metrics,
    _metric_row,
    _pipeline,
    _resolve_path,
    _scoped_metrics,
    loto_probabilities,
    policy_choice,
    sha256_file,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
E15_PREDICTION_SHA256 = (
    "75ebb3cb292202346619e6f81cc46d050fa394a73e7c0d01ad9376545ba95f43")
VIEWS = ("A", "B", "C")
REPRESENTATIONS = ("weighted", "mean")
FORBIDDEN_DATASETS = {"threemdot", "threemdot_test", "three_mdot_test"}

S0 = (
    "sender_score", "sender_apce", "sender_entropy",
    "receiver_score", "receiver_apce", "receiver_entropy",
    "score_difference", "apce_difference", "entropy_difference",
)


def feature_sets(representation):
    prefix = representation + "_"
    sender_core = tuple(prefix + "sender_" + name for name in (
        "self_prev", "self_ema", "template_consistency"))
    cross = (prefix + "cross_cosine",)
    receiver_core = tuple(prefix + "receiver_" + name for name in (
        "self_prev", "self_ema", "template_consistency"))
    directional = (
        prefix + "sender_self_window",
        prefix + "receiver_self_window",
        prefix + "ema_difference",
        prefix + "template_difference",
        prefix + "cross_sender_template_interaction",
    )
    return {
        "S0": S0,
        "S1": sender_core,
        "S2": cross,
        "S3": sender_core + cross,
        "S4": sender_core + cross + receiver_core,
        "S5": sender_core + cross + receiver_core + directional,
    }


def _manifest_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sequence_name_from_file(path, suffix):
    name = Path(path).name
    if not name.endswith(suffix):
        raise ValueError("unexpected result filename: {}".format(name))
    return name[:-len(suffix)]


def _view_from_sequence(name):
    target, index = name.rsplit("-", 1)
    return target, {"1": "A", "2": "B", "3": "C"}[index]


def _cosine(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or not (
            np.isfinite(first).all() and np.isfinite(second).all()):
        return float("nan")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) \
        if denominator > 1e-12 else float("nan")


def _load_e15_predictions(path):
    path = _resolve_path(path)
    if sha256_file(path) != E15_PREDICTION_SHA256:
        raise RuntimeError("frozen E1.5 prediction SHA256 mismatch")
    rows = _read_csv(path)
    return path, rows


def _e15_maps(rows):
    branches = {}
    local_metrics = {}
    sequence_by_frame = {}
    for row in rows:
        key = (row["target_id"], row["receiver_view"], int(row["frame_id"]))
        branch_key = key + (row["branch_name"],)
        if branch_key in branches:
            raise RuntimeError("duplicate E1.5 branch row")
        branches[branch_key] = row
        sequence_by_frame[key] = row["sequence_name"]
        if row["branch_name"] == "local":
            local_metrics[key] = {
                "score": _float(row, "local_max_score"),
                "apce": _float(row, "local_apce"),
                "entropy": _float(row, "local_entropy"),
            }
    return branches, local_metrics, sequence_by_frame


def _compare_rollout_identity(results_dir, e15_rows):
    results_dir = _resolve_path(results_dir)
    suffix = "_plain_collaboration_sender_counterfactual.csv"
    current_files = sorted(results_dir.glob("*" + suffix))
    if len(current_files) != 15:
        raise RuntimeError("expected 15 E2B sender counterfactual files")
    current = []
    source_map = {
        (row["sequence_name"], row["target_id"], row["receiver_view"],
         int(row["frame_id"]), row["branch_name"]): row for row in e15_rows}
    bbox_columns = tuple(
        "{}_bbox_{}".format(prefix, field)
        for prefix in ("local", "branch") for field in ("x", "y", "w", "h"))
    state_mismatches = 0
    bbox_mismatches = 0
    uses_gt_rows = 0
    for path in current_files:
        sequence_name = _sequence_name_from_file(path, suffix)
        for row in _read_csv(path):
            row = {"sequence_name": sequence_name, **row}
            key = (sequence_name, row["target_id"], row["receiver_view"],
                   int(row["frame_id"]), row["branch_name"])
            source = source_map.get(key)
            if source is None:
                raise RuntimeError("E2B branch key is absent from E1.5")
            for name in bbox_columns:
                left, right = _float(row, name), _float(source, name)
                if not ((math.isnan(left) and math.isnan(right)) or left == right):
                    bbox_mismatches += 1
            if row["persistent_state_digest_before"] != \
                    row["persistent_state_digest_after"]:
                state_mismatches += 1
            uses_gt_rows += int(_bool(row["uses_gt"]))
            current.append(row)
    if len(current) != len(e15_rows):
        raise RuntimeError("E2B/E1.5 sender row count mismatch")
    if bbox_mismatches or state_mismatches or uses_gt_rows:
        raise RuntimeError(
            "shadow identity failed: bbox={} state={} uses_gt={}".format(
                bbox_mismatches, state_mismatches, uses_gt_rows))

    txt_mismatches = 0
    e15_results = _resolve_path(
        "output/test/tracking_results/entertrack/"
        "plain_collaboration_v1_e15_sender_counterfactual_28315")
    sequence_names = sorted({row["sequence_name"] for row in e15_rows})
    for sequence_name in sequence_names:
        current_path = results_dir / (sequence_name + ".txt")
        source_path = e15_results / current_path.name
        if not current_path.is_file() or not source_path.is_file() \
                or current_path.read_bytes() != source_path.read_bytes():
            txt_mismatches += 1
    if txt_mismatches:
        raise RuntimeError("saved bbox result mismatch vs E1.5: {}".format(
            txt_mismatches))
    return {
        "sender_rows": len(current),
        "branch_bbox_mismatches_vs_e15": bbox_mismatches,
        "persistent_state_mutations": state_mismatches,
        "uses_gt_true_rows": uses_gt_rows,
        "saved_bbox_file_mismatches_vs_e15": txt_mismatches,
    }


def _load_prototypes(results_dir, branches):
    results_dir = _resolve_path(results_dir)
    suffix = "_plain_collaboration_target_prototypes.npz"
    files = sorted(results_dir.glob("*" + suffix))
    if len(files) != 15:
        raise RuntimeError("expected 15 target prototype NPZ files")
    rows = []
    source_files = []
    for path in files:
        sequence_name = _sequence_name_from_file(path, suffix)
        expected_target, expected_view = _view_from_sequence(sequence_name)
        with np.load(path, allow_pickle=False) as data:
            required = (
                "target_id", "view_id", "frame_id", "uses_gt", "source_local",
                "search_token_count", "template_token_count", "token_dim",
                "temperature", "target_bbox", "response_weighted",
                "global_mean", "template_conditioned",
                "response_weighted_norm", "global_mean_norm",
                "template_conditioned_norm", "persistent_state_digest_before",
                "persistent_state_digest_after")
            missing = [name for name in required if name not in data]
            if missing:
                raise RuntimeError("prototype NPZ fields missing: {}".format(missing))
            arrays = {name: data[name] for name in required}
            count = len(arrays["frame_id"])
            if arrays["response_weighted"].shape != (count, 192) \
                    or arrays["global_mean"].shape != (count, 192) \
                    or arrays["template_conditioned"].shape != (count, 192):
                raise RuntimeError("prototype dimension must be 192")
            for index in range(count):
                target = str(arrays["target_id"][index])
                view = str(arrays["view_id"][index])
                frame = int(arrays["frame_id"][index])
                if target != expected_target or view != expected_view or frame != index:
                    raise RuntimeError("prototype identity/order mismatch")
                if bool(arrays["uses_gt"][index]) or not bool(
                        arrays["source_local"][index]):
                    raise RuntimeError("prototype is not prediction-only local")
                if (int(arrays["search_token_count"][index]),
                        int(arrays["template_token_count"][index]),
                        int(arrays["token_dim"][index])) != (256, 64, 192):
                    raise RuntimeError("prototype token layout mismatch")
                weighted = np.asarray(
                    arrays["response_weighted"][index], dtype=np.float32)
                mean = np.asarray(
                    arrays["global_mean"][index], dtype=np.float32)
                template = np.asarray(
                    arrays["template_conditioned"][index], dtype=np.float32)
                finite = (np.isfinite(weighted).all() and np.isfinite(mean).all()
                          and np.isfinite(template).all())
                if frame == 0 and finite:
                    raise RuntimeError("initial prototype must be placeholder")
                if frame > 0 and not finite:
                    raise RuntimeError("active prototype is non-finite")
                before = str(arrays[
                    "persistent_state_digest_before"][index])
                after = str(arrays[
                    "persistent_state_digest_after"][index])
                if before != after:
                    raise RuntimeError("prototype extraction mutated state")
                bbox = np.asarray(arrays["target_bbox"][index], dtype=float)
                source = branches[(target, view, frame, "local")]
                source_bbox = np.asarray([
                    _float(source, "local_bbox_" + field)
                    for field in ("x", "y", "w", "h")])
                if not np.allclose(bbox, source_bbox, rtol=0.0, atol=1e-4):
                    raise RuntimeError("prototype local bbox mismatch vs E1.5")
                rows.append({
                    "sequence_name": sequence_name,
                    "target_id": target,
                    "view_id": view,
                    "frame_id": frame,
                    "uses_gt": False,
                    "source_local": True,
                    "search_token_count": 256,
                    "template_token_count": 64,
                    "token_dim": 192,
                    "temperature": float(arrays["temperature"][index]),
                    "target_bbox": bbox.astype(np.float32),
                    "weighted": weighted,
                    "mean": mean,
                    "template": template,
                    "weighted_norm": float(arrays[
                        "response_weighted_norm"][index]),
                    "mean_norm": float(arrays["global_mean_norm"][index]),
                    "template_norm": float(arrays[
                        "template_conditioned_norm"][index]),
                    "state_before": before,
                    "state_after": after,
                })
        source_files.append({
            "path": _manifest_path(path), "rows": count,
            "sha256": sha256_file(path)})
    rows.sort(key=lambda row: (row["target_id"], row["view_id"], row["frame_id"]))
    return rows, source_files


def _self_features(prototypes):
    output = {}
    grouped = defaultdict(list)
    for row in prototypes:
        grouped[(row["target_id"], row["view_id"])].append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["frame_id"])
        histories = {name: [] for name in REPRESENTATIONS}
        emas = {name: None for name in REPRESENTATIONS}
        previous_frame = None
        for row in rows:
            frame = row["frame_id"]
            if previous_frame is not None and frame != previous_frame + 1:
                raise RuntimeError("prototype frame gap")
            values = {}
            for representation in REPRESENTATIONS:
                current = row[representation]
                history = histories[representation]
                values[representation] = {
                    "self_prev": _cosine(current, history[-1]) if history else float("nan"),
                    "self_ema": _cosine(current, emas[representation])
                    if emas[representation] is not None else float("nan"),
                    "self_window": _cosine(current, np.mean(history[-7:], axis=0))
                    if history else float("nan"),
                    "template_consistency": _cosine(current, row["template"]),
                }
                if np.isfinite(current).all():
                    if emas[representation] is None:
                        emas[representation] = current.astype(np.float64)
                    else:
                        emas[representation] = (
                            0.9 * emas[representation] + 0.1 * current)
                    history.append(current.astype(np.float64))
            output[(key[0], key[1], frame)] = values
            previous_frame = frame
    return output


def _bbox_columns(row, source, prefix):
    for field in ("x", "y", "w", "h"):
        row[prefix + "_bbox_" + field] = _float(
            source, "branch_bbox_" + field)


def _build_directional_features(prototypes, branches, local_metrics,
                                sequence_by_frame):
    prototype_map = {
        (row["target_id"], row["view_id"], row["frame_id"]): row
        for row in prototypes}
    self_map = _self_features(prototypes)
    output = []
    for receiver_key in sorted(prototype_map):
        target, receiver, frame = receiver_key
        receiver_proto = prototype_map[receiver_key]
        receiver_metrics = local_metrics[receiver_key]
        senders = tuple(view for view in VIEWS if view != receiver)
        for slot, sender in enumerate(senders):
            sender_key = (target, sender, frame)
            sender_proto = prototype_map[sender_key]
            sender_metrics = local_metrics[sender_key]
            row = {
                "sequence_name": sequence_by_frame[receiver_key],
                "target_id": target,
                "receiver_view": receiver,
                "sender_view": sender,
                "sender_slot": slot,
                "frame_id": frame,
                "uses_gt": False,
                "sender_score": sender_metrics["score"],
                "sender_apce": sender_metrics["apce"],
                "sender_entropy": sender_metrics["entropy"],
                "receiver_score": receiver_metrics["score"],
                "receiver_apce": receiver_metrics["apce"],
                "receiver_entropy": receiver_metrics["entropy"],
                "score_difference": sender_metrics["score"] - receiver_metrics["score"],
                "apce_difference": sender_metrics["apce"] - receiver_metrics["apce"],
                "entropy_difference": sender_metrics["entropy"] - receiver_metrics["entropy"],
            }
            for representation in REPRESENTATIONS:
                sender_self = self_map[sender_key][representation]
                receiver_self = self_map[receiver_key][representation]
                cross = _cosine(receiver_proto[representation],
                                sender_proto[representation])
                prefix = representation + "_"
                for name, value in sender_self.items():
                    row[prefix + "sender_" + name] = value
                for name, value in receiver_self.items():
                    row[prefix + "receiver_" + name] = value
                row[prefix + "cross_cosine"] = cross
                row[prefix + "ema_difference"] = (
                    sender_self["self_ema"] - receiver_self["self_ema"])
                row[prefix + "template_difference"] = (
                    sender_self["template_consistency"]
                    - receiver_self["template_consistency"])
                row[prefix + "cross_sender_template_interaction"] = (
                    cross * sender_self["template_consistency"])
            local_branch = branches[(target, receiver, frame, "local")]
            sender_branch = branches[(
                target, receiver, frame,
                "sender{}_only".format(slot))]
            both_branch = branches[(target, receiver, frame, "both")]
            _bbox_columns(row, local_branch, "local")
            _bbox_columns(row, sender_branch, "candidate")
            _bbox_columns(row, both_branch, "both")
            output.append(row)
    return output


def _write_combined_npz(path, prototypes):
    np.savez_compressed(
        path,
        sequence_name=np.asarray([row["sequence_name"] for row in prototypes]),
        target_id=np.asarray([row["target_id"] for row in prototypes]),
        view_id=np.asarray([row["view_id"] for row in prototypes]),
        frame_id=np.asarray([row["frame_id"] for row in prototypes], dtype=np.int64),
        uses_gt=np.zeros(len(prototypes), dtype=bool),
        source_local=np.ones(len(prototypes), dtype=bool),
        search_token_count=np.full(len(prototypes), 256, dtype=np.int64),
        template_token_count=np.full(len(prototypes), 64, dtype=np.int64),
        token_dim=np.full(len(prototypes), 192, dtype=np.int64),
        temperature=np.ones(len(prototypes), dtype=np.float32),
        target_bbox=np.stack([row["target_bbox"] for row in prototypes]),
        response_weighted=np.stack([row["weighted"] for row in prototypes]),
        global_mean=np.stack([row["mean"] for row in prototypes]),
        template_conditioned=np.stack([row["template"] for row in prototypes]),
        response_weighted_norm=np.asarray(
            [row["weighted_norm"] for row in prototypes], dtype=np.float32),
        global_mean_norm=np.asarray(
            [row["mean_norm"] for row in prototypes], dtype=np.float32),
        template_conditioned_norm=np.asarray(
            [row["template_norm"] for row in prototypes], dtype=np.float32),
        persistent_state_digest_before=np.asarray(
            [row["state_before"] for row in prototypes]),
        persistent_state_digest_after=np.asarray(
            [row["state_after"] for row in prototypes]),
    )


def freeze(results_dir, e15_predictions, output_dir):
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    e15_path, e15_rows = _load_e15_predictions(e15_predictions)
    branches, local_metrics, sequence_by_frame = _e15_maps(e15_rows)
    identity = _compare_rollout_identity(results_dir, e15_rows)
    prototypes, source_files = _load_prototypes(results_dir, branches)
    if len(prototypes) != 14112:
        raise RuntimeError("expected 14,112 view/frame prototypes")
    features = _build_directional_features(
        prototypes, branches, local_metrics, sequence_by_frame)
    if len(features) != 28224:
        raise RuntimeError("expected 28,224 directional prototype rows")
    if any(_bool(row["uses_gt"]) for row in features):
        raise RuntimeError("prediction feature unexpectedly uses GT")

    npz_path = output_dir / "prediction_only_target_prototypes.npz"
    feature_path = output_dir / "prediction_only_target_consistency_features.csv"
    _write_combined_npz(npz_path, prototypes)
    write_csv(feature_path, features)
    definitions = []
    for representation in REPRESENTATIONS:
        for group, names in feature_sets(representation).items():
            for name in names:
                definitions.append({
                    "representation": representation,
                    "feature_set": group,
                    "feature_name": name,
                    "prediction_only": True,
                    "strictly_causal_history": "self_" in name or "ema" in name,
                    "primary": representation == "weighted" and group == "S5",
                })
    write_csv(output_dir / "prototype_definition.csv", definitions)
    manifest = {
        "schema_version": 1,
        "phase": "prediction_freeze_before_gt_join",
        "uses_gt": False,
        "dataset_rollout": "threemdot_val",
        "official_test_used": False,
        "results_dir": _manifest_path(_resolve_path(results_dir)),
        "e15_prediction_file": _manifest_path(e15_path),
        "e15_prediction_sha256": E15_PREDICTION_SHA256,
        "prototype_file": _manifest_path(npz_path),
        "prototype_rows": len(prototypes),
        "prototype_sha256": sha256_file(npz_path),
        "feature_file": _manifest_path(feature_path),
        "feature_rows": len(features),
        "feature_sha256": sha256_file(feature_path),
        "target_count": len({row["target_id"] for row in prototypes}),
        "view_count": len({(row["target_id"], row["view_id"])
                           for row in prototypes}),
        "active_prototype_rows": sum(row["frame_id"] > 0 for row in prototypes),
        "initial_placeholder_rows": sum(row["frame_id"] == 0 for row in prototypes),
        "token_layout": {"template": 64, "search": 256, "dim": 192},
        "response_temperature": 1.0,
        "ema_beta": 0.9,
        "window": 8,
        "source_files": source_files,
        "identity": identity,
    }
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _join_labels(features, labels, feature_sha):
    label_map = {
        (row["target_id"], row["receiver_view"], int(row["frame_id"])): row
        for row in labels}
    output = []
    for row in features:
        key = (row["target_id"], row["receiver_view"], int(row["frame_id"]))
        label = label_map[key]
        slot = int(row["sender_slot"])
        output.append({
            **row,
            "target_visible": _bool(label["target_visible"]),
            "valid_for_analysis": _bool(label["valid_for_analysis"]),
            "delta_iou": float(label["delta_iou_sender{}".format(slot)]),
            "label": label["label_sender{}".format(slot)],
            "remote_help_available": _bool(label["remote_help_available"]),
            "feature_sha256": feature_sha,
            "source_prediction_sha256": label["prediction_sha256"],
        })
    return output


def _frames(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["target_id"], row["receiver_view"],
                 int(row["frame_id"]))].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["sender_slot"]))
        if [int(row["sender_slot"]) for row in values] != [0, 1]:
            raise RuntimeError("receiver frame lacks two senders")
        output.append({
            "target_id": key[0], "receiver_view": key[1],
            "frame_id": key[2], "sequence_name": values[0]["sequence_name"],
            "sender0": values[0], "sender1": values[1],
            "valid_for_analysis": bool(values[0]["valid_for_analysis"]),
        })
    return output


def _ranking(frames, names, with_identity):
    eligible = [frame for frame in frames
                if frame["valid_for_analysis"] and frame["frame_id"] > 0
                and abs(frame["sender0"]["delta_iou"]
                        - frame["sender1"]["delta_iou"]) > 0.02]
    base_names = tuple("diff_{}".format(index) for index in range(len(names)))
    identity_names = tuple(
        "{}_{}".format(role, view)
        for role in ("receiver", "sender0", "sender1") for view in VIEWS)
    model_names = base_names + (identity_names if with_identity else ())
    rows, labels = [], []
    for frame in eligible:
        row = {"target_id": frame["target_id"]}
        for index, name in enumerate(names):
            row[base_names[index]] = (
                _float(frame["sender0"], name) - _float(frame["sender1"], name))
        if with_identity:
            for view in VIEWS:
                row["receiver_" + view] = int(frame["receiver_view"] == view)
                row["sender0_" + view] = int(
                    frame["sender0"]["sender_view"] == view)
                row["sender1_" + view] = int(
                    frame["sender1"]["sender_view"] == view)
        rows.append(row)
        labels.append(int(frame["sender0"]["delta_iou"]
                          > frame["sender1"]["delta_iou"]))
    x = np.asarray([[_float(row, name) for name in model_names] for row in rows])
    y = np.asarray(labels, dtype=int)
    groups = np.asarray([row["target_id"] for row in rows])
    probabilities = np.full(len(rows), np.nan)
    for target in sorted(set(groups)):
        train, test = groups != target, groups == target
        swapped = x[train].copy()
        swapped[:, :len(base_names)] *= -1.0
        if with_identity:
            offset = len(base_names)
            sender0 = swapped[:, offset + 3:offset + 6].copy()
            swapped[:, offset + 3:offset + 6] = swapped[:, offset + 6:offset + 9]
            swapped[:, offset + 6:offset + 9] = sender0
        model = _pipeline()
        model.fit(np.concatenate((x[train], swapped)),
                  np.concatenate((y[train], 1 - y[train])))
        probabilities[test] = model.predict_proba(x[test])[:, 1]
    return eligible, y, probabilities, len(model_names)


def _policy(joined, names):
    train = [row for row in joined
             if row["valid_for_analysis"] and int(row["frame_id"]) > 0]
    labels = [int(row["label"] == "helpful") for row in train]
    probabilities = loto_probabilities(train, names, labels, predict_rows=joined)
    probability_map = {
        (row["target_id"], row["receiver_view"], int(row["frame_id"]),
         int(row["sender_slot"])): value
        for row, value in zip(joined, probabilities)}
    output = []
    for frame in _frames(joined):
        key = (frame["target_id"], frame["receiver_view"], frame["frame_id"])
        first, second = probability_map[key + (0,)], probability_map[key + (1,)]
        output.append({**frame, "probability_sender0": first,
                       "probability_sender1": second,
                       "selected_branch": policy_choice(
                           first, second, frame["frame_id"])})
    return output


def _box(row, prefix):
    return np.asarray([
        _float(row, prefix + "_bbox_" + name)
        for name in ("x", "y", "w", "h")]).astype(int).astype(float)


def _tracking(joined, policies, dataset):
    from lib.test.analysis.fcvc_results import _curves
    sequence_map = {sequence.name: sequence for sequence in dataset}
    grouped = defaultdict(list)
    for frame in _frames(joined):
        grouped[frame["sequence_name"]].append(frame)
    policy_maps = {
        name: {(row["target_id"], row["receiver_view"], row["frame_id"]): row
               for row in values} for name, values in policies.items()}
    variants = ("local", "both_safe", "e2b_mean_s5",
                "e2b_weighted_s5", "oracle_single")
    sequence_rows = []
    for name, frames in sorted(grouped.items()):
        sequence = sequence_map[name]
        target = np.asarray(sequence.ground_truth_rect, dtype=float).reshape(-1, 4)
        visible_data = getattr(sequence, "target_visible", None)
        visible = None if visible_data is None else np.asarray(
            visible_data).reshape(-1).astype(bool)
        frames.sort(key=lambda row: row["frame_id"])
        predictions = {variant: [] for variant in variants}
        for frame in frames:
            first, second = frame["sender0"], frame["sender1"]
            local = _box(first, "local")
            sender0, sender1 = _box(first, "candidate"), _box(second, "candidate")
            both = _box(first, "both")
            candidates = {"local": local, "sender0_only": sender0,
                          "sender1_only": sender1}
            predictions["local"].append(local)
            predictions["both_safe"].append(both)
            key = (frame["target_id"], frame["receiver_view"], frame["frame_id"])
            for representation in REPRESENTATIONS:
                choice = policy_maps[representation][key]["selected_branch"]
                predictions["e2b_{}_s5".format(representation)].append(
                    candidates[choice])
            is_visible = True if visible is None else bool(visible[frame["frame_id"]])
            if is_visible:
                oracle = max((local, sender0, sender1),
                             key=lambda box: iou_xywh(box, target[frame["frame_id"]]))
            else:
                oracle = local
            predictions["oracle_single"].append(oracle)
        for variant, boxes in predictions.items():
            sequence_rows.append({
                "sequence_name": name,
                "target_id": name.rsplit("-", 1)[0],
                "receiver_view": _view_from_sequence(name)[1],
                "variant": variant,
                **_curves(np.asarray(boxes), target,
                          target_visible=visible_data, dataset=sequence.dataset),
            })
    return sequence_rows


def analyze(output_dir, labels_path, dataset_name, e2a_summary):
    if dataset_name.lower() in FORBIDDEN_DATASETS or "test" in dataset_name.lower():
        raise RuntimeError("official/outer test analysis is forbidden")
    from lib.test.evaluation import get_dataset
    from sklearn import __version__ as sklearn_version

    output_dir = _resolve_path(output_dir)
    manifest = json.loads((output_dir / "prediction_manifest.json").read_text())
    feature_path = _resolve_path(manifest["feature_file"])
    prototype_path = _resolve_path(manifest["prototype_file"])
    feature_sha = sha256_file(feature_path)
    if feature_sha != manifest["feature_sha256"] \
            or sha256_file(prototype_path) != manifest["prototype_sha256"]:
        raise RuntimeError("prediction artifact SHA mismatch; GT join refused")
    features = _read_csv(feature_path)
    labels = _read_csv(_resolve_path(labels_path))
    if {row["prediction_sha256"] for row in labels} != {E15_PREDICTION_SHA256}:
        raise RuntimeError("post-hoc labels do not match frozen E1.5 prediction")
    joined = _join_labels(features, labels, feature_sha)
    label_fields = (
        "sequence_name", "target_id", "receiver_view", "sender_view",
        "sender_slot", "frame_id", "target_visible", "valid_for_analysis",
        "delta_iou", "label", "remote_help_available", "feature_sha256",
        "source_prediction_sha256")
    write_csv(output_dir / "posthoc_target_consistency_labels.csv", [
        {name: row[name] for name in label_fields} for row in joined])

    active = [row for row in joined
              if row["valid_for_analysis"] and int(row["frame_id"]) > 0]
    non_tie = [row for row in active if row["label"] in ("helpful", "harmful")]
    task_a_labels = np.asarray(
        [int(row["label"] == "helpful") for row in non_tie])
    ablation = []
    single_self, single_cross = [], []
    for representation in REPRESENTATIONS:
        for group, names in feature_sets(representation).items():
            probability = loto_probabilities(non_tie, names, task_a_labels)
            ablation.extend(_scoped_metrics(
                "A_helpful_vs_harmful", representation + "_" + group,
                non_tie, task_a_labels, probability, active, len(names)))
            task_b_labels = np.asarray(
                [int(row["label"] == "helpful") for row in active])
            task_b_probability = loto_probabilities(active, names, task_b_labels)
            ablation.extend(_scoped_metrics(
                "B_replace_local", representation + "_" + group,
                active, task_b_labels, task_b_probability, active, len(names)))
        for feature in feature_sets(representation)["S5"]:
            if feature in S0:
                continue
            probability = loto_probabilities(non_tie, (feature,), task_a_labels)
            result = _metric_row(
                "A_helpful_vs_harmful", feature, "overall", task_a_labels,
                probability, len(non_tie), len(active), 1)
            (single_cross if "cross" in feature else single_self).append(result)
    write_csv(output_dir / "semantic_ablation_group_cv.csv", ablation)
    write_csv(output_dir / "self_consistency_analysis.csv", single_self)
    write_csv(output_dir / "cross_view_compatibility_analysis.csv", single_cross)

    frames = [frame for frame in _frames(joined)
              if frame["valid_for_analysis"] and frame["frame_id"] > 0]
    ranking = []
    for representation in REPRESENTATIONS:
        names = feature_sets(representation)["S5"]
        for with_identity in (False, True):
            eligible, values, probability, count = _ranking(
                frames, names, with_identity)
            ranking.extend(_scoped_metrics(
                "C_sender_ranking",
                representation + "_S5_" + (
                    "with_pair_identity" if with_identity else "no_identity"),
                eligible, values, probability, frames, count))
    write_csv(output_dir / "sender_ranking_group_cv.csv", ranking)

    policies = {
        representation: _policy(joined, feature_sets(representation)["S5"])
        for representation in REPRESENTATIONS}
    policy_rows = []
    for index, frame in enumerate(policies["weighted"]):
        row = {
            "sequence_name": frame["sequence_name"],
            "target_id": frame["target_id"],
            "receiver_view": frame["receiver_view"],
            "frame_id": frame["frame_id"],
            "sender0_view": frame["sender0"]["sender_view"],
            "sender1_view": frame["sender1"]["sender_view"],
            "weighted_probability_sender0": frame["probability_sender0"],
            "weighted_probability_sender1": frame["probability_sender1"],
            "weighted_selected_branch": frame["selected_branch"],
            "mean_probability_sender0": policies["mean"][index]["probability_sender0"],
            "mean_probability_sender1": policies["mean"][index]["probability_sender1"],
            "mean_selected_branch": policies["mean"][index]["selected_branch"],
            "uses_gt_at_decision": False,
            "safe_report_only": True,
            "threshold": 0.5,
            "feature_sha256": feature_sha,
        }
        for prefix in ("local", "both"):
            for field in ("x", "y", "w", "h"):
                row[prefix + "_bbox_" + field] = _float(
                    frame["sender0"], prefix + "_bbox_" + field)
        for field in ("x", "y", "w", "h"):
            row["sender0_bbox_" + field] = _float(
                frame["sender0"], "candidate_bbox_" + field)
            row["sender1_bbox_" + field] = _float(
                frame["sender1"], "candidate_bbox_" + field)
        policy_rows.append(row)
    write_csv(output_dir / "oof_policy_predictions.csv", policy_rows)

    sequence_rows = _tracking(joined, policies, get_dataset(dataset_name))
    summary = _macro_metrics(sequence_rows)
    e2a_rows = _read_csv(_resolve_path(e2a_summary))
    e2a = next(row for row in e2a_rows
               if row["variant"] == "policy_T4" and row.get("scope") == "all")
    summary.append({
        "scope": "all", "variant": "e2a_t4",
        "sequence_count": int(e2a["sequence_count"]),
        "auc": float(e2a["auc"]), "precision": float(e2a["precision"]),
        "normalized_precision": float(e2a["normalized_precision"]),
        "mean_iou": float(e2a["mean_iou"]),
        "auc_delta_vs_local": float(e2a["auc_delta_vs_local"]),
        "primary_policy": False, "gt_oracle": False,
    })
    per_view = _macro_metrics(sequence_rows, "receiver_view")
    per_target = _macro_metrics(sequence_rows, "target_id")
    write_csv(output_dir / "oof_tracking_summary.csv", summary)
    write_csv(output_dir / "oof_per_view.csv", per_view)
    write_csv(output_dir / "oof_per_target.csv", per_target)

    def metric(task, feature_set):
        return next(row for row in ablation
                    if row["task"] == task and row["feature_set"] == feature_set
                    and row["scope"] == "overall")

    def auc(variant):
        return float(next(row["auc"] for row in summary
                          if row["variant"] == variant))

    weighted_a = metric("A_helpful_vs_harmful", "weighted_S5")
    mean_a = metric("A_helpful_vs_harmful", "mean_S5")
    local_auc, weighted_auc = auc("local"), auc("e2b_weighted_s5")
    mean_auc, oracle_auc = auc("e2b_mean_s5"), auc("oracle_single")
    target_weighted = [row for row in per_target
                       if row["variant"] == "e2b_weighted_s5"]
    worst_target = min(float(row["auc_delta_vs_local"])
                       for row in target_weighted)
    gates = {
        "primary_gain_ge_0_003": weighted_auc - local_auc >= 0.003,
        "semantic_roc_ge_0_65": float(weighted_a["roc_auc"]) >= 0.65,
        "semantic_pr_above_prior": (
            float(weighted_a["pr_auc"]) > float(weighted_a["positive_rate"])),
        "representation_roc_advantage_ge_0_03": (
            float(weighted_a["roc_auc"]) - float(mean_a["roc_auc"]) >= 0.03),
        "representation_tracking_advantage_ge_0_001": (
            weighted_auc - mean_auc >= 0.001),
        "safety_no_target_le_minus_0_05": worst_target > -0.05,
    }
    representation_success = (
        gates["representation_roc_advantage_ge_0_03"]
        and gates["representation_tracking_advantage_ge_0_001"])
    e2a_roc = float(next(row["roc_auc"] for row in _read_csv(
        _resolve_path(
            "docs/results/plain_collaboration_temporal_reliability_e2a_20260827/"
            "temporal_ablation_group_cv.csv"))
        if row["task"] == "A_helpful_vs_harmful"
        and row["feature_set"] == "T4" and row["scope"] == "overall"))
    primary = gates["primary_gain_ge_0_003"]
    semantic = gates["semantic_roc_ge_0_65"] and gates["semantic_pr_above_prior"]
    safety = gates["safety_no_target_le_minus_0_05"]
    if primary and semantic and safety:
        decision = "A"
    elif semantic and float(weighted_a["roc_auc"]) - e2a_roc >= 0.10 and not primary:
        decision = "B"
    elif representation_success and float(weighted_a["roc_auc"]) < 0.65:
        decision = "C"
    elif float(weighted_a["roc_auc"]) < 0.60 and not representation_success:
        decision = "D"
    else:
        decision = "INCONCLUSIVE_STOP"
    utilization = {
        "dataset": dataset_name,
        "local_auc": local_auc,
        "both_safe_auc": auc("both_safe"),
        "e2a_t4_auc": auc("e2a_t4"),
        "e2b_mean_s5_auc": mean_auc,
        "e2b_weighted_s5_auc": weighted_auc,
        "gt_oracle_single_auc": oracle_auc,
        "weighted_gain": weighted_auc - local_auc,
        "oracle_headroom": oracle_auc - local_auc,
        "oracle_utilization": ((weighted_auc - 0.648856) /
                               (0.662076 - 0.648856)),
        "artifact_oracle_utilization": ((weighted_auc - local_auc) /
                                        (oracle_auc - local_auc)),
        "task_a_weighted_s5_roc_auc": float(weighted_a["roc_auc"]),
        "task_a_weighted_s5_pr_auc": float(weighted_a["pr_auc"]),
        "task_a_positive_prior": float(weighted_a["positive_rate"]),
        "task_a_mean_s5_roc_auc": float(mean_a["roc_auc"]),
        "e2a_task_a_t4_roc_auc": e2a_roc,
        "worst_target_auc_delta": worst_target,
        "gates": gates,
        "representation_success": representation_success,
        "decision_case": decision,
        "gt_oracle_upper_bound_only": True,
    }
    (output_dir / "oracle_utilization.json").write_text(
        json.dumps(utilization, indent=2, sort_keys=True) + "\n")
    provenance = {
        "artifact_date": "2026-08-27",
        "branch": "feature/pcum-cross-layer-arp",
        "dataset": dataset_name,
        "checkpoint": {
            "path": "output/diagnostics/plain_collaboration_v1/"
                    "e1_run_20260827_seed42_4gpu_r001/checkpoints/train/"
                    "entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar",
            "sha256": "0607e8dcd05d9732a0058bb7a3d9e356474dbeb219a65e0472ac83a83d58ad40",
            "epoch": 25,
        },
        "prediction_manifest_sha256": sha256_file(
            output_dir / "prediction_manifest.json"),
        "feature_sha256_verified": feature_sha,
        "prototype_sha256_verified": manifest["prototype_sha256"],
        "model": "LogisticRegression C=1 balanced liblinear seed=42",
        "cv": "leave-one-target-out",
        "target_folds": 5,
        "safe_report_only": True,
        "runtime_uses_gt": False,
        "official_test_used": False,
        "training_run": False,
        "python": platform.python_version(),
        "sklearn": sklearn_version,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(utilization, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--results-dir", required=True)
    freeze_parser.add_argument("--e15-predictions", required=True)
    freeze_parser.add_argument("--output-dir", required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", required=True)
    analyze_parser.add_argument("--posthoc-labels", required=True)
    analyze_parser.add_argument("--dataset", required=True)
    analyze_parser.add_argument("--e2a-summary", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.results_dir, args.e15_predictions, args.output_dir)
    else:
        analyze(args.output_dir, args.posthoc_labels, args.dataset,
                args.e2a_summary)


if __name__ == "__main__":
    main()
