#!/usr/bin/env python3
"""Nested prediction-only utility audit for C3R Reliability v2 signal design."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/multi_agent_collaboration_clean/reliability_instrumentation"
OUT = ROOT / "output/multi_agent_collaboration_clean/reliability_v2_design"
SPECS = ROOT / "lib/train/data_specs/threemdot"
SEED = 20260716
WINDOWS = (3, 8, 16)
EPSILON = 0.01
GATE_REFERENCE_AUC = 0.484992
DRONES = {0: "A", 1: "B", 2: "C"}


def candidate_catalog():
    rows = []

    def add(name, group, kind, formula, window, state, availability="AVAILABLE"):
        rows.append({
            "signal_name": name,
            "group": group,
            "kind": kind,
            "formula": formula,
            "window_frames": window,
            "state_required": state,
            "availability": availability,
            "prediction_only": True,
            "uses_cross_camera_absolute_center": False,
        })

    for window in WINDOWS:
        add("local_center_velocity_w{}".format(window), "A", "temporal",
            "mean_w(||c_t-c_(t-1)||/diag(local_t))", window, "local boxes")
        add("local_center_acceleration_w{}".format(window), "A", "temporal",
            "mean_w(||v_t-v_(t-1)||)", window, "local boxes")
        add("local_scale_change_w{}".format(window), "A", "temporal",
            "mean_w(abs(log(area_t/area_(t-1))))", window, "local boxes")
        add("local_trajectory_error_w{}".format(window), "A", "temporal",
            "mean_w(||c_t-(2*c_(t-1)-c_(t-2))||/diag(local_t))",
            window, "local boxes")
        for quality in ("peak", "apce", "entropy", "margin"):
            add("local_{}_trend_w{}".format(quality, window), "A", "temporal",
                "(local_{}_t-local_{}_(t-w))/w".format(quality, quality),
                window, "local response-quality history")
    add("local_conf_drop_magnitude", "A", "temporal",
        "max(causal_EMA16(conf)_(t-1)-conf_t,0)", 16, "local confidence")
    add("local_conf_drop_duration", "A", "temporal",
        "run_length(conf_t < causal_EMA16(conf)_(t-1))", 16,
        "local confidence")
    add("local_score_peak_displacement", "A", "temporal",
        "distance(score_peak_t,score_peak_(t-1))", 1,
        "score-map peak coordinates", "MISSING_DIAGNOSTIC")

    for window in WINDOWS:
        add("remote_state_velocity_mean_w{}".format(window), "B", "temporal",
            "mean_senders(mean_w(remote normalized state speed))", window,
            "per-sender packet boxes")
        add("remote_state_velocity_max_w{}".format(window), "B", "temporal",
            "max_senders(mean_w(remote normalized state speed))", window,
            "per-sender packet boxes")
        add("remote_state_acceleration_max_w{}".format(window), "B", "temporal",
            "max_senders(mean_w(remote normalized state acceleration))", window,
            "per-sender packet boxes")
        add("remote_scale_change_max_w{}".format(window), "B", "temporal",
            "max_senders(mean_w(abs(remote log-area change)))", window,
            "per-sender packet boxes")
    for quality in ("peak", "apce", "entropy", "margin"):
        add("remote_{}_trend_abs_max_w8".format(quality), "B", "temporal",
            "max_senders(abs((remote_q_t-remote_q_(t-8))/8))", 8,
            "per-sender packet quality")
    add("remote_message_norm_change_max_w8", "B", "temporal",
        "max_senders(mean_8(abs(message_l2_t-message_l2_(t-1))))", 8,
        "per-sender message L2")
    add("remote_abrupt_change_max_w16", "B", "temporal",
        "max_senders(RMS causal clipped-z of state speed and quality changes)",
        16, "per-sender state/quality history")
    add("remote_valid_run_min", "B", "temporal",
        "min_senders(consecutive valid-message run)", 1,
        "per-sender valid history")
    add("message_age_mean", "B", "single_frame", "mean_senders(message_age)",
        1, "current packet metadata")
    add("message_age_change_max", "B", "temporal",
        "max_senders(abs(age_t-age_(t-1)))", 1,
        "previous packet metadata")
    add("remote_prompt_temporal_cosine", "B", "temporal",
        "cosine(remote_prompt_t,remote_prompt_(t-1))", 1,
        "previous remote prompt vector", "MISSING_DIAGNOSTIC")

    add("prompt_cosine_trend_abs_mean_w8", "C", "temporal",
        "mean_senders(abs((prompt_cos_t-prompt_cos_(t-8))/8))", 8,
        "same-frame local/remote prompt cosine history")
    for quality in ("peak", "apce", "entropy", "margin"):
        add("local_remote_{}_trend_gap_max_w8".format(quality), "C", "temporal",
            "max_senders(abs(local_q_trend8-remote_q_trend8))", 8,
            "local and sender quality histories")
    add("local_remote_state_change_gap_max_w8", "C", "temporal",
        "max_senders(abs(local speed_w8-remote speed_w8))", 8,
        "within-stream local and sender state histories")
    add("residual_norm_trend_abs_max_w8", "C", "temporal",
        "max_senders(abs((residual_l2_t-residual_l2_(t-8))/8))", 8,
        "per-sender residual-norm history")
    for window in WINDOWS:
        add("gated_residual_history_ratio_w{}".format(window), "C", "temporal",
            "sum_w(gate*residual_l2/local_feature_l2)", window,
            "per-sender gated-residual scalar history")
    add("multi_remote_state_speed_conflict", "C", "single_frame",
        "range_senders(remote state speed)", 1, "current sender histories")
    add("multi_remote_quality_conflict", "C", "single_frame",
        "mean range_senders(peak,APCE,entropy,margin)", 1,
        "current packet quality")
    add("multi_remote_prompt_cosine_conflict", "C", "single_frame",
        "range_senders(local/remote prompt cosine)", 1,
        "current prompt cosines")
    add("multi_remote_gate_weight_conflict", "C", "single_frame",
        "range_senders(normalized remote weight)", 1,
        "current normalized weights")
    add("residual_direction_temporal_consistency", "C", "temporal",
        "cosine(residual_t,residual_(t-1))", 1,
        "previous residual vector", "MISSING_DIAGNOSTIC")
    add("pairwise_remote_prompt_conflict", "C", "single_frame",
        "1-cosine(remote_prompt_1,remote_prompt_2)", 1,
        "current prompt vectors", "MISSING_DIAGNOSTIC")

    add("state_center_divergence", "D", "single_frame",
        "||center(C1)-center(local)||/diag(local)", 1,
        "current local and C1 boxes")
    add("state_scale_divergence", "D", "single_frame",
        "abs(log(area(C1)/area(local)))", 1,
        "current local and C1 boxes")
    for window in WINDOWS:
        add("state_center_divergence_trend_w{}".format(window), "D", "divergence",
            "(center_div_t-center_div_(t-w))/w", window,
            "divergence history")
        add("state_center_divergence_ema_w{}".format(window), "D", "divergence",
            "causal EMA_w(center divergence)", window, "divergence history")
        add("state_tracker_prediction_error_w{}".format(window), "D", "divergence",
            "mean_w(||center(C1)_t-local constant-velocity prediction_t||/diag(local_t))",
            window, "local/C1 box history")
        add("state_recovery_debt_w{}".format(window), "D", "divergence",
            "sum_w(max(center_div_t-center_div_(t-1),0))", window,
            "divergence history")
    add("state_center_divergence_cumulative_w16", "D", "divergence",
        "mean_16(center divergence)", 16, "divergence history")
    add("state_divergence_growth_run", "D", "divergence",
        "run_length(center_div_t>center_div_(t-1))", 1,
        "previous divergence")
    add("state_confidence_improvement", "D", "single_frame",
        "C1_confidence-local_confidence", 1, "current qualities")
    add("state_apce_improvement", "D", "single_frame",
        "C1_APCE-local_APCE", 1, "current qualities")
    add("state_quality_not_improved_duration", "D", "divergence",
        "run_length(center_div>0 and C1_confidence<=local_confidence)", 1,
        "box and confidence history")
    add("state_no_recovery_duration", "D", "divergence",
        "run_length(center_div_t>=center_div_(t-1))", 1,
        "divergence history")
    return rows


def read_target_set(path):
    return sorted({line.rsplit("-", 1)[0] for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()})


def build_nested_manifest():
    rows = []
    for outer_fold in range(5):
        train = read_target_set(SPECS / "c3r_f{}_train.txt".format(outer_fold))
        holdout = read_target_set(SPECS / "c3r_f{}_holdout.txt".format(outer_fold))
        hashes = {
            target: hashlib.sha256("{}|{}|{}".format(
                SEED, outer_fold, target).encode("utf-8")).hexdigest()
            for target in train
        }
        dev = set(sorted(train, key=lambda target: hashes[target])[:6])
        for target in train:
            rows.append({
                "outer_fold": outer_fold,
                "target_id": target,
                "inner_role": "inner_dev" if target in dev else "inner_train",
                "split_hash": hashes[target],
                "outer_holdout_targets": "|".join(holdout),
                "seed": SEED,
                "views_bound": "1|2|3",
            })
    result = pd.DataFrame(rows).sort_values(["outer_fold", "target_id"])
    for outer_fold, group in result.groupby("outer_fold"):
        holdout = set(group.outer_holdout_targets.iloc[0].split("|"))
        if holdout & set(group.target_id):
            raise RuntimeError("outer leakage in fold {}".format(outer_fold))
        if sum(group.inner_role == "inner_dev") != 6:
            raise RuntimeError("wrong inner-dev size")
    result.to_csv(OUT / "nested_target_manifest.csv", index=False)
    return result


def write_catalog():
    catalog = pd.DataFrame(candidate_catalog())
    catalog.insert(0, "signal_id", ["RV2_{:03d}".format(i + 1)
                                    for i in range(len(catalog))])
    catalog.to_csv(OUT / "candidate_signal_catalog.csv", index=False)
    return catalog


def run_length(condition):
    condition = np.asarray(condition, dtype=bool)
    result = np.zeros(len(condition), dtype=np.int64)
    count = 0
    for index, value in enumerate(condition):
        count = count + 1 if value else 0
        result[index] = count
    return result


def safe_div(numerator, denominator):
    return np.asarray(numerator, dtype=float) / np.maximum(
        np.asarray(denominator, dtype=float), 1e-8)


def compute_aggregate_features(aggregate):
    frames = []
    for (_, _), group in aggregate.groupby(
            ["sequence_id", "receiver_view"], sort=False):
        g = group.sort_values("frame_id").copy()
        lx = g.local_bbox_x.to_numpy(float)
        ly = g.local_bbox_y.to_numpy(float)
        lw = g.local_bbox_w.to_numpy(float)
        lh = g.local_bbox_h.to_numpy(float)
        cx = lx + 0.5 * lw
        cy = ly + 0.5 * lh
        c1x = g.c1_bbox_x.to_numpy(float) + 0.5 * g.c1_bbox_w.to_numpy(float)
        c1y = g.c1_bbox_y.to_numpy(float) + 0.5 * g.c1_bbox_h.to_numpy(float)
        diag = np.sqrt(np.maximum(lw * lw + lh * lh, 1e-8))
        area = np.maximum(lw * lh, 1e-8)
        c1area = np.maximum(
            g.c1_bbox_w.to_numpy(float) * g.c1_bbox_h.to_numpy(float), 1e-8)
        dx = pd.Series(cx).diff().to_numpy()
        dy = pd.Series(cy).diff().to_numpy()
        velocity = np.sqrt(dx * dx + dy * dy) / diag
        ndx, ndy = dx / np.maximum(lw, 1e-8), dy / np.maximum(lh, 1e-8)
        acceleration = np.sqrt(
            pd.Series(ndx).diff().to_numpy() ** 2
            + pd.Series(ndy).diff().to_numpy() ** 2)
        scale_step = np.abs(pd.Series(np.log(area)).diff().to_numpy())
        predicted_x = 2.0 * pd.Series(cx).shift(1) - pd.Series(cx).shift(2)
        predicted_y = 2.0 * pd.Series(cy).shift(1) - pd.Series(cy).shift(2)
        trajectory_error = np.sqrt(
            (cx - predicted_x.to_numpy()) ** 2
            + (cy - predicted_y.to_numpy()) ** 2) / diag
        for window in WINDOWS:
            g["local_center_velocity_w{}".format(window)] = pd.Series(
                velocity).rolling(window, min_periods=window).mean().to_numpy()
            g["local_center_acceleration_w{}".format(window)] = pd.Series(
                acceleration).rolling(window, min_periods=window).mean().to_numpy()
            g["local_scale_change_w{}".format(window)] = pd.Series(
                scale_step).rolling(window, min_periods=window).mean().to_numpy()
            g["local_trajectory_error_w{}".format(window)] = pd.Series(
                trajectory_error).rolling(window, min_periods=window).mean().to_numpy()
            for name, column in (
                    ("peak", "local_quality_00"), ("apce", "local_quality_01"),
                    ("entropy", "local_quality_02"), ("margin", "local_quality_03")):
                values = g[column].astype(float)
                g["local_{}_trend_w{}".format(name, window)] = (
                    values - values.shift(window)) / float(window)
        confidence = g.local_confidence.astype(float)
        prior_ema = confidence.shift(1).ewm(span=16, adjust=False, min_periods=3).mean()
        g["local_conf_drop_magnitude"] = np.maximum(prior_ema - confidence, 0.0)
        g["local_conf_drop_duration"] = run_length(confidence < prior_ema)

        center_div = np.sqrt((c1x - cx) ** 2 + (c1y - cy) ** 2) / diag
        scale_div = np.abs(np.log(c1area / area))
        g["state_center_divergence"] = center_div
        g["state_scale_divergence"] = scale_div
        for window in WINDOWS:
            series = pd.Series(center_div)
            g["state_center_divergence_trend_w{}".format(window)] = (
                ((series - series.shift(window)) / float(window)).to_numpy())
            g["state_center_divergence_ema_w{}".format(window)] = series.ewm(
                span=window, adjust=False, min_periods=window).mean().to_numpy()
            tracker_error = np.sqrt(
                (c1x - predicted_x.to_numpy()) ** 2
                + (c1y - predicted_y.to_numpy()) ** 2) / diag
            g["state_tracker_prediction_error_w{}".format(window)] = pd.Series(
                tracker_error).rolling(window, min_periods=window).mean().to_numpy()
            debt = np.maximum(series.diff(), 0.0)
            g["state_recovery_debt_w{}".format(window)] = debt.rolling(
                window, min_periods=window).sum().to_numpy()
        g["state_center_divergence_cumulative_w16"] = pd.Series(center_div).rolling(
            16, min_periods=16).mean().to_numpy()
        difference = pd.Series(center_div).diff()
        g["state_divergence_growth_run"] = run_length(difference > 0)
        g["state_confidence_improvement"] = (
            g.c1_confidence.astype(float) - g.local_confidence.astype(float))
        g["state_apce_improvement"] = (
            g.c1_apce.astype(float) - g.local_apce.astype(float))
        g["state_quality_not_improved_duration"] = run_length(
            (center_div > 0) & (g.state_confidence_improvement.to_numpy() <= 0))
        g["state_no_recovery_duration"] = run_length(difference >= 0)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def causal_abrupt_score(values):
    matrix = np.asarray(values, dtype=float)
    components = []
    for column in range(matrix.shape[1]):
        series = pd.Series(matrix[:, column])
        prior_mean = series.shift(1).rolling(16, min_periods=8).mean()
        prior_std = series.shift(1).rolling(16, min_periods=8).std(ddof=0)
        zscore = np.abs((series - prior_mean) / prior_std.replace(0, np.nan))
        components.append(np.minimum(zscore.to_numpy(), 10.0) ** 2)
    matrix = np.vstack(components)
    valid_count = np.isfinite(matrix).sum(axis=0)
    squared_mean = np.divide(
        np.nansum(matrix, axis=0), valid_count,
        out=np.full(matrix.shape[1], np.nan), where=valid_count > 0)
    return np.sqrt(squared_mean)


def compute_source_features(source, aggregate_features):
    rows = []
    for (_, _, _), group in source.groupby(
            ["sequence_id", "receiver_view", "sender_view"], sort=False):
        g = group.sort_values("frame_id").copy()
        cx = g.remote_bbox_x.astype(float)
        cy = g.remote_bbox_y.astype(float)
        width = np.maximum(g.remote_bbox_w.to_numpy(float), 1e-8)
        height = np.maximum(g.remote_bbox_h.to_numpy(float), 1e-8)
        scale = np.sqrt(width * height)
        dx, dy = cx.diff().to_numpy(), cy.diff().to_numpy()
        speed = np.sqrt(dx * dx + dy * dy) / scale
        acceleration = np.sqrt(
            pd.Series(dx / scale).diff().to_numpy() ** 2
            + pd.Series(dy / scale).diff().to_numpy() ** 2)
        scale_step = np.abs(pd.Series(np.log(width * height)).diff().to_numpy())
        quality_steps = []
        for name, column in (
                ("peak", "remote_quality_00"), ("apce", "remote_quality_01"),
                ("entropy", "remote_quality_02"), ("margin", "remote_quality_03")):
            values = g[column].astype(float)
            g["remote_{}_trend_w8".format(name)] = (values - values.shift(8)) / 8.0
            quality_steps.append(values.diff().abs().to_numpy())
        for window in WINDOWS:
            g["remote_state_velocity_w{}".format(window)] = pd.Series(speed).rolling(
                window, min_periods=window).mean().to_numpy()
            g["remote_state_acceleration_w{}".format(window)] = pd.Series(
                acceleration).rolling(window, min_periods=window).mean().to_numpy()
            g["remote_scale_change_w{}".format(window)] = pd.Series(scale_step).rolling(
                window, min_periods=window).mean().to_numpy()
        g["remote_message_norm_change_w8"] = g.remote_message_l2.astype(float).diff().abs().rolling(
            8, min_periods=8).mean()
        g["remote_abrupt_change_w16"] = causal_abrupt_score(
            np.column_stack([speed] + quality_steps))
        g["remote_valid_run"] = run_length(g.valid.astype(bool))
        g["message_age"] = g.message_age_intervals.astype(float)
        g["message_age_change"] = g.message_age_intervals.astype(float).diff().abs()
        prompt_cos = g.raw_input_08.astype(float)
        g["prompt_cosine_trend_abs_w8"] = ((prompt_cos - prompt_cos.shift(8)) / 8.0).abs()
        g["residual_norm_trend_abs_w8"] = (
            (g.adapted_residual_l2.astype(float)
             - g.adapted_residual_l2.astype(float).shift(8)) / 8.0).abs()
        ratio = safe_div(g.gate_times_residual_l2, g.local_feature_l2)
        for window in WINDOWS:
            g["gated_residual_history_ratio_w{}".format(window)] = pd.Series(
                ratio).rolling(window, min_periods=window).sum().to_numpy()
        g["remote_state_speed_current"] = speed
        rows.append(g)
    source_features = pd.concat(rows, ignore_index=True)

    merge_columns = ["sequence_id", "receiver_view", "frame_id",
                     "local_center_velocity_w8"]
    for quality in ("peak", "apce", "entropy", "margin"):
        merge_columns.append("local_{}_trend_w8".format(quality))
    source_features = source_features.merge(
        aggregate_features[merge_columns],
        on=["sequence_id", "receiver_view", "frame_id"], how="left",
        validate="many_to_one")
    for quality in ("peak", "apce", "entropy", "margin"):
        source_features["local_remote_{}_trend_gap_w8".format(quality)] = (
            source_features["local_{}_trend_w8".format(quality)]
            - source_features["remote_{}_trend_w8".format(quality)]).abs()
    source_features["local_remote_state_change_gap_w8"] = (
        source_features.local_center_velocity_w8
        - source_features.remote_state_velocity_w8).abs()

    keys = ["sequence_id", "receiver_view", "frame_id"]
    output = source_features[keys].drop_duplicates().copy()

    def aggregate_column(source_name, output_name, operation):
        grouped = source_features.groupby(keys, sort=False)[source_name]
        if operation == "mean":
            values = grouped.mean()
        elif operation == "max":
            values = grouped.max()
        elif operation == "min":
            values = grouped.min()
        elif operation == "range":
            values = grouped.max() - grouped.min()
        else:
            raise ValueError(operation)
        nonlocal output
        output = output.merge(values.rename(output_name).reset_index(), on=keys,
                              how="left", validate="one_to_one")

    for window in WINDOWS:
        aggregate_column("remote_state_velocity_w{}".format(window),
                         "remote_state_velocity_mean_w{}".format(window), "mean")
        aggregate_column("remote_state_velocity_w{}".format(window),
                         "remote_state_velocity_max_w{}".format(window), "max")
        aggregate_column("remote_state_acceleration_w{}".format(window),
                         "remote_state_acceleration_max_w{}".format(window), "max")
        aggregate_column("remote_scale_change_w{}".format(window),
                         "remote_scale_change_max_w{}".format(window), "max")
    for quality in ("peak", "apce", "entropy", "margin"):
        source_features["remote_{}_trend_abs_w8".format(quality)] = source_features[
            "remote_{}_trend_w8".format(quality)].abs()
        aggregate_column("remote_{}_trend_abs_w8".format(quality),
                         "remote_{}_trend_abs_max_w8".format(quality), "max")
        aggregate_column("local_remote_{}_trend_gap_w8".format(quality),
                         "local_remote_{}_trend_gap_max_w8".format(quality), "max")
    aggregate_column("remote_message_norm_change_w8",
                     "remote_message_norm_change_max_w8", "max")
    aggregate_column("remote_abrupt_change_w16", "remote_abrupt_change_max_w16", "max")
    aggregate_column("remote_valid_run", "remote_valid_run_min", "min")
    aggregate_column("message_age", "message_age_mean", "mean")
    aggregate_column("message_age_change", "message_age_change_max", "max")
    aggregate_column("prompt_cosine_trend_abs_w8",
                     "prompt_cosine_trend_abs_mean_w8", "mean")
    aggregate_column("local_remote_state_change_gap_w8",
                     "local_remote_state_change_gap_max_w8", "max")
    aggregate_column("residual_norm_trend_abs_w8",
                     "residual_norm_trend_abs_max_w8", "max")
    for window in WINDOWS:
        aggregate_column("gated_residual_history_ratio_w{}".format(window),
                         "gated_residual_history_ratio_w{}".format(window), "mean")
    aggregate_column("remote_state_speed_current", "multi_remote_state_speed_conflict", "range")
    for column in ("remote_quality_00", "remote_quality_01",
                   "remote_quality_02", "remote_quality_03"):
        source_features["{}_range".format(column)] = source_features.groupby(keys)[
            column].transform("max") - source_features.groupby(keys)[column].transform("min")
    source_features["quality_conflict"] = source_features[[
        "remote_quality_00_range", "remote_quality_01_range",
        "remote_quality_02_range", "remote_quality_03_range"]].mean(axis=1)
    aggregate_column("quality_conflict", "multi_remote_quality_conflict", "mean")
    aggregate_column("raw_input_08", "multi_remote_prompt_cosine_conflict", "range")
    aggregate_column("multi_remote_normalized_weight",
                     "multi_remote_gate_weight_conflict", "range")
    return output


def add_offline_labels(features):
    result = features.copy()
    delta = result.iou_delta_offline.astype(float)
    result["label_primary"] = np.where(
        delta > EPSILON, "helpful", np.where(delta < -EPSILON, "harmful", "tied"))
    result["label_eps0"] = np.where(
        delta > 0, "helpful", np.where(delta < 0, "harmful", "tied"))
    event_frames = []
    for (_, _), group in result.groupby(["sequence_id", "receiver_view"], sort=False):
        g = group.sort_values("frame_id").copy()
        harmful = g.label_primary.eq("harmful").to_numpy()
        run = run_length(harmful)
        prior_persistent = pd.Series(run >= 8).shift(1, fill_value=False).to_numpy()
        g["harmful_run_length"] = run
        g["divergence_onset"] = harmful & ~pd.Series(harmful).shift(
            1, fill_value=False).to_numpy()
        g["persistent_harmful_state"] = run >= 8
        g["recovery"] = ~harmful & prior_persistent
        event_frames.append(g)
    return pd.concat(event_frames, ignore_index=True)


def safe_auc(labels, scores, positive):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores) & np.isin(labels, ("helpful", "harmful"))
    y = (labels[mask] == positive).astype(int)
    if len(np.unique(y)) < 2 or np.std(scores[mask]) <= 0:
        return float("nan"), float("nan"), int(mask.sum())
    return (float(roc_auc_score(y, scores[mask])),
            float(average_precision_score(y, scores[mask])), int(mask.sum()))


def safe_corr(values, delta, method):
    values = np.asarray(values, dtype=float)
    delta = np.asarray(delta, dtype=float)
    mask = np.isfinite(values) & np.isfinite(delta)
    if mask.sum() < 3 or np.std(values[mask]) <= 0 or np.std(delta[mask]) <= 0:
        return float("nan")
    function = pearsonr if method == "pearson" else spearmanr
    return float(function(values[mask], delta[mask]).statistic)


def effect_size(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    helpful = values[(labels == "helpful") & np.isfinite(values)]
    harmful = values[(labels == "harmful") & np.isfinite(values)]
    if len(helpful) < 2 or len(harmful) < 2:
        return float("nan")
    pooled = math.sqrt(((len(helpful) - 1) * helpful.var(ddof=1)
                        + (len(harmful) - 1) * harmful.var(ddof=1))
                       / (len(helpful) + len(harmful) - 2))
    return float((helpful.mean() - harmful.mean()) / pooled) if pooled > 0 else 0.0


def orientation(train, signal, label_column):
    auc, _, _ = safe_auc(train[label_column], train[signal], "harmful")
    return 1.0 if not np.isfinite(auc) or auc >= 0.5 else -1.0


def target_metrics(frame, scores, label_column):
    rows = []
    work = frame[["target_id", "receiver_drone", label_column]].copy()
    work["score"] = np.asarray(scores, dtype=float)
    for target, group in work.groupby("target_id"):
        auc, _, count = safe_auc(group[label_column], group.score, "harmful")
        rows.append((target, auc, count))
    return rows


def contribution_concentration(target_rows):
    excess = [max(auc - 0.5, 0.0) for _, auc, _ in target_rows if np.isfinite(auc)]
    if not excess or sum(excess) <= 0:
        return 1.0
    return float(max(excess) / sum(excess))


def scalar_utility(features, manifest, catalog):
    available = catalog[catalog.availability.str.startswith("AVAILABLE")]
    signals = [name for name in available.signal_name if name in features.columns]
    fold_rows, target_rows, receiver_rows = [], [], []
    for outer_fold in range(5):
        split = manifest[manifest.outer_fold == outer_fold]
        train_targets = set(split[split.inner_role == "inner_train"].target_id)
        dev_targets = set(split[split.inner_role == "inner_dev"].target_id)
        train = features[features.target_id.isin(train_targets)]
        dev = features[features.target_id.isin(dev_targets)]
        for signal in signals:
            sign = orientation(train, signal, "label_primary")
            risk = sign * dev[signal].to_numpy(float)
            harmful_auc, harmful_pr, count = safe_auc(
                dev.label_primary, risk, "harmful")
            helpful_auc, helpful_pr, _ = safe_auc(
                dev.label_primary, -risk, "helpful")
            eps0_auc, eps0_pr, eps0_count = safe_auc(
                dev.label_eps0, risk, "harmful")
            per_target = target_metrics(dev, risk, "label_primary")
            finite_target = [auc for _, auc, _ in per_target if np.isfinite(auc)]
            loto = []
            for target in dev_targets:
                keep = dev.target_id != target
                auc, _, _ = safe_auc(dev.loc[keep, "label_primary"], risk[keep], "harmful")
                if np.isfinite(auc):
                    loto.append(auc)
            receiver_aucs = []
            for receiver in "ABC":
                mask = dev.receiver_drone == receiver
                auc, pr, receiver_count = safe_auc(
                    dev.loc[mask, "label_primary"], risk[mask], "harmful")
                receiver_rows.append({
                    "outer_fold": outer_fold, "signal_name": signal,
                    "receiver": receiver, "harmful_roc_auc": auc,
                    "harmful_pr_auc": pr, "count": receiver_count,
                })
                if np.isfinite(auc):
                    receiver_aucs.append(auc)
            for target, auc, target_count in per_target:
                target_rows.append({
                    "outer_fold": outer_fold, "signal_name": signal,
                    "target_id": target, "harmful_roc_auc": auc,
                    "count": target_count,
                })
            fold_rows.append({
                "row_type": "fold", "outer_fold": outer_fold,
                "signal_name": signal, "orientation": sign,
                "harmful_roc_auc": harmful_auc, "harmful_pr_auc": harmful_pr,
                "helpful_roc_auc": helpful_auc, "helpful_pr_auc": helpful_pr,
                "pearson_risk_vs_iou_delta": safe_corr(
                    risk, dev.iou_delta_offline, "pearson"),
                "spearman_risk_vs_iou_delta": safe_corr(
                    risk, dev.iou_delta_offline, "spearman"),
                "helpful_minus_harmful_effect_size": effect_size(
                    risk, dev.label_primary),
                "epsilon0_harmful_roc_auc": eps0_auc,
                "epsilon0_harmful_pr_auc": eps0_pr,
                "binary_frame_count": count,
                "epsilon0_binary_frame_count": eps0_count,
                "target_auc_mean": float(np.nanmean(finite_target)) if finite_target else np.nan,
                "target_auc_std": float(np.nanstd(finite_target)) if finite_target else np.nan,
                "target_auc_min": float(np.nanmin(finite_target)) if finite_target else np.nan,
                "targets_above_half": int(sum(auc > 0.5 for auc in finite_target)),
                "valid_target_count": len(finite_target),
                "target_contribution_concentration": contribution_concentration(per_target),
                "loto_auc_min": float(np.nanmin(loto)) if loto else np.nan,
                "receiver_auc_min": float(np.nanmin(receiver_aucs)) if receiver_aucs else np.nan,
                "receiver_auc_max": float(np.nanmax(receiver_aucs)) if receiver_aucs else np.nan,
            })
    fold_df = pd.DataFrame(fold_rows)
    target_df = pd.DataFrame(target_rows)
    receiver_df = pd.DataFrame(receiver_rows)
    summary_rows = []
    for signal, group in fold_df.groupby("signal_name"):
        target_group = target_df[target_df.signal_name == signal]
        per_target = [("{}:{}".format(row.outer_fold, row.target_id),
                       row.harmful_roc_auc, row["count"])
                      for _, row in target_group.iterrows()]
        receiver_summary = receiver_df[receiver_df.signal_name == signal].groupby(
            "receiver").harmful_roc_auc.mean()
        row = {
            "row_type": "summary", "outer_fold": "all",
            "signal_name": signal, "orientation": np.nan,
            "harmful_roc_auc": group.harmful_roc_auc.mean(),
            "harmful_pr_auc": group.harmful_pr_auc.mean(),
            "helpful_roc_auc": group.helpful_roc_auc.mean(),
            "helpful_pr_auc": group.helpful_pr_auc.mean(),
            "pearson_risk_vs_iou_delta": group.pearson_risk_vs_iou_delta.mean(),
            "spearman_risk_vs_iou_delta": group.spearman_risk_vs_iou_delta.mean(),
            "helpful_minus_harmful_effect_size": group.helpful_minus_harmful_effect_size.mean(),
            "epsilon0_harmful_roc_auc": group.epsilon0_harmful_roc_auc.mean(),
            "epsilon0_harmful_pr_auc": group.epsilon0_harmful_pr_auc.mean(),
            "binary_frame_count": group.binary_frame_count.sum(),
            "epsilon0_binary_frame_count": group.epsilon0_binary_frame_count.sum(),
            "target_auc_mean": target_group.harmful_roc_auc.mean(),
            "target_auc_std": target_group.harmful_roc_auc.std(ddof=0),
            "target_auc_min": target_group.harmful_roc_auc.min(),
            "targets_above_half": int((target_group.harmful_roc_auc > 0.5).sum()),
            "valid_target_count": int(target_group.harmful_roc_auc.notna().sum()),
            "target_contribution_concentration": contribution_concentration(per_target),
            "loto_auc_min": group.loto_auc_min.min(),
            "receiver_auc_min": receiver_summary.min(),
            "receiver_auc_max": receiver_summary.max(),
            "folds_above_half": int((group.harmful_roc_auc > 0.5).sum()),
            "fold_auc_std": group.harmful_roc_auc.std(ddof=0),
            "valid_fold_count": int(group.harmful_roc_auc.notna().sum()),
        }
        summary_rows.append(row)
    utility = pd.concat([fold_df, pd.DataFrame(summary_rows)], ignore_index=True)
    utility = utility.merge(catalog[["signal_name", "group", "kind"]],
                            on="signal_name", how="left", validate="many_to_one")
    utility.to_csv(OUT / "single_signal_utility.csv", index=False)
    utility[utility.kind == "temporal"].to_csv(
        OUT / "temporal_signal_utility.csv", index=False)
    utility[utility.group == "D"].to_csv(
        OUT / "divergence_signal_utility.csv", index=False)
    receiver_summary_rows = []
    for (signal, receiver), group in receiver_df.groupby(["signal_name", "receiver"]):
        receiver_summary_rows.append({
            "signal_name": signal, "receiver": receiver,
            "mean_harmful_roc_auc": group.harmful_roc_auc.mean(),
            "std_harmful_roc_auc": group.harmful_roc_auc.std(ddof=0),
            "folds_above_half": int((group.harmful_roc_auc > 0.5).sum()),
            "valid_folds": int(group.harmful_roc_auc.notna().sum()),
            "frame_count": int(group["count"].sum()),
        })
    pd.DataFrame(receiver_summary_rows).to_csv(
        OUT / "receiver_stability.csv", index=False)
    return utility, target_df, receiver_df


COMBINATIONS = {
    "temporal_core": [
        "local_center_velocity_w3", "local_center_acceleration_w3",
        "local_trajectory_error_w3", "local_conf_drop_duration",
        "remote_state_velocity_max_w3", "remote_abrupt_change_max_w16",
        "local_remote_apce_trend_gap_max_w8",
        "prompt_cosine_trend_abs_mean_w8",
    ],
    "divergence_safety": [
        "state_center_divergence", "state_scale_divergence",
        "state_center_divergence_trend_w3",
        "state_center_divergence_ema_w8", "state_divergence_growth_run",
        "state_quality_not_improved_duration", "state_recovery_debt_w8",
        "state_tracker_prediction_error_w3",
    ],
}
COMBINATIONS["cross_group"] = COMBINATIONS["temporal_core"] + COMBINATIONS[
    "divergence_safety"]


def target_balanced_weights(frame, label_column):
    counts = frame.groupby("target_id").size()
    target_weight = frame.target_id.map(1.0 / counts).to_numpy(float)
    return target_weight / np.mean(target_weight)


def combination_utility(features, manifest):
    rows, all_target_rows = [], []
    for outer_fold in range(5):
        split = manifest[manifest.outer_fold == outer_fold]
        train_targets = set(split[split.inner_role == "inner_train"].target_id)
        dev_targets = set(split[split.inner_role == "inner_dev"].target_id)
        for model_name, signal_names in COMBINATIONS.items():
            train_all = features[features.target_id.isin(train_targets)].copy()
            dev_all = features[features.target_id.isin(dev_targets)].copy()
            train = train_all[train_all.label_primary.isin(("helpful", "harmful"))]
            dev = dev_all[dev_all.label_primary.isin(("helpful", "harmful"))]
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=1000,
                    random_state=SEED, solver="liblinear")),
            ])
            y_train = train.label_primary.eq("harmful").astype(int)
            pipeline.fit(train[signal_names], y_train,
                         model__sample_weight=target_balanced_weights(
                             train, "label_primary"))
            risk = pipeline.predict_proba(dev[signal_names])[:, 1]
            risk_all = pipeline.predict_proba(dev_all[signal_names])[:, 1]
            harmful_auc, harmful_pr, count = safe_auc(
                dev.label_primary, risk, "harmful")
            helpful_auc, helpful_pr, _ = safe_auc(
                dev.label_primary, -risk, "helpful")
            per_target = target_metrics(dev, risk, "label_primary")
            for target, auc, target_count in per_target:
                all_target_rows.append({
                    "outer_fold": outer_fold, "model_name": model_name,
                    "target_id": target, "harmful_roc_auc": auc,
                    "count": target_count,
                })
            receiver_auc = {}
            for receiver in "ABC":
                mask = dev.receiver_drone == receiver
                auc, _, _ = safe_auc(dev.loc[mask, "label_primary"], risk[mask], "harmful")
                if np.isfinite(auc):
                    receiver_auc[receiver] = auc
            eps0_mask = dev_all.label_eps0.isin(("helpful", "harmful"))
            eps0_auc, eps0_pr, eps0_count = safe_auc(
                dev_all.loc[eps0_mask, "label_eps0"], risk_all[eps0_mask], "harmful")
            loto = []
            for target in dev_targets:
                keep = dev.target_id != target
                auc, _, _ = safe_auc(dev.loc[keep, "label_primary"], risk[keep], "harmful")
                if np.isfinite(auc):
                    loto.append(auc)
            coefficients = pipeline.named_steps["model"].coef_[0]
            rows.append({
                "row_type": "fold", "outer_fold": outer_fold,
                "model_name": model_name, "feature_count": len(signal_names),
                "features": "|".join(signal_names),
                "harmful_roc_auc": harmful_auc, "harmful_pr_auc": harmful_pr,
                "helpful_roc_auc": helpful_auc, "helpful_pr_auc": helpful_pr,
                "epsilon0_harmful_roc_auc": eps0_auc,
                "epsilon0_harmful_pr_auc": eps0_pr,
                "binary_frame_count": count,
                "epsilon0_binary_frame_count": eps0_count,
                "fold_above_half": harmful_auc > 0.5,
                "target_contribution_concentration": contribution_concentration(per_target),
                "target_auc_mean": np.nanmean([x[1] for x in per_target]),
                "target_auc_min": np.nanmin([x[1] for x in per_target]),
                "loto_auc_min": np.nanmin(loto),
                "receiver_auc_A": receiver_auc.get("A", np.nan),
                "receiver_auc_B": receiver_auc.get("B", np.nan),
                "receiver_auc_C": receiver_auc.get("C", np.nan),
                "receiver_auc_min": np.nanmin(list(receiver_auc.values())),
                "receiver_auc_max": np.nanmax(list(receiver_auc.values())),
                "standardized_coefficients": json.dumps(
                    dict(zip(signal_names, coefficients)), sort_keys=True),
            })
    fold_df = pd.DataFrame(rows)
    target_df = pd.DataFrame(all_target_rows)
    summaries = []
    for model_name, group in fold_df.groupby("model_name"):
        model_targets = target_df[target_df.model_name == model_name]
        target_tuples = [("{}:{}".format(row.outer_fold, row.target_id),
                          row.harmful_roc_auc, row["count"])
                         for _, row in model_targets.iterrows()]
        receiver_means = {
            receiver: group["receiver_auc_{}".format(receiver)].mean()
            for receiver in "ABC"
        }
        summaries.append({
            "row_type": "summary", "outer_fold": "all",
            "model_name": model_name,
            "feature_count": group.feature_count.iloc[0],
            "features": group.features.iloc[0],
            "harmful_roc_auc": group.harmful_roc_auc.mean(),
            "harmful_pr_auc": group.harmful_pr_auc.mean(),
            "helpful_roc_auc": group.helpful_roc_auc.mean(),
            "helpful_pr_auc": group.helpful_pr_auc.mean(),
            "epsilon0_harmful_roc_auc": group.epsilon0_harmful_roc_auc.mean(),
            "epsilon0_harmful_pr_auc": group.epsilon0_harmful_pr_auc.mean(),
            "binary_frame_count": group.binary_frame_count.sum(),
            "epsilon0_binary_frame_count": group.epsilon0_binary_frame_count.sum(),
            "fold_above_half": int((group.harmful_roc_auc > 0.5).sum()),
            "target_contribution_concentration": contribution_concentration(target_tuples),
            "target_auc_mean": model_targets.harmful_roc_auc.mean(),
            "target_auc_min": model_targets.harmful_roc_auc.min(),
            "loto_auc_min": group.loto_auc_min.min(),
            "receiver_auc_A": receiver_means["A"],
            "receiver_auc_B": receiver_means["B"],
            "receiver_auc_C": receiver_means["C"],
            "receiver_auc_min": min(receiver_means.values()),
            "receiver_auc_max": max(receiver_means.values()),
            "standardized_coefficients": "per-fold rows",
            "fold_auc_std": group.harmful_roc_auc.std(ddof=0),
            "valid_fold_count": int(group.harmful_roc_auc.notna().sum()),
        })
    result = pd.concat([fold_df, pd.DataFrame(summaries)], ignore_index=True)
    result.to_csv(OUT / "combination_model_results.csv", index=False)
    return result


def signal_passes(row, threshold):
    receiver_opposite = row.receiver_auc_min < 0.48 and row.receiver_auc_max > 0.55
    return bool(
        row.harmful_roc_auc >= threshold
        and row.valid_fold_count == 5
        and row.folds_above_half >= 4
        and row.target_contribution_concentration <= 0.25
        and row.harmful_roc_auc > GATE_REFERENCE_AUC
        and not receiver_opposite)


def select_decision(utility, combinations, catalog, receiver_stability):
    summary = utility[utility.row_type == "summary"].copy()
    summary["passes"] = summary.apply(lambda row: signal_passes(row, 0.55), axis=1)
    combo_summary = combinations[combinations.row_type == "summary"].copy()
    combo_summary["passes"] = combo_summary.apply(
        lambda row: signal_passes(row.rename({"fold_above_half": "folds_above_half"}), 0.58),
        axis=1)
    temporal_pass = summary[(summary.passes) & (summary.group != "D")]
    divergence_pass = summary[(summary.passes) & (summary.group == "D")]
    broad_combo_pass = combo_summary[
        (combo_summary.passes) & combo_summary.model_name.isin(("temporal_core", "cross_group"))]
    divergence_combo_pass = combo_summary[
        (combo_summary.passes) & combo_summary.model_name.eq("divergence_safety")]
    reversal_counts = {}
    for signal, group in receiver_stability.groupby("signal_name"):
        by_fold = group.pivot(
            index="outer_fold", columns="receiver", values="harmful_roc_auc")
        reversal_counts[signal] = int(((by_fold.max(axis=1) >= 0.55)
                                       & (by_fold.min(axis=1) <= 0.45)).sum())
    receiver_specific = [name for name, count in reversal_counts.items() if count >= 4]
    if len(temporal_pass) or len(broad_combo_pass):
        decision = "S1"
    elif len(divergence_pass) or len(divergence_combo_pass):
        decision = "S2"
    elif receiver_specific:
        decision = "S3"
    else:
        decision = "S4"

    best_single = summary.sort_values("harmful_roc_auc", ascending=False).iloc[0]
    best_temporal = summary[summary.kind == "temporal"].sort_values(
        "harmful_roc_auc", ascending=False).iloc[0]
    best_divergence = summary[summary.group == "D"].sort_values(
        "harmful_roc_auc", ascending=False).iloc[0]
    best_combo = combo_summary.sort_values("harmful_roc_auc", ascending=False).iloc[0]
    allowed = sorted(set(temporal_pass.signal_name) | set(divergence_pass.signal_name))
    if len(broad_combo_pass):
        for features in broad_combo_pass.features:
            allowed.extend(features.split("|"))
    allowed = sorted(set(allowed))
    lines = [
        "# Reliability v2 signal-selection decision", "",
        "Decision: **{}**.".format(decision), "",
        "All selection metrics below are nested inner-dev utility, not tracker performance.", "",
        "- Best scalar: `{}` harmful ROC-AUC {:.6f}.".format(
            best_single.signal_name, best_single.harmful_roc_auc),
        "- Best temporal scalar: `{}` harmful ROC-AUC {:.6f}.".format(
            best_temporal.signal_name, best_temporal.harmful_roc_auc),
        "- Best divergence scalar: `{}` harmful ROC-AUC {:.6f}.".format(
            best_divergence.signal_name, best_divergence.harmful_roc_auc),
        "- Best fixed combination: `{}` harmful ROC-AUC {:.6f}.".format(
            best_combo.model_name, best_combo.harmful_roc_auc),
        "- Qualifying scalar signals: {}.".format(
            ", ".join(summary[summary.passes].signal_name) or "none"),
        "- Qualifying combinations: {}.".format(
            ", ".join(combo_summary[combo_summary.passes].model_name) or "none"),
        "- Receiver-specific reversal signals meeting S3 rule: {}.".format(
            ", ".join(receiver_specific) or "none"),
        "- Inputs eligible for a later Gate v2 proposal: {}.".format(
            ", ".join(allowed) if allowed else "none"), "",
    ]
    if decision == "S1":
        lines.append("Recommendation: the frozen criteria support a later, separately authorized Gate v2 implementation using only the qualifying prediction-only inputs.")
    elif decision == "S2":
        lines.append("Recommendation: do not build a general reliability scorer; a later design may consider only divergence-triggered safety suppression/rollback.")
    else:
        lines.append("Recommendation: do not implement Gate v2 from this candidate set; stop this Gate-input route and reconsider output-level safety or message design.")
    (OUT / "signal_selection_decision.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return {
        "decision": decision,
        "best_single": best_single.to_dict(),
        "best_temporal": best_temporal.to_dict(),
        "best_divergence": best_divergence.to_dict(),
        "best_combo": best_combo.to_dict(),
        "allowed_inputs": allowed,
        "qualifying_signals": list(summary[summary.passes].signal_name),
        "qualifying_combinations": list(combo_summary[combo_summary.passes].model_name),
        "receiver_specific_signals": receiver_specific,
    }


def write_nested_fold_results(utility, combinations, manifest, features):
    rows = []
    for outer_fold in range(5):
        fold = utility[(utility.row_type == "fold") & (utility.outer_fold == outer_fold)]
        best = fold.sort_values("harmful_roc_auc", ascending=False).iloc[0]
        combo = combinations[(combinations.row_type == "fold")
                             & (combinations.outer_fold == outer_fold)].sort_values(
                                 "harmful_roc_auc", ascending=False).iloc[0]
        split = manifest[manifest.outer_fold == outer_fold]
        dev_targets = set(split[split.inner_role == "inner_dev"].target_id)
        dev_frames = features[features.target_id.isin(dev_targets)]
        rows.append({
            "outer_fold": outer_fold,
            "inner_train_targets": "|".join(split[split.inner_role == "inner_train"].target_id),
            "inner_dev_targets": "|".join(split[split.inner_role == "inner_dev"].target_id),
            "best_single_signal": best.signal_name,
            "best_single_harmful_roc_auc": best.harmful_roc_auc,
            "best_combination": combo.model_name,
            "best_combination_harmful_roc_auc": combo.harmful_roc_auc,
            "helpful_frame_count": int(dev_frames.label_primary.eq("helpful").sum()),
            "harmful_frame_count": int(dev_frames.label_primary.eq("harmful").sum()),
            "tied_frame_count": int(dev_frames.label_primary.eq("tied").sum()),
            "divergence_onset_count": int(dev_frames.divergence_onset.sum()),
            "persistent_harmful_frame_count": int(
                dev_frames.persistent_harmful_state.sum()),
            "recovery_count": int(dev_frames.recovery.sum()),
        })
    pd.DataFrame(rows).to_csv(OUT / "nested_fold_results.csv", index=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(decision, catalog, manifest):
    files = [
        "nested_split_protocol.md", "nested_target_manifest.csv",
        "data_usage_boundary.md", "candidate_signal_catalog.csv",
        "candidate_signal_spec.md", "inference_availability_audit.md",
        "label_definition.md", "acceptance_criteria.md",
        "single_signal_utility.csv", "temporal_signal_utility.csv",
        "divergence_signal_utility.csv", "combination_model_results.csv",
        "receiver_stability.csv", "nested_fold_results.csv",
        "signal_selection_decision.md",
    ]
    lines = [
        "# Reliability v2 design manifest", "", "Status: **COMPLETE**.", "",
        "- Split seed: {}.".format(SEED),
        "- Windows: 3/8/16 frames; primary epsilon: 0.01; sensitivity epsilon: 0.",
        "- Candidate catalog: {} designed, {} evaluated, {} unavailable diagnostics.".format(
            len(catalog), sum(catalog.availability.str.startswith("AVAILABLE")),
            sum(catalog.availability == "MISSING_DIAGNOSTIC")),
        "- Nested memberships: {}; unique targets: {}; all views target-bound.".format(
            len(manifest), manifest.target_id.nunique()),
        "- Outer-holdout leakage checks: PASS for all five folds.",
        "- Decision: **{}**.".format(decision["decision"]), "",
        "| File | SHA256 |", "|---|---|",
    ]
    for name in files:
        lines.append("| `{}` | `{}` |".format(name, sha256(OUT / name)))
    lines.extend(["", "No tracker inference, training, Gate implementation, validation/test, C0, checkpoint mutation, hyperparameter sweep, or Git mutation was performed."])
    (OUT / "reliability_v2_design_manifest.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def load_prediction_only_tables():
    aggregate = pd.read_csv(SOURCE / "frame_aggregate_diagnostics.csv.gz")
    source = pd.read_csv(SOURCE / "frame_source_diagnostics.csv.gz")
    aggregate = aggregate[aggregate.frame_id > 0].copy()
    required_aggregate = [
        "fold_id", "target_id", "sequence_id", "receiver_view", "frame_id",
        "local_bbox_x", "local_bbox_y", "local_bbox_w", "local_bbox_h",
        "c1_bbox_x", "c1_bbox_y", "c1_bbox_w", "c1_bbox_h",
        "local_confidence", "c1_confidence", "local_apce", "c1_apce",
        "local_quality_00", "local_quality_01", "local_quality_02", "local_quality_03",
        "iou_delta_offline",
    ]
    required_source = [
        "target_id", "sequence_id", "receiver_view", "sender_view", "frame_id",
        "remote_bbox_x", "remote_bbox_y", "remote_bbox_w", "remote_bbox_h",
        "remote_quality_00", "remote_quality_01", "remote_quality_02", "remote_quality_03",
        "remote_message_l2", "adapted_residual_l2", "gate_times_residual_l2",
        "local_feature_l2", "message_age_intervals", "valid", "raw_input_08",
        "multi_remote_normalized_weight",
    ]
    missing = set(required_aggregate) - set(aggregate.columns)
    missing |= set(required_source) - set(source.columns)
    if missing:
        raise RuntimeError("missing instrumentation columns: {}".format(sorted(missing)))
    aggregate["receiver_drone"] = aggregate.receiver_view.map(DRONES)
    return aggregate, source


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_nested_manifest()
    catalog = write_catalog()
    if args.prepare_only:
        print(json.dumps({
            "status": "PROTOCOL_FROZEN", "seed": SEED,
            "candidate_count": len(catalog),
            "available_count": int(catalog.availability.str.startswith("AVAILABLE").sum()),
            "nested_memberships": len(manifest),
        }, indent=2))
        return
    aggregate, source = load_prediction_only_tables()
    aggregate_features = compute_aggregate_features(aggregate)
    source_features = compute_source_features(source, aggregate_features)
    feature_columns = [row["signal_name"] for row in candidate_catalog()
                       if row["availability"].startswith("AVAILABLE")]
    features = aggregate_features.merge(
        source_features, on=["sequence_id", "receiver_view", "frame_id"],
        how="left", validate="one_to_one")
    missing_features = set(feature_columns) - set(features.columns)
    if missing_features:
        raise RuntimeError("computed feature missing: {}".format(sorted(missing_features)))
    features = add_offline_labels(features)
    utility, _, receiver_stability = scalar_utility(features, manifest, catalog)
    combinations = combination_utility(features, manifest)
    write_nested_fold_results(utility, combinations, manifest, features)
    decision = select_decision(
        utility, combinations, catalog, receiver_stability)
    write_manifest(decision, catalog, manifest)
    print(json.dumps({
        "status": "COMPLETE", "targets": int(features.target_id.nunique()),
        "views": int(features.sequence_id.nunique()),
        "frames": len(features), "candidate_count": len(catalog),
        "evaluated_count": len(feature_columns), "decision": decision,
    }, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
