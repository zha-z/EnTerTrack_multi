"""Prediction-only E0/C1/T1 inner-dev closed-loop framework and GT join."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence


FORBIDDEN_EXECUTION_FIELDS = frozenset((
    "gt_bbox", "ground_truth", "visibility", "target_visible", "iou",
    "oracle_mask", "failure", "label", "test_iou",
))
POLICIES = ("E0", "C1", "T1")


def assert_prediction_only(payload: Mapping[str, object]) -> None:
    present = FORBIDDEN_EXECUTION_FIELDS.intersection(
        str(key).lower() for key in payload)
    if present:
        raise RuntimeError("closed-loop execution observed GT fields: {}".format(
            sorted(present)))


def digest_rows(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        assert_prediction_only(row)
        digest.update((json.dumps(
            dict(row), sort_keys=True, separators=(",", ":"),
            allow_nan=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def run_prediction_only(
        targets: Sequence[str],
        policy_runner: Callable[[str, str], Iterable[Mapping[str, object]]],
        outer_holdout_targets: Iterable[str] = ()) -> Dict[str, object]:
    """Run independent policy callbacks; callbacks receive no GT payload."""
    forbidden = set(str(value) for value in outer_holdout_targets)
    if set(targets) & forbidden:
        raise RuntimeError("outer holdout target requested by closed-loop runner")
    outputs = {}
    for policy in POLICIES:
        policy_rows = []
        for target in targets:
            rows = list(policy_runner(policy, str(target)))
            for row in rows:
                assert_prediction_only(row)
                if str(row.get("target_id")) != str(target):
                    raise RuntimeError("closed-loop runner mixed targets")
                if str(row.get("policy")) != policy:
                    raise RuntimeError("closed-loop runner mixed policy state")
            policy_rows.extend(rows)
        outputs[policy] = {
            "rows": policy_rows,
            "prediction_sha256": digest_rows(policy_rows),
        }
    return outputs


def evaluate_joined_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Aggregate already post-joined metrics; never called during execution."""
    by_policy_target = defaultdict(lambda: defaultdict(list))
    accounting = defaultdict(dict)
    t1_gates = []
    for row in rows:
        policy = str(row["policy"])
        if policy not in POLICIES:
            raise ValueError("unknown closed-loop policy")
        target = str(row["target_id"])
        metrics = tuple(float(row[name]) for name in (
            "success_auc", "precision", "normalized_precision"))
        if not all(math.isfinite(value) for value in metrics):
            raise RuntimeError("non-finite closed-loop metric")
        by_policy_target[policy][target].append(metrics)
        view_key = (target, int(row.get("receiver_id", row.get("view_id", 0))))
        accounting[policy][view_key] = (
            int(row.get("communication_bytes", 0)),
            int(row.get("accepted_packets", 0)),
            int(row.get("packet_bytes", 0)),
            int(row.get("sender_count", 0)),
        )
        if policy == "T1":
            gates = [float(value) for value in row.get("gates", ())]
            if not all(math.isfinite(value) and 0.0 <= value <= 0.25
                       for value in gates):
                raise RuntimeError("T1 gate outside frozen range")
            t1_gates.extend(gates)
    if accounting["C1"] != accounting["T1"]:
        raise RuntimeError("T1 packet/accounting differs from C1")
    target_metrics = {}
    for policy in POLICIES:
        target_metrics[policy] = {
            target: tuple(sum(values[index] for values in views) / len(views)
                          for index in range(3))
            for target, views in by_policy_target[policy].items()
        }
    targets = sorted(set(target_metrics["C1"]) & set(target_metrics["T1"])
                     & set(target_metrics["E0"]))
    if not targets:
        raise RuntimeError("no matched closed-loop targets")
    macro = {
        policy: tuple(sum(target_metrics[policy][target][index]
                          for target in targets) / len(targets)
                      for index in range(3))
        for policy in POLICIES
    }
    deltas_by_target = {
        target: target_metrics["T1"][target][0]
        - target_metrics["C1"][target][0]
        for target in targets
    }
    positive = sum(value > 0 for value in deltas_by_target.values())
    negative = sum(value < 0 for value in deltas_by_target.values())
    tied = sum(value == 0 for value in deltas_by_target.values())
    ordered_gates = sorted(t1_gates)

    def percentile(fraction):
        if not ordered_gates:
            return 0.0
        position = fraction * (len(ordered_gates) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered_gates) - 1)
        weight = position - lower
        return ordered_gates[lower] * (1.0 - weight) + ordered_gates[upper] * weight

    gate_mean = sum(t1_gates) / len(t1_gates) if t1_gates else 0.0
    gate_std = math.sqrt(sum((value - gate_mean) ** 2 for value in t1_gates)
                         / len(t1_gates)) if t1_gates else 0.0
    result = {
        "target_count": len(targets),
        "per_target": {
            policy: {target: list(values) for target, values in mapping.items()}
            for policy, mapping in target_metrics.items()
        },
        "target_macro": {policy: list(values) for policy, values in macro.items()},
        "t1_minus_c1": [macro["T1"][i] - macro["C1"][i] for i in range(3)],
        "t1_minus_e0": [macro["T1"][i] - macro["E0"][i] for i in range(3)],
        "t1_minus_c1_auc_target_counts": {
            "positive": positive, "negative": negative, "tied": tied},
        "worst_target_t1_minus_c1_auc": min(deltas_by_target.values()),
        "packet_accounting_identical": True,
        "t1_gate_mean": gate_mean,
        "t1_gate_std": gate_std,
        "t1_gate_percentiles": {
            "p05": percentile(0.05), "p25": percentile(0.25),
            "p50": percentile(0.50), "p75": percentile(0.75),
            "p95": percentile(0.95)},
        "t1_gate_fraction_lt_0.025": (
            sum(value < 0.025 for value in t1_gates) / len(t1_gates)
            if t1_gates else 0.0),
        "t1_gate_fraction_gt_0.225": (
            sum(value > 0.225 for value in t1_gates) / len(t1_gates)
            if t1_gates else 0.0),
        "finite_diagnostics": True,
    }
    result["gate2_closed_loop_pass"] = bool(
        result["t1_minus_c1"][0] >= 0.005
        and result["t1_minus_c1"][1] >= 0.0
        and result["t1_minus_c1"][2] >= 0.0
        and all(value >= 0.0 for value in result["t1_minus_e0"])
        and positive + tied >= 3
        and result["worst_target_t1_minus_c1_auc"] >= -0.02
    )
    return result


def distribution_shift_audit(behavior_rows, t1_rows):
    """Non-decisive per-dimension quantile/range comparison."""
    dimensions = []
    for index in range(10):
        baseline = sorted(float(row["normalized_features"][index])
                          for row in behavior_rows)
        observed = sorted(float(row["normalized_features"][index])
                          for row in t1_rows)
        if not baseline or not observed:
            raise RuntimeError("distribution-shift audit received empty rows")
        if not all(math.isfinite(value) for value in baseline + observed):
            raise RuntimeError("distribution-shift audit contains non-finite values")

        def nearest(values, fraction):
            return values[int(round(fraction * (len(values) - 1)))]

        dimensions.append({
            "dimension": index,
            "behavior_range": [baseline[0], baseline[-1]],
            "t1_range": [observed[0], observed[-1]],
            "t1_range_violation_fraction": sum(
                value < baseline[0] or value > baseline[-1]
                for value in observed) / len(observed),
            "behavior_quantiles": [nearest(baseline, q) for q in (0.05, 0.5, 0.95)],
            "t1_quantiles": [nearest(observed, q) for q in (0.05, 0.5, 0.95)],
        })
    return {"dimensions": dimensions, "decisive": False,
            "finite_diagnostics": True}
