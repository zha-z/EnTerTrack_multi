#!/usr/bin/env python3
"""Target-level bootstrap analysis for existing per-sequence tracking metrics.

The input is a long CSV with columns ``target_id``, ``view_id``, ``method``,
``auc``, ``precision`` and ``norm_precision``.  All views of a sampled target
are kept together; frames and views are never treated as independent units.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from dataset_development_utils import mean, parse_sequence_name, sample_std, write_csv
except ImportError:  # pragma: no cover
    from tracking.dataset_development_utils import mean, parse_sequence_name, sample_std, write_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("auc", "precision", "norm_precision")


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_metric_rows(rows: Sequence[Mapping], expected_views: int = 3) -> None:
    if not rows:
        raise ValueError("Metric input has no rows.")
    required = {"target_id", "view_id", "method"}.union(METRICS)
    missing = required - set(rows[0])
    if missing:
        raise ValueError("Metric input is missing columns: {}".format(sorted(missing)))
    grouped: Dict[Tuple[str, str], set] = {}
    for row in rows:
        key = (str(row["method"]), str(row["target_id"]))
        view_id = str(row["view_id"])
        grouped.setdefault(key, set())
        if view_id in grouped[key]:
            raise ValueError("Duplicate method/target/view row: {} / {}".format(key, view_id))
        grouped[key].add(view_id)
        for metric in METRICS:
            try:
                float(row[metric])
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid {} value in row {}.".format(metric, row)) from error
    if expected_views:
        offenders = {
            "{}:{}".format(method, target): sorted(views)
            for (method, target), views in grouped.items()
            if len(views) != expected_views
        }
        if offenders:
            raise ValueError(
                "Expected {} grouped views per target; found {}.".format(
                    expected_views, offenders
                )
            )


def target_deltas(
    rows: Sequence[Mapping], baseline: str, candidate: str, metric: str
) -> Dict[str, float]:
    by_method_target: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["target_id"]))
        by_method_target.setdefault(key, {})[str(row["view_id"])] = float(row[metric])
    baseline_targets = {
        target for method, target in by_method_target if method == baseline
    }
    candidate_targets = {
        target for method, target in by_method_target if method == candidate
    }
    if baseline_targets != candidate_targets:
        raise ValueError(
            "Target mismatch for {} vs {}: baseline_only={}, candidate_only={}.".format(
                baseline,
                candidate,
                sorted(baseline_targets - candidate_targets),
                sorted(candidate_targets - baseline_targets),
            )
        )
    result: Dict[str, float] = {}
    for target in sorted(baseline_targets):
        baseline_views = by_method_target[(baseline, target)]
        candidate_views = by_method_target[(candidate, target)]
        if set(baseline_views) != set(candidate_views):
            raise ValueError(
                "View mismatch for target {} in {} vs {}.".format(
                    target, baseline, candidate
                )
            )
        result[target] = mean(
            [candidate_views[view] - baseline_views[view] for view in sorted(baseline_views)]
        )
    if not result:
        raise ValueError("No shared targets for {} vs {}.".format(baseline, candidate))
    return result


def bootstrap_target_differences(
    rows: Sequence[Mapping],
    baseline: str,
    candidate: str,
    iterations: int = 10000,
    seed: int = 0,
    expected_views: int = 3,
) -> Tuple[List[dict], List[dict]]:
    """Bootstrap whole target groups and return summary and replicate rows."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    validate_metric_rows(rows, expected_views=expected_views)
    rng = random.Random(seed)
    summary_rows: List[dict] = []
    replicate_rows: List[dict] = []
    for metric in METRICS:
        deltas = target_deltas(rows, baseline, candidate, metric)
        target_ids = sorted(deltas)
        samples = []
        for iteration in range(iterations):
            sampled_targets = [rng.choice(target_ids) for _ in target_ids]
            value = mean([deltas[target] for target in sampled_targets])
            samples.append(value)
            replicate_rows.append(
                {
                    "comparison": "{} vs {}".format(candidate, baseline),
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "iteration": iteration,
                    "delta": value,
                    "resampling_unit": "target_group",
                }
            )
        target_values = list(deltas.values())
        summary_rows.append(
            {
                "comparison": "{} vs {}".format(candidate, baseline),
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "target_count": len(target_ids),
                "mean_delta": mean(target_values),
                "target_delta_std": sample_std(target_values),
                "bootstrap_ci_low_95": percentile(samples, 0.025),
                "bootstrap_ci_high_95": percentile(samples, 0.975),
                "positive_target_ratio": sum(value > 0 for value in target_values) / len(target_values),
                "probability_delta_gt_zero": sum(value > 0 for value in samples) / len(samples),
                "resampling_unit": "target_group",
            }
        )
    return summary_rows, replicate_rows


def read_long_metrics(path: Path) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Per-sequence metric CSV not found: {}".format(path))
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv(path: Path) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Per-sequence metric CSV not found: {}".format(path))
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _long_row(sequence: str, method: str, auc: object, precision: object, norm_precision: object) -> dict:
    target_id, view_id = parse_sequence_name(sequence)
    return {
        "target_id": target_id,
        "view_id": view_id,
        "sequence_name": sequence,
        "method": method,
        "auc": auc,
        "precision": precision,
        "norm_precision": norm_precision,
    }


def load_mcr_analysis_metrics(active_csv: Path, safegeom_csv: Path) -> List[dict]:
    """Adapt the existing runid-11/runid-12 analysis CSVs to long format."""
    active = _read_csv(active_csv)
    safegeom = _read_csv(safegeom_csv)
    active_by_sequence = {row["sequence"]: row for row in active}
    safegeom_by_sequence = {row["sequence"]: row for row in safegeom}
    if set(active_by_sequence) != set(safegeom_by_sequence):
        raise ValueError("Active and safegeom sequence sets do not match.")
    rows: List[dict] = []
    for sequence in sorted(active_by_sequence):
        active_row = active_by_sequence[sequence]
        safe_row = safegeom_by_sequence[sequence]
        rows.extend(
            [
                _long_row(
                    sequence,
                    "A0",
                    active_row["baseline_auc"],
                    active_row["baseline_precision"],
                    active_row["baseline_norm_precision"],
                ),
                _long_row(
                    sequence,
                    "MCR-v0",
                    active_row["mcr_auc"],
                    active_row["mcr_precision"],
                    active_row["mcr_norm_precision"],
                ),
                _long_row(
                    sequence,
                    "Safegeom",
                    safe_row["run12_auc"],
                    safe_row["run12_precision"],
                    safe_row["run12_norm_precision"],
                ),
            ]
        )
    return rows


def load_baseline_candidate_metrics(path: Path, candidate_name: str) -> List[dict]:
    """Load a generic baseline/candidate per-sequence CSV, suitable for D1."""
    rows = _read_csv(path)
    required = {
        "sequence",
        "baseline_auc",
        "candidate_auc",
        "baseline_precision",
        "candidate_precision",
        "baseline_norm_precision",
        "candidate_norm_precision",
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError("D1 metric CSV is missing columns: {}".format(sorted(missing)))
    result: List[dict] = []
    for row in rows:
        result.append(
            _long_row(
                row["sequence"],
                "A0",
                row["baseline_auc"],
                row["baseline_precision"],
                row["baseline_norm_precision"],
            )
        )
        result.append(
            _long_row(
                row["sequence"],
                candidate_name,
                row["candidate_auc"],
                row["candidate_precision"],
                row["candidate_norm_precision"],
            )
        )
    return result


def parse_comparison(value: str) -> Tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            "Comparison must be BASELINE:CANDIDATE, got {!r}.".format(value)
        )
    return parts[0], parts[1]


def render_report(rows: Sequence[dict], iterations: int, seed: int) -> str:
    lines = [
        "# Target-level uncertainty report",
        "",
        "Result label: **validation uncertainty analysis**. Resampling unit is the target; all views from a target remain grouped. GT or frame-level samples are not treated as independent observations.",
        "",
        "Bootstrap iterations: **{}**; seed: **{}**.".format(iterations, seed),
        "",
        "| Comparison | Metric | Targets | Mean delta | Target SD | 95% bootstrap CI | Positive-target ratio | P(delta > 0) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {comparison} | {metric} | {target_count} | {mean_delta:+.4f} | {target_delta_std:.4f} | [{bootstrap_ci_low_95:+.4f}, {bootstrap_ci_high_95:+.4f}] | {positive_target_ratio:.3f} | {probability_delta_gt_zero:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "With only five independent validation targets, confidence intervals should be interpreted as sensitivity diagnostics rather than proof of generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--mcr-active-csv", type=Path)
    parser.add_argument("--safegeom-csv", type=Path)
    parser.add_argument("--d1-csv", type=Path)
    parser.add_argument("--comparison", action="append", type=parse_comparison, default=[])
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--expected-views", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "dataset_development_audit")
    args = parser.parse_args()

    metric_rows: List[dict] = []
    comparisons = list(args.comparison)
    if args.metrics_csv:
        metric_rows.extend(read_long_metrics(args.metrics_csv))
    if bool(args.mcr_active_csv) != bool(args.safegeom_csv):
        parser.error("--mcr-active-csv and --safegeom-csv must be supplied together")
    if args.mcr_active_csv:
        metric_rows.extend(load_mcr_analysis_metrics(args.mcr_active_csv, args.safegeom_csv))
        comparisons.extend([("A0", "MCR-v0"), ("A0", "Safegeom")])
    if args.d1_csv:
        metric_rows.extend(load_baseline_candidate_metrics(args.d1_csv, "D1"))
        comparisons.append(("A0", "D1"))
    if not metric_rows:
        parser.error("provide --metrics-csv, the MCR CSV pair, or --d1-csv")
    if not comparisons:
        parser.error("provide at least one --comparison")

    # Multiple adapters may repeat identical A0 rows. Keep one exact row per
    # method/target/view and reject conflicting values.
    deduplicated: Dict[Tuple[str, str, str], dict] = {}
    for row in metric_rows:
        key = (str(row["method"]), str(row["target_id"]), str(row["view_id"]))
        if key in deduplicated:
            for metric in METRICS:
                if float(deduplicated[key][metric]) != float(row[metric]):
                    raise ValueError("Conflicting duplicate metric row for {}.".format(key))
        else:
            deduplicated[key] = row
    metric_rows = list(deduplicated.values())
    all_summary: List[dict] = []
    all_replicates: List[dict] = []
    for baseline, candidate in dict.fromkeys(comparisons):
        summary, replicates = bootstrap_target_differences(
            metric_rows,
            baseline,
            candidate,
            iterations=args.iterations,
            seed=args.seed,
            expected_views=args.expected_views,
        )
        all_summary.extend(summary)
        all_replicates.extend(replicates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "target_level_bootstrap.csv",
        all_replicates,
        ["comparison", "baseline", "candidate", "metric", "iteration", "delta", "resampling_unit"],
    )
    (args.output_dir / "target_level_uncertainty_report.md").write_text(
        render_report(all_summary, args.iterations, args.seed), encoding="utf-8"
    )
    print("Wrote target-level uncertainty analysis to {}".format(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
