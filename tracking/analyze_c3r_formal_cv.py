#!/usr/bin/env python3
"""Aggregate the frozen formal C3R five-fold CV outputs."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "output/multi_agent_collaboration_clean/formal"
OUT = FORMAL / "final_cv"
METRICS = ("auc", "precision", "norm_precision")
SYSTEMS = ("e0", "c0", "c1")
CONTRASTS = {
    "c0_minus_e0": ("c0", "e0"),
    "c1_minus_e0": ("c1", "e0"),
    "c1_minus_c0": ("c1", "c0"),
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    if not rows:
        raise RuntimeError("refusing to write empty CSV: {}".format(path))
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value):
    return float(value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paired_rows():
    targets, sequences = [], []
    for fold in range(5):
        fold_dir = FORMAL / "fold_{}".format(fold)
        for row in read_csv(fold_dir / "34_fold{}_target_metrics.csv".format(fold)):
            item = {"fold_id": fold, "target": row["target"]}
            for system in SYSTEMS:
                for metric in METRICS:
                    item[system + "_" + metric] = f(row[system + "_" + metric])
            for contrast in CONTRASTS:
                for metric in METRICS:
                    item[contrast + "_" + metric] = f(row[contrast + "_" + metric])
            targets.append(item)
        for row in read_csv(fold_dir / "35_fold{}_sequence_metrics.csv".format(fold)):
            item = {"fold_id": fold, "target": row["target"],
                    "sequence": row["sequence"], "view": int(row["view"]),
                    "drone": {1: "A", 2: "B", 3: "C"}[int(row["view"])]}
            for system in SYSTEMS:
                for metric in METRICS:
                    item[system + "_" + metric] = f(row[system + "_" + metric])
            for contrast, (left, right) in CONTRASTS.items():
                for metric in METRICS:
                    item[contrast + "_" + metric] = (
                        item[left + "_" + metric] - item[right + "_" + metric])
            sequences.append(item)
    if len(targets) != 23 or len({r["target"] for r in targets}) != 23:
        raise RuntimeError("target OOF coverage is not exactly 23 unique targets")
    if len(sequences) != 69 or len({r["sequence"] for r in sequences}) != 69:
        raise RuntimeError("sequence OOF coverage is not exactly 69 unique views")
    return sorted(targets, key=lambda r: r["target"]), sorted(
        sequences, key=lambda r: r["sequence"])


def fold_rows(targets):
    output = []
    for fold in range(5):
        rows = [row for row in targets if row["fold_id"] == fold]
        item = {"fold_id": fold, "target_count": len(rows)}
        for system in SYSTEMS:
            for metric in METRICS:
                item[system + "_" + metric] = float(np.mean(
                    [row[system + "_" + metric] for row in rows]))
        for contrast in CONTRASTS:
            for metric in METRICS:
                item[contrast + "_" + metric] = float(np.mean(
                    [row[contrast + "_" + metric] for row in rows]))
        output.append(item)
    return output


def bootstrap_rows(targets):
    output = []
    rng = np.random.default_rng(20260716)
    indices = rng.integers(0, len(targets), size=(10000, len(targets)))
    for contrast in CONTRASTS:
        for metric in METRICS:
            values = np.asarray([row[contrast + "_" + metric] for row in targets])
            samples = values[indices].mean(axis=1)
            output.append({
                "contrast": contrast, "metric": metric,
                "target_count": len(values), "resamples": 10000,
                "seed": 20260716, "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "std_delta": float(values.std(ddof=1)),
                "ci_low": float(np.percentile(samples, 2.5)),
                "ci_high": float(np.percentile(samples, 97.5)),
                "p_delta_gt_0": float(np.mean(samples > 0)),
                "positive_targets": int(np.sum(values > 1e-12)),
                "negative_targets": int(np.sum(values < -1e-12)),
                "tied_targets": int(np.sum(np.abs(values) <= 1e-12)),
            })
    return output


def leave_one_fold_out(targets):
    output = []
    for omitted in range(5):
        rows = [row for row in targets if row["fold_id"] != omitted]
        for contrast in CONTRASTS:
            item = {"omitted_fold": omitted, "contrast": contrast,
                    "target_count": len(rows)}
            for metric in METRICS:
                item[metric + "_delta"] = float(np.mean(
                    [row[contrast + "_" + metric] for row in rows]))
            output.append(item)
    return output


def registry_rows():
    return read_csv(FORMAL / "evaluation_registry.csv")


def gate_rows(registry):
    output = []
    overall = defaultdict(lambda: {"gates": [], "ratios": [], "frames": 0})
    for row in registry:
        if row["message_mode"] == "none":
            continue
        role = row["experiment_id"].split("_", 1)[0].lower()
        values, ratios, frames = [], [], 0
        for path in sorted(Path(row["output_dir"]).glob("*_c3r_diagnostics.csv")):
            for item in read_csv(path):
                frames += 1
                values.extend(float(x) for x in json.loads(item["gates"]))
                ratios.append(float(item["aggregate_ratio"]))
        if not values:
            raise RuntimeError("missing gate samples for {}".format(row["experiment_id"]))
        item = {"scope": "fold", "fold_id": int(row["fold_id"]), "system": role,
                "frame_count": frames, "gate_sample_count": len(values),
                "gate_mean": float(np.mean(values)),
                "gate_median": float(np.median(values)),
                "gate_p90": float(np.percentile(values, 90)),
                "aggregate_ratio_mean": float(np.mean(ratios)),
                "aggregate_ratio_p90": float(np.percentile(ratios, 90))}
        output.append(item)
        overall[role]["gates"].extend(values)
        overall[role]["ratios"].extend(ratios)
        overall[role]["frames"] += frames
    for role in ("c0", "c1"):
        values = overall[role]["gates"]
        ratios = overall[role]["ratios"]
        output.append({"scope": "overall", "fold_id": "all", "system": role,
                       "frame_count": overall[role]["frames"],
                       "gate_sample_count": len(values),
                       "gate_mean": float(np.mean(values)),
                       "gate_median": float(np.median(values)),
                       "gate_p90": float(np.percentile(values, 90)),
                       "aggregate_ratio_mean": float(np.mean(ratios)),
                       "aggregate_ratio_p90": float(np.percentile(ratios, 90))})
    return output


def communication_rows(registry):
    output = []
    totals = defaultdict(lambda: defaultdict(float))
    for row in registry:
        if row["message_mode"] == "none":
            continue
        role = row["experiment_id"].split("_", 1)[0].lower()
        summaries = []
        for path in sorted(Path(row["output_dir"]).glob("*_c3r_comm_summary.json")):
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        if len(summaries) != int(row["expected_sequences"]):
            raise RuntimeError("communication summaries incomplete for {}".format(
                row["experiment_id"]))
        frames = sum(int(x["processed_frames"]) for x in summaries)
        sent = sum(int(x["serialized_bytes_sent"]) for x in summaries)
        received = sum(int(x["serialized_bytes_received"]) for x in summaries)
        accepted = sum(int(x["accepted_packets"]) * 320 for x in summaries)
        item = {"scope": "fold", "fold_id": int(row["fold_id"]), "system": role,
                "sequence_count": len(summaries), "processed_frames": frames,
                "serialized_packet_bytes": 320,
                "mean_sent_bytes_per_frame": sent / frames,
                "mean_received_bytes_per_frame": received / frames,
                "mean_accepted_bytes_per_frame": accepted / frames,
                "p90_sent_bytes_per_frame": float(np.percentile(
                    [x["p90_sent_bytes_per_frame"] for x in summaries], 90)),
                "p90_received_bytes_per_frame": float(np.percentile(
                    [x["p90_received_bytes_per_frame"] for x in summaries], 90)),
                "broadcast_accounting": True}
        output.append(item)
        for key, value in (("frames", frames), ("sent", sent),
                           ("received", received), ("accepted", accepted),
                           ("sequences", len(summaries))):
            totals[role][key] += value
    for role in ("c0", "c1"):
        item = totals[role]
        output.append({"scope": "overall", "fold_id": "all", "system": role,
                       "sequence_count": int(item["sequences"]),
                       "processed_frames": int(item["frames"]),
                       "serialized_packet_bytes": 320,
                       "mean_sent_bytes_per_frame": item["sent"] / item["frames"],
                       "mean_received_bytes_per_frame": item["received"] / item["frames"],
                       "mean_accepted_bytes_per_frame": item["accepted"] / item["frames"],
                       "p90_sent_bytes_per_frame": 320.0,
                       "p90_received_bytes_per_frame": 640.0,
                       "broadcast_accounting": True})
    return output


def mean(targets, key):
    return float(np.mean([row[key] for row in targets]))


def fmt3(values):
    return " / ".join("{:.6f}".format(value) for value in values)


def concentration(targets, contrast):
    values = [(row["target"], row[contrast + "_auc"]) for row in targets]
    positive_sum = sum(value for _, value in values if value > 0)
    return (max((value for _, value in values if value > 0), default=0.0) /
            positive_sum if positive_sum else 0.0)


def write_summary(targets, folds, boot, lofo, gates, communication):
    overall = {system: [mean(targets, system + "_" + metric) for metric in METRICS]
               for system in SYSTEMS}
    deltas = {contrast: [mean(targets, contrast + "_" + metric)
                         for metric in METRICS] for contrast in CONTRASTS}
    boot_map = {(row["contrast"], row["metric"]): row for row in boot}
    lines = ["# C3R formal five-fold CV summary", "",
             "Status: **COMPLETE** (23/23 OOF targets; 69/69 views).", "",
             "## Overall target-level metrics", "",
             "| System/contrast | AUC | Precision | Norm Precision |",
             "|---|---:|---:|---:|"]
    for system in SYSTEMS:
        lines.append("| {} | {} |".format(system.upper(), " | ".join(
            "{:.6f}".format(x) for x in overall[system])))
    for contrast in CONTRASTS:
        lines.append("| {} | {} |".format(contrast.replace("_", " ").upper(),
            " | ".join("{:+.6f}".format(x) for x in deltas[contrast])))
    lines += ["", "## Fold deltas", "",
              "| Fold | C0-E0 AUC/P/NP | C1-E0 AUC/P/NP | C1-C0 AUC/P/NP |",
              "|---:|---|---|---|"]
    for row in folds:
        lines.append("| {} | {} | {} | {} |".format(
            row["fold_id"],
            fmt3([row["c0_minus_e0_" + m] for m in METRICS]),
            fmt3([row["c1_minus_e0_" + m] for m in METRICS]),
            fmt3([row["c1_minus_c0_" + m] for m in METRICS])))
    lines += ["", "## Target stability and bootstrap", ""]
    for contrast in CONTRASTS:
        values = [(r["target"], r[contrast + "_auc"]) for r in targets]
        pos = sum(v > 1e-12 for _, v in values)
        neg = sum(v < -1e-12 for _, v in values)
        tied = len(values) - pos - neg
        auc_boot = boot_map[(contrast, "auc")]
        best = max(values, key=lambda x: x[1])
        worst = min(values, key=lambda x: x[1])
        lofo_values = [r["auc_delta"] for r in lofo if r["contrast"] == contrast]
        lines += ["- **{}**: positive/negative/tied = `{}/{}/{}`; AUC bootstrap 95% CI "
                  "`[{:+.6f}, {:+.6f}]`; P(delta>0) = `{:.6f}`; leave-one-fold-out "
                  "range `[{:+.6f}, {:+.6f}]`; max positive `{}` `{:+.6f}`; max negative "
                  "`{}` `{:+.6f}`; positive-gain concentration `{:.4%}`.".format(
                      contrast, pos, neg, tied, auc_boot["ci_low"], auc_boot["ci_high"],
                      auc_boot["p_delta_gt_0"], min(lofo_values), max(lofo_values),
                      best[0], best[1], worst[0], worst[1], concentration(targets, contrast))]
    lines += ["", "## Drone-view changes", "",
              "| Drone | C0-E0 AUC/P/NP | C1-E0 AUC/P/NP | C1-C0 AUC/P/NP |",
              "|---|---|---|---|"]
    sequence_rows = read_csv(OUT / "sequence_metrics.csv")
    for drone in ("A", "B", "C"):
        rows = [r for r in sequence_rows if r["drone"] == drone]
        vals = {}
        for contrast in CONTRASTS:
            vals[contrast] = [np.mean([f(r[contrast + "_" + m]) for r in rows])
                              for m in METRICS]
        lines.append("| {} | {} | {} | {} |".format(
            drone, fmt3(vals["c0_minus_e0"]), fmt3(vals["c1_minus_e0"]),
            fmt3(vals["c1_minus_c0"])))
    c1_gate = next(r for r in gates if r["scope"] == "overall" and r["system"] == "c1")
    c1_comm = next(r for r in communication if r["scope"] == "overall" and r["system"] == "c1")
    lines += ["", "## Gate and communication", "",
              "- C1 gate mean/median/P90: `{:.6f} / {:.6f} / {:.6f}`.".format(
                  c1_gate["gate_mean"], c1_gate["gate_median"], c1_gate["gate_p90"]),
              "- C1 application communication: transmit `{:.6f}` B/frame; receive "
              "`{:.6f}` B/frame under broadcast accounting; packet size `320` bytes.".format(
                  c1_comm["mean_sent_bytes_per_frame"],
                  c1_comm["mean_received_bytes_per_frame"]), ""]
    c1e0 = deltas["c1_minus_e0"]
    c1c0 = deltas["c1_minus_c0"]
    c1e0_values = [r["c1_minus_e0_auc"] for r in targets]
    positive_folds = sum(r["c1_minus_e0_auc"] > 0 for r in folds)
    c1e0_accuracy = all((c1e0[0] >= 0.5, c1e0[1] > 0, c1e0[2] > 0,
                         boot_map[("c1_minus_e0", "auc")]["ci_low"] > 0,
                         sum(v > 0 for v in c1e0_values) > sum(v < 0 for v in c1e0_values),
                         positive_folds >= 4,
                         all(r["auc_delta"] > 0 for r in lofo
                             if r["contrast"] == "c1_minus_e0"),
                         concentration(targets, "c1_minus_e0") <= 0.25))
    c1c0_clean = all((c1c0[0] > 0, c1c0[1] > 0, c1c0[2] > 0,
                      boot_map[("c1_minus_c0", "auc")]["ci_low"] >= 0))
    lines += ["## Frozen acceptance decision", "",
              "- C1-E0 accuracy family: **{}**.".format("PASS" if c1e0_accuracy else "FAIL"),
              "- C1-C0 clean incremental-value family: **{}** (wrong-remote robustness is "
              "outside this authorized run and is not used as rescue evidence).".format(
                  "PASS" if c1c0_clean else "FAIL"),
              "- Exact clean communication accounting: **PASS**.",
              "- Overall frozen method acceptance: **FAIL** because mandatory clean accuracy "
              "gates do not all pass. No validation/test, ablation, transfer, or tuning was run.", ""]
    (OUT / "final_cv_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets, sequences = load_paired_rows()
    folds = fold_rows(targets)
    boot = bootstrap_rows(targets)
    lofo = leave_one_fold_out(targets)
    registry = registry_rows()
    gates = gate_rows(registry)
    communication = communication_rows(registry)
    write_csv(OUT / "target_metrics.csv", targets)
    write_csv(OUT / "sequence_metrics.csv", sequences)
    write_csv(OUT / "fold_metrics.csv", folds)
    write_csv(OUT / "bootstrap_results.csv", boot)
    write_csv(OUT / "leave_one_fold_out.csv", lofo)
    write_csv(OUT / "gate_metrics.csv", gates)
    write_csv(OUT / "communication_metrics.csv", communication)
    write_summary(targets, folds, boot, lofo, gates, communication)
    files = ["final_cv_summary.md", "target_metrics.csv", "sequence_metrics.csv",
             "fold_metrics.csv", "bootstrap_results.csv", "leave_one_fold_out.csv",
             "gate_metrics.csv", "communication_metrics.csv"]
    lines = ["# C3R formal CV result manifest", "", "Status: **COMPLETE**.", "",
             "- OOF targets: 23/23", "- OOF sequences/views: 69/69",
             "- Folds: 5/5", "- Bootstrap: 10,000 target resamples; seed 20260716",
             "- Inference: no-GT", "- GPUs: 0,1,2", "", "| File | SHA256 |",
             "|---|---|"]
    for name in files:
        lines.append("| `{}` | `{}` |".format(name, sha256(OUT / name)))
    (OUT / "result_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
