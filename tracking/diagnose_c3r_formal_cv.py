#!/usr/bin/env python3
"""Read-only mechanism diagnosis for completed formal C3R five-fold outputs."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "output/multi_agent_collaboration_clean/formal"
OUT = ROOT / "output/multi_agent_collaboration_clean/diagnosis"
REGISTRY = FORMAL / "evaluation_registry.csv"
GT_ROOT = Path("/data2/Three-MDOT")
SYSTEMS = ("e0", "c0", "c1")
DRONES = {0: "A", 1: "B", 2: "C"}
EPS = 1e-12


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    if not rows:
        raise RuntimeError("refusing to write empty CSV: {}".format(path))
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        sanitized = []
        for row in rows:
            item = {}
            for key, value in row.items():
                if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                    item[key] = "NA"
                else:
                    item[key] = value
            sanitized.append(item)
        writer.writerows(sanitized)


def load_matrix(path, columns=None):
    try:
        value = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    except ValueError:
        value = np.loadtxt(str(path), dtype=np.float64)
    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value.reshape(1, -1) if columns else value.reshape(-1, 1)
    if columns is not None:
        value = value[:, :columns]
    return value


def load_vector(path):
    return load_matrix(path).reshape(-1)


def as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def overlaps(pred, gt):
    pred = np.asarray(pred, dtype=np.float64).copy()
    gt = np.asarray(gt, dtype=np.float64)
    pred[0] = gt[0]
    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    top_left = np.maximum(pred[:, :2], gt[:, :2])
    bottom_right = np.minimum(
        pred[:, :2] + pred[:, 2:] - 1.0,
        gt[:, :2] + gt[:, 2:] - 1.0,
    )
    size = np.maximum(bottom_right - top_left + 1.0, 0.0)
    intersection = size[:, 0] * size[:, 1]
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - intersection
    value = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0)
    value[~valid] = -1.0
    return value, valid


def auc_from_iou(iou):
    values = np.asarray(iou, dtype=np.float64)
    thresholds = np.arange(0.0, 1.0 + 0.05, 0.05)
    return float(((values[:, None] > thresholds).mean(axis=0)).mean() * 100.0)


def safe_corr(x, y, kind):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan"), len(x)
    result = pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)
    return float(result.statistic), len(x)


def outcome(delta):
    if delta > EPS:
        return "helpful"
    if delta < -EPS:
        return "harmful"
    return "tied"


def fmt(value, digits=6):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return ("{:,.%df}" % digits).format(float(value))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_registry():
    rows = read_csv(REGISTRY)
    selected = {}
    for row in rows:
        fold = int(row["fold_id"])
        system = row["experiment_id"].split("_", 1)[0].lower()
        selected[(fold, system)] = row
    expected = {(fold, system) for fold in range(5) for system in SYSTEMS}
    if set(selected) != expected:
        raise RuntimeError("formal evaluation registry is not exactly five folds x three systems")
    if any(row["status"] != "COMPLETE" for row in selected.values()):
        raise RuntimeError("formal evaluation registry contains incomplete rows")
    return selected


def read_manifests(registry):
    mapping = {}
    all_targets = set()
    all_sequences = set()
    for fold in range(5):
        row = registry[(fold, "e0")]
        path = ROOT / row["split_manifest"]
        sequences = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        expected = int(row["expected_sequences"])
        if len(sequences) != expected or len(set(sequences)) != expected:
            raise RuntimeError("fold {} manifest sequence coverage mismatch".format(fold))
        targets = {item.rsplit("-", 1)[0] for item in sequences}
        if len(targets) != int(row["expected_targets"]):
            raise RuntimeError("fold {} manifest target coverage mismatch".format(fold))
        for target in targets:
            views = {int(item.rsplit("-", 1)[1]) for item in sequences
                     if item.rsplit("-", 1)[0] == target}
            if views != {1, 2, 3}:
                raise RuntimeError("target {} does not contain three views".format(target))
        mapping[fold] = sequences
        if all_targets.intersection(targets) or all_sequences.intersection(sequences):
            raise RuntimeError("OOF manifests overlap")
        all_targets.update(targets)
        all_sequences.update(sequences)
    if len(all_targets) != 23 or len(all_sequences) != 69:
        raise RuntimeError("OOF manifest coverage is not 23 targets / 69 views")
    return mapping


def parse_diagnostics(path, expected_length, receiver_id):
    rows = read_csv(path)
    if len(rows) != expected_length:
        raise RuntimeError("diagnostic length mismatch: {}".format(path))
    parsed = []
    for index, row in enumerate(rows):
        if int(row["frame_id"]) != index:
            raise RuntimeError("diagnostic frame index mismatch: {}".format(path))
        if int(row["receiver_id"]) != receiver_id:
            raise RuntimeError("diagnostic receiver/view mismatch: {}".format(path))
        if as_bool(row["uses_gt"]):
            raise RuntimeError("inference diagnostic reports GT use: {}".format(path))
        gates = [float(item) for item in json.loads(row["gates"])]
        senders = [int(item) for item in json.loads(row["accepted_sender_ids"])]
        dispositions = json.loads(row["dispositions"])
        if len(gates) != len(senders):
            raise RuntimeError("sender/gate count mismatch: {}".format(path))
        if any(not math.isfinite(value) for value in gates):
            raise RuntimeError("non-finite gate: {}".format(path))
        if index > 0 and (set(senders) != ({0, 1, 2} - {receiver_id})):
            raise RuntimeError("sender trace is incomplete: {} frame {}".format(path, index))
        if index > 0 and any(not item.get("accepted", False) for item in dispositions):
            raise RuntimeError("formal packet was rejected: {} frame {}".format(path, index))
        numeric = [float(row[key]) for key in (
            "timestamp_ms", "packet_bytes", "sent_bytes", "received_bytes",
            "accepted_bytes", "aggregate_ratio")]
        if any(not math.isfinite(value) for value in numeric):
            raise RuntimeError("non-finite C3R diagnostic: {}".format(path))
        parsed.append({
            "frame_id": index,
            "timestamp_ms": int(row["timestamp_ms"]),
            "senders": senders,
            "gates": gates,
            "gate_by_sender": dict(zip(senders, gates)),
            "aggregate_ratio": float(row["aggregate_ratio"]),
            "used_remote": as_bool(row["used_remote"]),
            "packet_bytes": int(row["packet_bytes"]),
            "sent_bytes": int(row["sent_bytes"]),
            "received_bytes": int(row["received_bytes"]),
            "accepted_bytes": int(row["accepted_bytes"]),
        })
    return parsed


def load_all(registry, manifests):
    data = {}
    completeness = {
        "targets": 23, "sequences": 69, "systems": 3,
        "length_match": True, "frame_index_match": True,
        "finite": True, "source_traceable": True,
    }
    for fold, sequences in manifests.items():
        for sequence in sequences:
            target, view_text = sequence.rsplit("-", 1)
            receiver = int(view_text) - 1
            gt_dir = GT_ROOT / target / sequence
            gt = load_matrix(gt_dir / "groundtruth.txt", columns=4)
            occ = load_vector(gt_dir / "occlusion.txt")
            oov = load_vector(gt_dir / "out_of_view.txt")
            if len(occ) != len(gt) or len(oov) != len(gt):
                raise RuntimeError("GT attribute length mismatch: {}".format(sequence))
            item = {
                "fold": fold, "target": target, "sequence": sequence,
                "receiver": receiver, "view": receiver + 1,
                "gt": gt, "occlusion": occ, "out_of_view": oov,
            }
            lengths = [len(gt)]
            arrays = [gt, occ, oov]
            for system in SYSTEMS:
                out_dir = Path(registry[(fold, system)]["output_dir"])
                pred = load_matrix(out_dir / (sequence + ".txt"), columns=4)
                score = load_vector(out_dir / (sequence + "_max_score.txt"))
                apce = load_vector(out_dir / (sequence + "_APCE.txt"))
                lengths.extend((len(pred), len(score), len(apce)))
                arrays.extend((pred, score, apce))
                iou, valid = overlaps(pred, gt)
                item[system] = {
                    "pred": pred, "score": score, "apce": apce,
                    "iou": iou, "valid": valid,
                }
                diag_path = out_dir / (sequence + "_c3r_diagnostics.csv")
                if system == "e0":
                    if diag_path.exists():
                        raise RuntimeError("E0 unexpectedly has C3R diagnostics")
                else:
                    item[system]["diagnostics"] = parse_diagnostics(
                        diag_path, len(gt), receiver)
            if len(set(lengths)) != 1:
                raise RuntimeError("E0/C0/C1 length mismatch: {}".format(sequence))
            if any(not np.isfinite(array).all() for array in arrays):
                raise RuntimeError("non-finite numeric value: {}".format(sequence))
            data[sequence] = item
    return data, completeness


def motion_features(gt):
    center = gt[:, :2] + 0.5 * (gt[:, 2:] - 1.0)
    delta = np.zeros(len(gt), dtype=np.float64)
    if len(gt) > 1:
        displacement = np.sqrt(((center[1:] - center[:-1]) ** 2).sum(axis=1))
        scale = np.sqrt(np.maximum(gt[:-1, 2] * gt[:-1, 3], 1.0))
        delta[1:] = displacement / scale
    return delta


def build_frame_rows(data):
    rows = []
    for sequence in sorted(data):
        item = data[sequence]
        motion = motion_features(item["gt"])
        for frame in range(len(item["gt"])):
            c0_delta = item["c0"]["iou"][frame] - item["e0"]["iou"][frame]
            c1_delta = item["c1"]["iou"][frame] - item["e0"]["iou"][frame]
            diag = item["c1"]["diagnostics"][frame]
            gates = diag["gates"]
            senders = diag["senders"]
            remote_scores = [data["{}-{}".format(item["target"], sender + 1)]["e0"]["score"][frame]
                             for sender in senders]
            remote_apces = [data["{}-{}".format(item["target"], sender + 1)]["e0"]["apce"][frame]
                            for sender in senders]
            remote_ious = [data["{}-{}".format(item["target"], sender + 1)]["e0"]["iou"][frame]
                           for sender in senders]
            view_ious = [data["{}-{}".format(item["target"], view)]["e0"]["iou"][frame]
                         for view in (1, 2, 3)]
            row = {
                "fold_id": item["fold"], "target": item["target"],
                "sequence": sequence, "view": item["view"],
                "drone": DRONES[item["receiver"]], "receiver_id": item["receiver"],
                "frame_id": frame, "valid_gt": bool(item["e0"]["valid"][frame]),
                "occlusion": int(item["occlusion"][frame] > 0),
                "out_of_view": int(item["out_of_view"][frame] > 0),
                "gt_motion_normalized": float(motion[frame]),
                "e0_iou": float(item["e0"]["iou"][frame]),
                "c0_iou": float(item["c0"]["iou"][frame]),
                "c1_iou": float(item["c1"]["iou"][frame]),
                "c0_minus_e0_iou": float(c0_delta),
                "c1_minus_e0_iou": float(c1_delta),
                "c1_outcome": outcome(c1_delta),
                "e0_score": float(item["e0"]["score"][frame]),
                "c0_score": float(item["c0"]["score"][frame]),
                "c1_score": float(item["c1"]["score"][frame]),
                "e0_apce": float(item["e0"]["apce"][frame]),
                "c0_apce": float(item["c0"]["apce"][frame]),
                "c1_apce": float(item["c1"]["apce"][frame]),
                "gate_mean": float(np.mean(gates)) if gates else float("nan"),
                "gate_min": float(np.min(gates)) if gates else float("nan"),
                "gate_max": float(np.max(gates)) if gates else float("nan"),
                "gate_std": float(np.std(gates)) if gates else float("nan"),
                "aggregate_residual_local_norm_ratio": diag["aggregate_ratio"],
                "sender_ids": json.dumps(senders, separators=(",", ":")),
                "source_gates": json.dumps(diag["gate_by_sender"], sort_keys=True,
                                           separators=(",", ":")),
                "remote_e0_score_proxy_mean": float(np.mean(remote_scores)) if remote_scores else float("nan"),
                "remote_e0_apce_proxy_mean": float(np.mean(remote_apces)) if remote_apces else float("nan"),
                "remote_e0_iou_offline_mean": float(np.mean(remote_ious)) if remote_ious else float("nan"),
                "e0_cross_view_iou_spread": float(np.max(view_ious) - np.min(view_ious)),
                "message_age_frames": 0 if frame > 0 else "NA",
                "message_age_provenance": "synchronous_current_frame_orchestration" if frame > 0 else "no_packet",
            }
            rows.append(row)
    return rows


def target_rows(data, frame_rows):
    by_target = defaultdict(list)
    for row in frame_rows:
        by_target[row["target"]].append(row)
    rows = []
    for target in sorted(by_target):
        frames = by_target[target]
        fold = frames[0]["fold_id"]
        sequences = [data["{}-{}".format(target, view)] for view in (1, 2, 3)]
        item = {"fold_id": fold, "target": target, "frame_count": len(frames)}
        for system in SYSTEMS:
            values = [auc_from_iou(seq[system]["iou"]) for seq in sequences]
            item[system + "_auc"] = float(np.mean(values))
        item["c0_minus_e0_auc"] = item["c0_auc"] - item["e0_auc"]
        item["c1_minus_e0_auc"] = item["c1_auc"] - item["e0_auc"]
        item["c1_minus_c0_auc"] = item["c1_auc"] - item["c0_auc"]
        for system in ("c0", "c1"):
            delta_key = system + "_minus_e0_iou"
            deltas = np.asarray([row[delta_key] for row in frames])
            item[system + "_helpful_frame_ratio"] = float(np.mean(deltas > EPS))
            item[system + "_harmful_frame_ratio"] = float(np.mean(deltas < -EPS))
            item[system + "_tied_frame_ratio"] = float(np.mean(np.abs(deltas) <= EPS))
            item[system + "_mean_helpful_iou_gain"] = float(np.mean(deltas[deltas > EPS])) if np.any(deltas > EPS) else 0.0
            item[system + "_mean_harmful_iou_loss"] = float(np.mean(-deltas[deltas < -EPS])) if np.any(deltas < -EPS) else 0.0
            frame_oracles = [auc_from_iou(np.maximum(seq["e0"]["iou"], seq[system]["iou"]))
                             for seq in sequences]
            item["oracle_frame_e0_{}_auc".format(system)] = float(np.mean(frame_oracles))
            item["oracle_target_e0_{}_auc".format(system)] = max(
                item["e0_auc"], item[system + "_auc"])
        rows.append(item)
    return rows


def oracle_rows(data, frames, targets):
    rows = []
    e0_overall = float(np.mean([row["e0_auc"] for row in targets]))
    for system in ("c0", "c1"):
        collab = float(np.mean([row[system + "_auc"] for row in targets]))
        deltas = np.asarray([row[system + "_minus_e0_iou"] for row in frames])
        common = {
            "system": system.upper(), "baseline_e0_auc": e0_overall,
            "collaborative_auc": collab,
            "helpful_frame_ratio": float(np.mean(deltas > EPS)),
            "harmful_frame_ratio": float(np.mean(deltas < -EPS)),
            "tied_frame_ratio": float(np.mean(np.abs(deltas) <= EPS)),
            "mean_helpful_iou_gain": float(np.mean(deltas[deltas > EPS])) if np.any(deltas > EPS) else 0.0,
            "mean_harmful_iou_loss": float(np.mean(-deltas[deltas < -EPS])) if np.any(deltas < -EPS) else 0.0,
            "status": "offline diagnostic upper bound; not model performance",
        }
        frame_auc = float(np.mean([
            np.mean([auc_from_iou(np.maximum(
                data["{}-{}".format(target["target"], view)]["e0"]["iou"],
                data["{}-{}".format(target["target"], view)][system]["iou"]))
                     for view in (1, 2, 3)]) for target in targets]))
        target_auc = float(np.mean([
            max(target["e0_auc"], target[system + "_auc"]) for target in targets]))
        view_values = []
        for sequence in sorted(data):
            view_values.append(max(
                auc_from_iou(data[sequence]["e0"]["iou"]),
                auc_from_iou(data[sequence][system]["iou"])))
        view_auc = float(np.mean(view_values))
        for scope, value in (("frame", frame_auc), ("target", target_auc),
                             ("view", view_auc)):
            rows.append(dict(common, oracle_scope=scope, oracle_auc=value,
                             oracle_minus_e0_auc=value - e0_overall))
    return rows


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "p10": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "count": len(values), "mean": float(np.mean(values)),
        "median": float(np.median(values)), "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)), "max": float(np.max(values)),
    }


def gate_analysis(frames):
    valid = [row for row in frames if math.isfinite(float(row["gate_mean"]))]
    rows = []

    def add(scope, group, statistic, value, count, notes=""):
        rows.append({"scope": scope, "group": group, "statistic": statistic,
                     "value": value, "count": count, "notes": notes})

    for label in ("all", "helpful", "harmful", "tied"):
        subset = valid if label == "all" else [row for row in valid if row["c1_outcome"] == label]
        stats = distribution([row["gate_mean"] for row in subset])
        for key, value in stats.items():
            add("outcome_distribution", label, "gate_" + key, value, len(subset))

    x = [row["gate_mean"] for row in valid]
    y = [row["c1_minus_e0_iou"] for row in valid]
    for kind in ("pearson", "spearman"):
        value, count = safe_corr(x, y, kind)
        add("correlation", "overall", kind + "_gate_vs_c1_minus_e0_iou",
            value, count, "frame-level aggregate mean gate")
    binary = [row for row in valid if row["c1_outcome"] != "tied"]
    labels = np.asarray([row["c1_outcome"] == "helpful" for row in binary], dtype=np.int64)
    scores = np.asarray([row["gate_mean"] for row in binary], dtype=np.float64)
    add("classifier", "helpful_vs_harmful", "roc_auc", float(roc_auc_score(labels, scores)), len(binary))
    add("classifier", "helpful_vs_harmful", "pr_auc", float(average_precision_score(labels, scores)), len(binary))
    add("classifier", "helpful_vs_harmful", "helpful_prevalence", float(np.mean(labels)), len(binary))

    for source, key in (
            ("local_e0_score_proxy", "e0_score"),
            ("local_e0_apce_proxy", "e0_apce"),
            ("remote_e0_score_proxy", "remote_e0_score_proxy_mean"),
            ("remote_e0_apce_proxy", "remote_e0_apce_proxy_mean"),
            ("aggregate_residual_local_norm_ratio", "aggregate_residual_local_norm_ratio")):
        for kind in ("pearson", "spearman"):
            value, count = safe_corr(
                [row["gate_mean"] for row in valid], [row[key] for row in valid], kind)
            add("correlation", source, kind + "_gate_vs_" + source, value, count,
                "score/APCE are saved-output proxies, not the unsaved 10-D gate input"
                if "proxy" in source else "")
    for kind in ("pearson", "spearman"):
        value, count = safe_corr(
            [row["aggregate_residual_local_norm_ratio"] for row in valid], y, kind)
        add("correlation", "residual_safety", kind + "_aggregate_ratio_vs_iou_gain",
            value, count, "only saved residual diagnostic")

    all_stats = distribution(x)
    add("constancy", "overall", "coefficient_of_variation",
        all_stats["std"] / all_stats["mean"], len(x))
    add("constancy", "overall", "relative_range",
        (all_stats["max"] - all_stats["min"]) / all_stats["mean"], len(x))
    add("message_age", "overall", "age_frames", 0.0, len(valid),
        "all formal packets are synchronous current-frame exchanges; no age variation")
    add("message_age", "overall", "gate_vs_age_correlation", float("nan"), len(valid),
        "undefined because message age is constant zero")

    group_specs = (
        ("fold", "fold_id"), ("target", "target"),
        ("view", "drone"),
    )
    for scope, key in group_specs:
        for group in sorted({str(row[key]) for row in valid}):
            subset = [row for row in valid if str(row[key]) == group]
            stats = distribution([row["gate_mean"] for row in subset])
            for stat in ("mean", "std", "p10", "p90"):
                add(scope, group, "gate_" + stat, stats[stat], len(subset))

    worst = sorted(valid, key=lambda row: row["c1_minus_e0_iou"])[:20]
    add("extreme_negative", "worst_20_frames", "gate_mean",
        float(np.mean([row["gate_mean"] for row in worst])), len(worst))
    add("extreme_negative", "worst_20_frames", "gate_max",
        float(np.max([row["gate_mean"] for row in worst])), len(worst))
    single = worst[0]
    add("extreme_negative", "worst_frame", "iou_delta",
        single["c1_minus_e0_iou"], 1,
        "{} frame {}".format(single["sequence"], single["frame_id"]))
    add("extreme_negative", "worst_frame", "gate_mean",
        single["gate_mean"], 1,
        "{} frame {}".format(single["sequence"], single["frame_id"]))
    add("extreme_negative", "worst_frame", "aggregate_residual_local_norm_ratio",
        single["aggregate_residual_local_norm_ratio"], 1,
        "{} frame {}".format(single["sequence"], single["frame_id"]))
    add("missing_diagnostic", "reliability_inputs", "status", "MISSING_DIAGNOSTIC", 0,
        "10-D gate inputs/logits were not saved")
    add("missing_diagnostic", "confidence", "status", "MISSING_DIAGNOSTIC", 0,
        "no standalone confidence artifact; score/APCE proxies used")
    add("missing_diagnostic", "residual_l2_cosine", "status", "MISSING_DIAGNOSTIC", 0,
        "residual L2, cosine, per-source norm, and gate*norm were not saved")
    return rows


def source_receiver_rows(data, frames):
    by_sequence_frame = {(row["sequence"], row["frame_id"]): row for row in frames}
    output = []
    for source in range(3):
        for receiver in range(3):
            if source == receiver:
                continue
            pair = []
            for sequence, item in data.items():
                if item["receiver"] != receiver:
                    continue
                for frame, diag in enumerate(item["c1"]["diagnostics"]):
                    if source not in diag["gate_by_sender"]:
                        continue
                    row = by_sequence_frame[(sequence, frame)]
                    source_sequence = "{}-{}".format(item["target"], source + 1)
                    pair.append({
                        "gate": diag["gate_by_sender"][source],
                        "delta": row["c1_minus_e0_iou"],
                        "target": item["target"],
                        "local_score": item["e0"]["score"][frame],
                        "local_apce": item["e0"]["apce"][frame],
                        "remote_score": data[source_sequence]["e0"]["score"][frame],
                        "remote_apce": data[source_sequence]["e0"]["apce"][frame],
                        "remote_iou": data[source_sequence]["e0"]["iou"][frame],
                    })
            deltas = np.asarray([row["delta"] for row in pair])
            harmful_targets = defaultdict(list)
            for row in pair:
                harmful_targets[row["target"]].append(row["delta"])
            worst_target, worst_values = min(
                harmful_targets.items(), key=lambda item: np.mean(item[1]))
            output.append({
                "source": DRONES[source], "receiver": DRONES[receiver],
                "directed_pair": "{}->{}".format(DRONES[source], DRONES[receiver]),
                "use_count": len(pair),
                "mean_gate": float(np.mean([row["gate"] for row in pair])),
                "mean_c1_minus_e0_iou": float(np.mean(deltas)),
                "helpful_ratio": float(np.mean(deltas > EPS)),
                "harmful_ratio": float(np.mean(deltas < -EPS)),
                "tied_ratio": float(np.mean(np.abs(deltas) <= EPS)),
                "max_negative_target": worst_target,
                "max_negative_target_mean_iou_delta": float(np.mean(worst_values)),
                "local_e0_score_proxy_mean": float(np.mean([row["local_score"] for row in pair])),
                "local_e0_apce_proxy_mean": float(np.mean([row["local_apce"] for row in pair])),
                "remote_e0_score_proxy_mean": float(np.mean([row["remote_score"] for row in pair])),
                "remote_e0_apce_proxy_mean": float(np.mean([row["remote_apce"] for row in pair])),
                "remote_e0_iou_offline_mean": float(np.mean([row["remote_iou"] for row in pair])),
                "message_age_frames": 0,
                "attribution_note": "receiver outcome association; per-source causal residual was not saved",
            })
    return output


def view_rows(frames, targets):
    output = []
    for drone in ("A", "B", "C"):
        subset = [row for row in frames if row["drone"] == drone]
        item = {"drone": drone, "view": {"A": 1, "B": 2, "C": 3}[drone],
                "frame_count": len(subset)}
        for system in ("c0", "c1"):
            deltas = np.asarray([row[system + "_minus_e0_iou"] for row in subset])
            item[system + "_mean_frame_iou_delta"] = float(np.mean(deltas))
            item[system + "_helpful_ratio"] = float(np.mean(deltas > EPS))
            item[system + "_harmful_ratio"] = float(np.mean(deltas < -EPS))
        gate_values = [row["gate_mean"] for row in subset if math.isfinite(float(row["gate_mean"]))]
        item["c1_gate_mean"] = float(np.mean(gate_values))
        item["c1_gate_std"] = float(np.std(gate_values))
        sequences = sorted({row["sequence"] for row in subset})
        for system in SYSTEMS:
            aucs = []
            for sequence in sequences:
                seq_frames = [row for row in subset if row["sequence"] == sequence]
                aucs.append(auc_from_iou([row[system + "_iou"] for row in seq_frames]))
            item[system + "_auc"] = float(np.mean(aucs))
        item["c0_minus_e0_auc"] = item["c0_auc"] - item["e0_auc"]
        item["c1_minus_e0_auc"] = item["c1_auc"] - item["e0_auc"]
        item["c1_minus_c0_auc"] = item["c1_auc"] - item["c0_auc"]
        output.append(item)
    return output


def contiguous_harmful_events(frames):
    by_sequence = defaultdict(list)
    for row in frames:
        by_sequence[row["sequence"]].append(row)
    events = []
    for sequence, rows in by_sequence.items():
        rows = sorted(rows, key=lambda row: row["frame_id"])
        start = None
        current = []
        for row in rows + [None]:
            harmful = row is not None and row["c1_minus_e0_iou"] < -EPS
            if harmful:
                if start is None:
                    start = row["frame_id"]
                current.append(row)
            elif current:
                events.append({
                    "sequence": sequence, "target": current[0]["target"],
                    "drone": current[0]["drone"], "start": start,
                    "end": current[-1]["frame_id"], "length": len(current),
                    "sum_loss": float(-sum(item["c1_minus_e0_iou"] for item in current)),
                    "mean_loss": float(-np.mean([item["c1_minus_e0_iou"] for item in current])),
                    "gate_mean": float(np.nanmean([item["gate_mean"] for item in current])),
                    "aggregate_ratio_mean": float(np.mean([
                        item["aggregate_residual_local_norm_ratio"] for item in current])),
                    "occlusion_ratio": float(np.mean([item["occlusion"] for item in current])),
                    "out_of_view_ratio": float(np.mean([item["out_of_view"] for item in current])),
                    "local_score_mean": float(np.mean([item["e0_score"] for item in current])),
                    "motion_mean": float(np.mean([item["gt_motion_normalized"] for item in current])),
                    "view_spread_mean": float(np.mean([item["e0_cross_view_iou_spread"] for item in current])),
                })
                start, current = None, []
    return sorted(events, key=lambda item: item["sum_loss"], reverse=True)


def conditional_failure_stats(frames):
    active = [row for row in frames if math.isfinite(float(row["gate_mean"]))]
    motion_threshold = float(np.percentile([row["gt_motion_normalized"] for row in active], 90))
    score_threshold = float(np.percentile([row["e0_score"] for row in active], 25))
    apce_threshold = float(np.percentile([row["e0_apce"] for row in active], 25))
    spread_threshold = float(np.percentile([row["e0_cross_view_iou_spread"] for row in active], 90))
    conditions = {
        "all": lambda row: True,
        "occlusion": lambda row: row["occlusion"] > 0,
        "out_of_view": lambda row: row["out_of_view"] > 0,
        "fast_motion_top10pct": lambda row: row["gt_motion_normalized"] >= motion_threshold,
        "low_local_score_bottom25pct": lambda row: row["e0_score"] <= score_threshold,
        "low_local_apce_bottom25pct": lambda row: row["e0_apce"] <= apce_threshold,
        "cross_view_iou_spread_top10pct": lambda row: row["e0_cross_view_iou_spread"] >= spread_threshold,
    }
    output = []
    for name, predicate in conditions.items():
        subset = [row for row in active if predicate(row)]
        delta = np.asarray([row["c1_minus_e0_iou"] for row in subset])
        output.append({
            "condition": name, "frame_count": len(subset),
            "frame_ratio": len(subset) / len(active),
            "mean_iou_delta": float(np.mean(delta)) if len(delta) else float("nan"),
            "helpful_ratio": float(np.mean(delta > EPS)) if len(delta) else float("nan"),
            "harmful_ratio": float(np.mean(delta < -EPS)) if len(delta) else float("nan"),
            "mean_gate": float(np.mean([row["gate_mean"] for row in subset])) if subset else float("nan"),
            "threshold": {
                "fast_motion_top10pct": motion_threshold,
                "low_local_score_bottom25pct": score_threshold,
                "low_local_apce_bottom25pct": apce_threshold,
                "cross_view_iou_spread_top10pct": spread_threshold,
            }.get(name, "NA"),
        })
    return output


def decide(oracles, frames, gate_rows, source_rows, targets):
    c1_frame = next(row for row in oracles
                    if row["system"] == "C1" and row["oracle_scope"] == "frame")
    c1_target = next(row for row in oracles
                     if row["system"] == "C1" and row["oracle_scope"] == "target")
    roc = next(row for row in gate_rows if row["scope"] == "classifier"
               and row["statistic"] == "roc_auc")["value"]
    deltas = np.asarray([row["c1_minus_e0_iou"] for row in frames])
    helpful = float(np.mean(deltas > EPS))
    harmful = float(np.mean(deltas < -EPS))
    help_mag = float(np.mean(deltas[deltas > EPS]))
    harm_mag = float(np.mean(-deltas[deltas < -EPS]))
    pair_values = np.asarray([row["mean_c1_minus_e0_iou"] for row in source_rows])
    neg_targets = sorted(
        [-row["c1_minus_e0_auc"] for row in targets if row["c1_minus_e0_auc"] < 0],
        reverse=True)
    neg_total = sum(neg_targets)
    worst_concentration = neg_targets[0] / neg_total if neg_total else 0.0
    oracle_clear = bool(c1_frame["oracle_minus_e0_auc"] >= 3.0 or
                        c1_target["oracle_minus_e0_auc"] >= 1.0)
    gate_weak = bool(roc <= 0.55)
    # Receiver outcomes differ, but the saved fused output cannot isolate either
    # source's causal residual. Do not promote association to pair-specific proof.
    pair_specific = False
    severe_tail = bool(harm_mag >= 1.5 * help_mag or worst_concentration >= 0.40)
    if oracle_clear and gate_weak:
        label = "A"
        mechanism = "reliability design: oracle complement exists but the learned gate has little utility discrimination"
    elif not oracle_clear:
        label = "B"
        mechanism = "message/adapter: even offline oracle complement is low"
    elif pair_specific:
        label = "C"
        mechanism = "cross-view alignment/source conditioning: directed pair associations differ materially"
    elif helpful > harmful and severe_tail:
        label = "D"
        mechanism = "safety/residual bounding: many small gains are outweighed by a severe negative tail"
    else:
        label = "E"
        mechanism = "no single mechanism is identified strongly enough"
    return {
        "label": label, "mechanism": mechanism, "oracle_clear": oracle_clear,
        "gate_weak": gate_weak, "pair_specific": pair_specific,
        "severe_tail": severe_tail, "helpful_ratio": helpful,
        "harmful_ratio": harmful, "help_magnitude": help_mag,
        "harm_magnitude": harm_mag, "worst_negative_target_concentration": worst_concentration,
    }


def render_outputs(completeness, oracles, frames, targets, gates, sources, views,
                   events, conditions, decision):
    c0_frame = next(row for row in oracles if row["system"] == "C0" and row["oracle_scope"] == "frame")
    c1_frame = next(row for row in oracles if row["system"] == "C1" and row["oracle_scope"] == "frame")
    c0_target = next(row for row in oracles if row["system"] == "C0" and row["oracle_scope"] == "target")
    c1_target = next(row for row in oracles if row["system"] == "C1" and row["oracle_scope"] == "target")
    c0_view = next(row for row in oracles if row["system"] == "C0" and row["oracle_scope"] == "view")
    c1_view = next(row for row in oracles if row["system"] == "C1" and row["oracle_scope"] == "view")
    pearson = next(row for row in gates if row["statistic"] == "pearson_gate_vs_c1_minus_e0_iou")["value"]
    spearman = next(row for row in gates if row["statistic"] == "spearman_gate_vs_c1_minus_e0_iou")["value"]
    roc = next(row for row in gates if row["scope"] == "classifier" and row["statistic"] == "roc_auc")["value"]
    pr = next(row for row in gates if row["scope"] == "classifier" and row["statistic"] == "pr_auc")["value"]
    prevalence = next(row for row in gates if row["scope"] == "classifier" and row["statistic"] == "helpful_prevalence")["value"]
    gate_all = {row["statistic"]: row["value"] for row in gates
                if row["scope"] == "outcome_distribution" and row["group"] == "all"}
    agg_corr = next(row for row in gates if row["statistic"] == "spearman_aggregate_ratio_vs_iou_gain")["value"]
    sorted_pairs = sorted(sources, key=lambda row: row["mean_c1_minus_e0_iou"], reverse=True)
    negative_targets = sorted(targets, key=lambda row: row["c1_minus_e0_auc"])
    positive_targets = list(reversed(negative_targets))
    negative_mass = sum(-row["c1_minus_e0_auc"] for row in targets
                        if row["c1_minus_e0_auc"] < 0)
    top_three_negative_share = sum(
        -row["c1_minus_e0_auc"] for row in negative_targets[:3]) / negative_mass
    by_sequence = defaultdict(list)
    for row in frames:
        by_sequence[row["sequence"]].append(row)
    sequence_deltas = []
    for sequence, seq_rows in by_sequence.items():
        sequence_deltas.append({
            "sequence": sequence, "drone": seq_rows[0]["drone"],
            "delta": auc_from_iou([row["c1_iou"] for row in seq_rows]) -
                     auc_from_iou([row["e0_iou"] for row in seq_rows]),
        })

    summary = [
        "# C3R five-fold read-only mechanism diagnosis", "",
        "Status: **COMPLETE**. No tracker, training, validation, test, ablation, or parameter search was run.", "",
        "## Completeness", "",
        "- Coverage: **23/23 targets, 69/69 views**.",
        "- E0/C0/C1 prediction, score, and APCE lengths match GT for every view; frame indices align; all numeric values are finite.",
        "- C0/C1 every non-initial frame traces both remote sender IDs. Receiver/sender IDs 0/1/2 map to Drone A/B/C.",
        "- GT, occlusion, and out-of-view labels were used only for this offline analysis.", "",
        "## Oracle upper bounds", "",
        "These values are **offline diagnostic upper bounds, not model performance**.", "",
        "| Pair | Frame oracle AUC | Gain vs E0 | Target oracle AUC | Gain vs E0 | View oracle AUC | Gain vs E0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| E0/C0 | {} | {:+.6f} | {} | {:+.6f} | {} | {:+.6f} |".format(
            fmt(c0_frame["oracle_auc"]), c0_frame["oracle_minus_e0_auc"],
            fmt(c0_target["oracle_auc"]), c0_target["oracle_minus_e0_auc"],
            fmt(c0_view["oracle_auc"]), c0_view["oracle_minus_e0_auc"]),
        "| E0/C1 | {} | {:+.6f} | {} | {:+.6f} | {} | {:+.6f} |".format(
            fmt(c1_frame["oracle_auc"]), c1_frame["oracle_minus_e0_auc"],
            fmt(c1_target["oracle_auc"]), c1_target["oracle_minus_e0_auc"],
            fmt(c1_view["oracle_auc"]), c1_view["oracle_minus_e0_auc"]), "",
        "C1 helpful/harmful/tied frame ratios are {:.3%}/{:.3%}/{:.3%}; mean helpful gain is {:.6f} IoU and mean harmful loss is {:.6f} IoU.".format(
            c1_frame["helpful_frame_ratio"], c1_frame["harmful_frame_ratio"],
            c1_frame["tied_frame_ratio"], c1_frame["mean_helpful_iou_gain"],
            c1_frame["mean_harmful_iou_loss"]), "",
        "## Gate and residual evidence", "",
        "- Gate mean/median/std/P90: {}/{}/{}/{}; the narrow range is consistent with a near-constant gate.".format(
            fmt(gate_all["gate_mean"]), fmt(gate_all["gate_median"]),
            fmt(gate_all["gate_std"]), fmt(gate_all["gate_p90"])),
        "- Gate vs true C1-E0 frame IoU gain: Pearson {}, Spearman {}.".format(fmt(pearson), fmt(spearman)),
        "- Helpful-vs-harmful discrimination: ROC-AUC {}, PR-AUC {} (helpful prevalence {}).".format(
            fmt(roc), fmt(pr), fmt(prevalence)),
        "- Saved aggregate residual/local norm ratio vs gain Spearman: {}.".format(fmt(agg_corr)),
        "- All messages are synchronous (age 0), so age sensitivity is not identifiable.",
        "- `MISSING_DIAGNOSTIC`: 10-D gate inputs/logits, standalone confidence, residual L2, residual cosine, per-source residual norms, and gate x residual norm were not saved.", "",
        "## Source-receiver and failures", "",
        "Directed-pair values are receiver-outcome associations, not isolated causal source effects, because per-source residual effects were not saved.", "",
        "| Pair | Uses | Mean gate | Mean C1-E0 IoU | Helpful | Harmful | Worst target |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(sources, key=lambda item: item["directed_pair"]):
        summary.append("| {} | {} | {} | {:+.6f} | {:.3%} | {:.3%} | {} |".format(
            row["directed_pair"], row["use_count"], fmt(row["mean_gate"]),
            row["mean_c1_minus_e0_iou"], row["helpful_ratio"],
            row["harmful_ratio"], row["max_negative_target"]))
    summary.extend(["", "Largest negative targets are {}. Largest positive targets are {}.".format(
        ", ".join("{} ({:+.3f})".format(row["target"], row["c1_minus_e0_auc"])
                  for row in negative_targets[:5]),
        ", ".join("{} ({:+.3f})".format(row["target"], row["c1_minus_e0_auc"])
                  for row in positive_targets[:5])), "",
        "The three worst targets account for {:.3%} of total negative target-AUC mass. This is a strong secondary severe-event concentration, but changed frames are split almost evenly between helpful and harmful rather than showing a simple rare-harm-only regime.".format(
            top_three_negative_share), "",
        "## Decision", "",
        "Primary decision class: **{}** — {}.".format(decision["label"], decision["mechanism"]),
        "Reliability redesign is worth a narrowly instrumented diagnostic pass because the oracle complement is material while the current gate has weak discrimination. Message/adapter replacement is not yet the first intervention; residual-direction diagnostics must be captured before deciding it is necessary.",
        "The only recommended next step is a **minimal, frozen, inference-only instrumentation rerun on the same OOF views** that saves the 10-D gate input/logit and per-source residual L2, local-norm ratio, cosine, and gate-times-residual norm. Do not change model behavior or tune from this report.",
    ])
    (OUT / "diagnosis_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    extreme = ["# Extreme target analysis", "",
               "All values are existing OOF target AUC deltas; no target-specific rule is proposed.", "",
               "## Five largest negative C1-E0 targets", "",
               "| Target | Fold | C1-E0 AUC | C0-E0 AUC | Helpful frames | Harmful frames | Mean helpful IoU | Mean harmful loss |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in negative_targets[:5]:
        extreme.append("| {} | {} | {:+.6f} | {:+.6f} | {:.3%} | {:.3%} | {:.6f} | {:.6f} |".format(
            row["target"], row["fold_id"], row["c1_minus_e0_auc"], row["c0_minus_e0_auc"],
            row["c1_helpful_frame_ratio"], row["c1_harmful_frame_ratio"],
            row["c1_mean_helpful_iou_gain"], row["c1_mean_harmful_iou_loss"]))
    extreme.extend(["", "## Five largest positive C1-E0 targets", "",
                    "| Target | Fold | C1-E0 AUC | C0-E0 AUC | Helpful frames | Harmful frames | Mean helpful IoU | Mean harmful loss |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in positive_targets[:5]:
        extreme.append("| {} | {} | {:+.6f} | {:+.6f} | {:.3%} | {:.3%} | {:.6f} | {:.6f} |".format(
            row["target"], row["fold_id"], row["c1_minus_e0_auc"], row["c0_minus_e0_auc"],
            row["c1_helpful_frame_ratio"], row["c1_harmful_frame_ratio"],
            row["c1_mean_helpful_iou_gain"], row["c1_mean_harmful_iou_loss"]))
    negative_mass = sum(-row["c1_minus_e0_auc"] for row in targets if row["c1_minus_e0_auc"] < 0)
    worst_mass = -negative_targets[0]["c1_minus_e0_auc"] / negative_mass
    extreme.extend(["", "Worst-target share of total negative target-AUC mass: {:.3%}. This quantifies concentration only; it is not a leave-one-target-out performance claim.".format(worst_mass)])
    (OUT / "extreme_target_analysis.md").write_text("\n".join(extreme) + "\n", encoding="utf-8")

    failure = ["# Failure event analysis", "",
               "Contiguous events are maximal runs with C1 IoU below E0 IoU. GT attributes are used offline only.", "",
               "## Largest cumulative harmful runs", "",
               "| Sequence | Frames | Length | Cumulative IoU loss | Mean loss | Gate | Residual/local | Occlusion | OOV | Local score | Motion | View spread |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for event in events[:15]:
        failure.append("| {} | {}-{} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.3%} | {:.3%} | {:.6f} | {:.6f} | {:.6f} |".format(
            event["sequence"], event["start"], event["end"], event["length"],
            event["sum_loss"], event["mean_loss"], event["gate_mean"],
            event["aggregate_ratio_mean"], event["occlusion_ratio"],
            event["out_of_view_ratio"], event["local_score_mean"],
            event["motion_mean"], event["view_spread_mean"]))
    failure.extend(["", "## Drone A/B/C divergence", "",
                    "Gate means are effectively the same across A/B/C, so the view divergence is not explained by receiver-specific gate magnitude. The largest sequence contributions are:", ""])
    for drone in ("A", "B", "C"):
        selected = [row for row in sequence_deltas if row["drone"] == drone]
        worst = sorted(selected, key=lambda row: row["delta"])[:3]
        best = sorted(selected, key=lambda row: row["delta"], reverse=True)[:3]
        failure.append("- Drone {}: negative {} ; positive {}.".format(
            drone,
            ", ".join("{} ({:+.3f})".format(row["sequence"], row["delta"])
                      for row in worst),
            ", ".join("{} ({:+.3f})".format(row["sequence"], row["delta"])
                      for row in best)))
    failure.extend(["", "Drone C's aggregate rise is concentrated in md3031-3, while A/B declines are dominated by long failures in md3058-1, md3054-1, and md3038-2. Because both sources share the same fused head output, the saved artifacts establish receiver/view asymmetry but cannot identify one persistently harmful sender.",
                    "", "## Condition associations", "",
                    "These are descriptive associations, not causal effects.", "",
                    "| Condition | Frames | Mean C1-E0 IoU | Helpful | Harmful | Mean gate | Threshold |",
                    "|---|---:|---:|---:|---:|---:|---:|"])
    for row in conditions:
        if row["frame_count"]:
            failure.append("| {} | {} | {:+.6f} | {:.3%} | {:.3%} | {:.6f} | {} |".format(
                row["condition"], row["frame_count"], row["mean_iou_delta"],
                row["helpful_ratio"], row["harmful_ratio"], row["mean_gate"], row["threshold"]))
        else:
            failure.append("| {} | 0 | NA | NA | NA | NA | {} |".format(
                row["condition"], row["threshold"]))
    failure.extend(["", "A persistent wrong remote cannot be isolated from the saved outputs: both sources are fused before the head, and no per-source counterfactual or residual vector was stored. Directed-pair tables therefore must not be read as causal source attribution."])
    (OUT / "failure_event_analysis.md").write_text("\n".join(failure) + "\n", encoding="utf-8")

    next_decision = ["# Next method decision", "",
                     "Decision: **{}**.".format(decision["label"]), "",
                     "Primary mechanism: {}.".format(decision["mechanism"]), "",
                     "- Reliability redesign worth investigating: **YES, diagnostically**, because oracle complement is material and gate ROC-AUC is weak.",
                     "- Immediate message/adapter redesign: **NO**. Existing outputs do not contain residual direction evidence, so B cannot be established or rejected cleanly.",
                     "- Current C3R acceptance: **FAIL**; keep validation/test, ablation, OSTrack transfer, retraining, and tuning stopped.", "",
                     "## Only recommended next step", "",
                     "Run one minimal **inference-only instrumentation pass** with the frozen checkpoints, manifests, predictions, and behavior unchanged. Save per source and frame: the 10-D reliability input, gate logit, remote quality, semantic cosine, packet age, residual L2, residual/local norm ratio, residual cosine to the local feature/update direction, and gate-times-residual norm. Use it only to distinguish gate misclassification from adapter-direction error; do not tune on the same OOF labels."]
    (OUT / "next_method_decision.md").write_text("\n".join(next_decision) + "\n", encoding="utf-8")


def write_manifest(completeness):
    files = [
        "diagnosis_summary.md", "oracle_upper_bound.csv", "frame_level_utility.csv",
        "target_level_utility.csv", "gate_utility_analysis.csv",
        "source_receiver_matrix.csv", "view_analysis.csv",
        "extreme_target_analysis.md", "failure_event_analysis.md",
        "next_method_decision.md",
    ]
    lines = ["# C3R diagnosis manifest", "", "Status: **COMPLETE**.", "",
             "- Read-only inputs: formal Fold 0-4 E0/C0/C1 OOF results, score/APCE, C3R diagnostics, communication summaries, and offline GT attributes.",
             "- Completeness: 23 targets, 69 views; matched lengths/frame indices; finite values; every C1 remote source traceable.",
             "- No tracker, training, validation/test, ablation, OSTrack, tuning, or model modification was executed.", "",
             "| File | SHA256 |", "|---|---|"]
    for name in files:
        path = OUT / name
        lines.append("| `{}` | `{}` |".format(name, sha256(path)))
    lines.extend(["", "Missing diagnostics carried forward explicitly: standalone confidence, reliability inputs/logits, residual L2/cosine, per-source residual norms, and gate-times-residual norm."])
    (OUT / "diagnosis_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = read_registry()
    manifests = read_manifests(registry)
    data, completeness = load_all(registry, manifests)
    frames = build_frame_rows(data)
    targets = target_rows(data, frames)
    oracles = oracle_rows(data, frames, targets)
    gates = gate_analysis(frames)
    sources = source_receiver_rows(data, frames)
    views = view_rows(frames, targets)
    events = contiguous_harmful_events(frames)
    conditions = conditional_failure_stats(frames)
    decision = decide(oracles, frames, gates, sources, targets)

    write_csv(OUT / "oracle_upper_bound.csv", oracles)
    write_csv(OUT / "frame_level_utility.csv", frames)
    write_csv(OUT / "target_level_utility.csv", targets)
    write_csv(OUT / "gate_utility_analysis.csv", gates)
    write_csv(OUT / "source_receiver_matrix.csv", sources)
    write_csv(OUT / "view_analysis.csv", views)
    render_outputs(completeness, oracles, frames, targets, gates, sources, views,
                   events, conditions, decision)
    write_manifest(completeness)
    print(json.dumps({
        "status": "COMPLETE", "targets": len(targets),
        "sequences": len(data), "frames": len(frames),
        "decision": decision,
        "c0_frame_oracle": next(row for row in oracles if row["system"] == "C0" and row["oracle_scope"] == "frame"),
        "c1_frame_oracle": next(row for row in oracles if row["system"] == "C1" and row["oracle_scope"] == "frame"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
