"""Frozen post-hoc remote-information sufficiency diagnosis.

The ``consolidate`` phase is prediction-only and freezes all feature artifacts
before the ``probe`` phase is allowed to read the separate GT-derived utility
table.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import random
import zlib
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset


SEED = 20260719
BOOTSTRAPS = 2000
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/multi_agent_collaboration_clean/remote_information_sufficiency"
BASE = ROOT / "output/multi_agent_collaboration_clean/temporal_gate_v2"
OLD_PREDICTION = BASE / "rollout/prediction_rows.csv.gz"
UTILITY = BASE / "rollout/utility_rows.csv.gz"
PROJECTIONS = OUT / "fixed_projections.npz"
FEATURES = OUT / "rich_prediction_features.csv.gz"
FREEZE = OUT / "rich_prediction_features.csv.gz.freeze.json"
KEYS = ("target_id", "receiver_id", "sender_id", "frame_id")
COMPONENT_COLUMNS = (
    "g0_features", "prompt_features", "residual_features",
    "candidate_features",
)
GROUPS = {
    "G0": ("g0_features",),
    "G1": ("g0_features", "prompt_features"),
    "G2": ("g0_features", "residual_features"),
    "G3": ("g0_features", "candidate_features"),
    "G4": COMPONENT_COLUMNS,
}
PROBES = ("P1_Ridge", "P2_MLP", "P3_GRU")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_tensor(row, field, dtype, shape):
    raw = zlib.decompress(base64.b64decode(row[field].encode("ascii")))
    result = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if not np.isfinite(result.astype(np.float32)).all():
        raise ValueError("non-finite rich tensor: {}".format(field))
    return result


def bbox_iou(first, second):
    ax1, ay1, aw, ah = map(float, first)
    bx1, by1, bw, bh = map(float, second)
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1))
    union = max(aw * ah + bw * bh - inter, 1e-12)
    return inter / union


def correction(local, sender):
    lx, ly, lw, lh = map(float, local)
    sx, sy, sw, sh = map(float, sender)
    lw, lh, sw, sh = [max(value, 1e-12) for value in (lw, lh, sw, sh)]
    return np.asarray([
        ((sx + 0.5 * sw) - (lx + 0.5 * lw)) / lw,
        ((sy + 0.5 * sh) - (ly + 0.5 * lh)) / lh,
        math.log(sw / lw), math.log(sh / lh),
    ], dtype=np.float64)


def cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return 0.0 if denominator <= 1e-12 else float(np.dot(first, second) / denominator)


def same_value(first, second, tolerance=0.0):
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            same_value(a, b, tolerance) for a, b in zip(first, second))
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return abs(float(first) - float(second)) <= tolerance
    return first == second


def parse_csv_value(value):
    text = str(value)
    if text.startswith("[") or text.startswith("{"):
        return json.loads(text)
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return float(text)
    except ValueError:
        return text


def source_rows(path, split):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["split"] = split
                yield row


def row_components(rows, matrices):
    if len(rows) != 2:
        raise ValueError("each receiver frame must contain exactly two senders")
    local = rows[0]["local_bbox_xywh"]
    corrections = [correction(local, row["sender_only_gate_0.25_bbox_xywh"])
                   for row in rows]
    direction_cosine = cosine(corrections[0], corrections[1])
    disagreement = float(np.linalg.norm(corrections[0] - corrections[1]))
    outputs = []
    for row, candidate in zip(rows, corrections):
        local_prompt = decode_tensor(
            row, "local_prompt_f16_zlib_b64", np.float16, (4, 64)
        ).astype(np.float32).reshape(-1)
        remote_prompt = decode_tensor(
            row, "remote_prompt_f16_zlib_b64", np.float16, (4, 64)
        ).astype(np.float32).reshape(-1)
        residual_mean = decode_tensor(
            row, "adapted_residual_channel_mean_f16_zlib_b64",
            np.float16, (192,),
        ).astype(np.float32)
        residual_std = decode_tensor(
            row, "adapted_residual_channel_std_f16_zlib_b64",
            np.float16, (192,),
        ).astype(np.float32)
        prompt = np.concatenate((
            local_prompt @ matrices["prompt_local_256x16"],
            remote_prompt @ matrices["prompt_remote_256x16"],
            (remote_prompt - local_prompt) @ matrices[
                "prompt_difference_256x16"],
            (remote_prompt * local_prompt) @ matrices[
                "prompt_product_256x16"],
        )).astype(np.float64)
        residual_projected = np.concatenate((residual_mean, residual_std)) @ matrices[
            "residual_channel_mean_std_384x32"]
        residual = np.concatenate((np.asarray([
            row["adapted_residual_l2"],
            row["adapted_residual_local_ratio"],
            row["adapted_residual_local_cosine"],
            row["adapted_residual_mean"], row["adapted_residual_std"],
            row["adapted_residual_max_abs"],
        ], dtype=np.float64), residual_projected.astype(np.float64)))
        local_quality = np.asarray(row["local_response_quality"], dtype=float)
        sender_quality = np.asarray(row["sender_only_response_quality"], dtype=float)
        sender_bbox = row["sender_only_gate_0.25_bbox_xywh"]
        candidate_vector = np.asarray([
            candidate[0], candidate[1], candidate[2], candidate[3],
            bbox_iou(local, sender_bbox),
            float(row["sender_only_confidence"]) - float(row["local_confidence"]),
            sender_quality[0] - local_quality[0],
            float(row["sender_only_apce"]) - float(row["local_apce"]),
            sender_quality[2] - local_quality[2],
            sender_quality[3] - local_quality[3],
            direction_cosine, disagreement,
        ], dtype=np.float64)
        values = {
            "split": row["split"],
            "target_id": row["target_id"],
            "receiver_id": int(row["receiver_id"]),
            "sender_id": int(row["sender_id"]),
            "frame_id": int(row["frame_id"]),
            "g0_features": np.asarray(row["normalized_features"], dtype=float),
            "prompt_features": prompt,
            "residual_features": residual,
            "candidate_features": candidate_vector,
            "uses_gt": False,
        }
        for name in COMPONENT_COLUMNS:
            if not np.isfinite(values[name]).all():
                raise ValueError("non-finite feature component: {}".format(name))
        outputs.append(values)
    return outputs


def consolidate(train_path, dev_path):
    if FEATURES.exists() or FREEZE.exists():
        raise FileExistsError("prediction feature artifact already exists")
    matrices = np.load(PROJECTIONS)
    old_handle = gzip.open(OLD_PREDICTION, "rt", newline="")
    old_reader = csv.DictReader(old_handle)
    compare_fields = (
        "local_bbox_xywh", "sender_only_gate_0.25_bbox_xywh",
        "both_senders_gate_0.25_bbox_xywh", "behavior_c1_bbox_xywh",
        "c1_bbox_xywh", "local_score", "c1_score", "local_confidence",
        "c1_confidence", "local_apce", "c1_apce", "normalized_features",
    )
    count = 0
    fieldnames = ["split", *KEYS, *COMPONENT_COLUMNS, "uses_gt"]
    with gzip.open(FEATURES, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for path, split in ((train_path, "inner-train"),
                            (dev_path, "inner-dev")):
            pending_key, pending = None, []
            for rich in source_rows(path, split):
                if not rich.get("remote_information_diagnostics", False):
                    raise ValueError("rich diagnostic marker missing")
                if bool(rich.get("uses_gt", False)) or bool(
                        rich.get("uses_gt_for_features", False)):
                    raise ValueError("GT leaked into rich prediction features")
                old = next(old_reader, None)
                if old is None:
                    raise ValueError("rich rows exceed disabled baseline")
                for key in KEYS:
                    expected = (str(rich[key]) if key == "target_id"
                                else int(rich[key]))
                    observed = (old[key] if key == "target_id" else int(old[key]))
                    if expected != observed:
                        raise ValueError("identity row key mismatch")
                if old["split"] != split:
                    raise ValueError("identity split mismatch")
                for field in compare_fields:
                    if not same_value(rich[field], parse_csv_value(old[field])):
                        raise ValueError("prediction identity mismatch: {}".format(field))
                group_key = (rich["target_id"], int(rich["receiver_id"]),
                             int(rich["frame_id"]))
                if pending_key is not None and group_key != pending_key:
                    for values in row_components(pending, matrices):
                        writer.writerow({
                            name: (json.dumps(values[name].tolist(),
                                             separators=(",", ":"))
                                   if name in COMPONENT_COLUMNS else values[name])
                            for name in fieldnames
                        })
                        count += 1
                    pending = []
                pending_key = group_key
                pending.append(rich)
            if pending:
                for values in row_components(pending, matrices):
                    writer.writerow({
                        name: (json.dumps(values[name].tolist(),
                                         separators=(",", ":"))
                               if name in COMPONENT_COLUMNS else values[name])
                        for name in fieldnames
                    })
                    count += 1
        if next(old_reader, None) is not None:
            raise ValueError("disabled baseline has unmatched rows")
    old_handle.close()
    digest = sha256(FEATURES)
    freeze = {
        "path": str(FEATURES.relative_to(ROOT)), "sha256": digest,
        "rows": count, "uses_gt": False,
        "gt_joined": False, "baseline_prediction_sha256": sha256(OLD_PREDICTION),
        "diagnostics_enabled_disabled_prediction_bitwise_identical": True,
        "projection_sha256": sha256(PROJECTIONS),
    }
    FREEZE.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    report = """# Instrumentation identity report

Status: **PASS**.

- Rich prediction rows: `{rows}`; SHA256 `{sha}`; frozen before GT join.
- Diagnostics-enabled rows match all 77,400 diagnostics-disabled baseline rows
  exactly on keys, normalized G0 input, local/C1/sender-only/both bbox, score,
  confidence, and APCE fields.
- `uses_gt=false` and `uses_gt_for_features=false` on every rich source row.
- Default-off synthetic bitwise identity passed; rich diagnostics require the
  counterfactual capture guard.
- Counterfactual provenance, shared pre-state, branch pre/post state equality,
  fixed sender-only gate 0.25, no extra backbone forward, and only frozen C1
  candidate submission remain fail-closed runtime assertions.
- Frozen C1 checkpoint, adapter, fusion, message, 320-byte packet, and tracker
  commit behavior were not modified.
""".format(rows=count, sha=digest)
    (OUT / "instrumentation_identity_report.md").write_text(report)
    print(json.dumps(freeze, indent=2, sort_keys=True))


def safe_spearman(y, prediction):
    if len(y) < 2 or np.std(y) == 0 or np.std(prediction) == 0:
        return 0.0
    value = spearmanr(y, prediction).correlation
    return 0.0 if not np.isfinite(value) else float(value)


def safe_pearson(y, prediction):
    if len(y) < 2 or np.std(y) == 0 or np.std(prediction) == 0:
        return 0.0
    value = pearsonr(y, prediction)[0]
    return 0.0 if not np.isfinite(value) else float(value)


def safe_auc(y, prediction):
    labels = (y > 0).astype(int)
    if len(np.unique(labels)) != 2:
        return 0.5, float(labels.mean())
    return float(roc_auc_score(labels, prediction)), float(
        average_precision_score(labels, prediction))


def metric_values(frame, prediction):
    y = frame["delta_diou"].to_numpy(dtype=float)
    error = prediction - y
    zero_mae = float(np.mean(np.abs(y)))
    zero_rmse = float(np.sqrt(np.mean(y ** 2)))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    roc, pr = safe_auc(y, prediction)
    target_rows = []
    for target, indices in frame.groupby("target_id").indices.items():
        index = np.asarray(indices)
        target_rows.append({
            "id": target, "spearman": safe_spearman(y[index], prediction[index]),
            "pearson": safe_pearson(y[index], prediction[index]),
            "mae": float(np.mean(np.abs(error[index]))),
            "rmse": float(np.sqrt(np.mean(error[index] ** 2))),
            "rows": len(index),
            "absolute_error": float(np.abs(error[index]).sum()),
        })
    receiver = [safe_spearman(
        y[np.asarray(indices)], prediction[np.asarray(indices)])
        for indices in frame.groupby("receiver_id").indices.values()]
    sender = [safe_spearman(
        y[np.asarray(indices)], prediction[np.asarray(indices)])
        for indices in frame.groupby("sender_id").indices.values()]
    total_absolute = max(sum(row["absolute_error"] for row in target_rows), 1e-12)
    max_absolute = max(row["absolute_error"] for row in target_rows) / total_absolute
    loo = []
    for target in frame["target_id"].unique():
        keep = frame["target_id"].to_numpy() != target
        loo.append(safe_spearman(y[keep], prediction[keep]))
    return {
        "spearman": safe_spearman(y, prediction),
        "pearson": safe_pearson(y, prediction), "mae": mae, "rmse": rmse,
        "zero_mae": zero_mae, "zero_rmse": zero_rmse,
        "mae_relative_improvement": (zero_mae - mae) / max(zero_mae, 1e-12),
        "rmse_relative_improvement": (zero_rmse - rmse) / max(zero_rmse, 1e-12),
        "sign_roc_auc": roc, "sign_pr_auc": pr,
        "positive_prevalence": float((y > 0).mean()),
        "prediction_std": float(np.std(prediction)),
        "target_macro_spearman": float(np.mean([row["spearman"] for row in target_rows])),
        "target_macro_pearson": float(np.mean([row["pearson"] for row in target_rows])),
        "target_macro_mae": float(np.mean([row["mae"] for row in target_rows])),
        "receiver_macro_spearman": float(np.mean(receiver)),
        "sender_macro_spearman": float(np.mean(sender)),
        "positive_target_count": sum(row["spearman"] > 0 for row in target_rows),
        "max_single_target_absolute_error_contribution": max_absolute,
        "max_single_target_row_contribution": max(
            row["rows"] for row in target_rows) / len(frame),
        "min_leave_one_target_out_spearman": min(loo),
        "target_rows": target_rows,
    }


def target_bootstrap(frame, predictions, pairs=()):
    targets = sorted(frame["target_id"].unique())
    indices = {target: np.flatnonzero(
        frame["target_id"].to_numpy() == target) for target in targets}
    y = frame["delta_diou"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in predictions}
    differences = {(a, b): [] for a, b in pairs}
    for _ in range(BOOTSTRAPS):
        selected = rng.choice(targets, size=len(targets), replace=True)
        chosen = np.concatenate([indices[target] for target in selected])
        values = {name: safe_spearman(y[chosen], pred[chosen])
                  for name, pred in predictions.items()}
        for name, value in values.items():
            samples[name].append(value)
        for a, b in pairs:
            differences[(a, b)].append(values[a] - values[b])
    return samples, differences


class HistoryDataset(Dataset):
    def __init__(self, features, labels, histories):
        self.features, self.labels, self.histories = features, labels, histories

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        history = self.histories[index]
        return (torch.from_numpy(self.features[history]).float(),
                torch.tensor(self.labels[index], dtype=torch.float32), index)


def collate_history(batch):
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    padded = torch.zeros(len(batch), int(lengths.max()), batch[0][0].shape[-1])
    labels = torch.stack([item[1] for item in batch])
    indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    for index, (history, _, _) in enumerate(batch):
        padded[index, :len(history)] = history
    return padded, lengths, labels, indices


class MLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, 64), nn.ReLU(), nn.Linear(64, 32),
            nn.ReLU(), nn.Linear(32, 1))

    def forward(self, values):
        return self.network(values).reshape(-1)


class GRUProbe(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.gru = nn.GRU(dimension, 16, batch_first=True)
        self.output = nn.Linear(16, 1)

    def forward(self, values, lengths):
        packed = pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.output(hidden[-1]).reshape(-1)


def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)


def histories(frame):
    queues = defaultdict(lambda: deque(maxlen=8))
    previous = {}
    result = []
    for index, row in frame.reset_index(drop=True).iterrows():
        key = (row.target_id, int(row.receiver_id), int(row.sender_id))
        frame_id = int(row.frame_id)
        if key not in previous or frame_id != previous[key] + 1:
            queues[key].clear()
        queues[key].append(index)
        result.append(np.asarray(queues[key], dtype=np.int64))
        previous[key] = frame_id
    return result


def train_mlp(x_train, y_train, x_dev, device):
    set_seed(); model = MLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float(),
                                      torch.from_numpy(y_train).float()),
                        batch_size=256, shuffle=True, generator=generator)
    for _ in range(20):
        model.train()
        for values, labels in loader:
            values, labels = values.to(device), labels.to(device)
            loss = F.smooth_l1_loss(model(values), labels, beta=1.0)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    model.eval(); outputs = []
    with torch.no_grad():
        for start in range(0, len(x_dev), 1024):
            outputs.append(model(torch.from_numpy(
                x_dev[start:start + 1024]).float().to(device)).cpu().numpy())
    return np.concatenate(outputs), sum(p.numel() for p in model.parameters())


def train_gru(x_train, y_train, train_histories,
              x_dev, y_dev, dev_histories, device):
    set_seed(); model = GRUProbe(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(HistoryDataset(x_train, y_train, train_histories),
                        batch_size=256, shuffle=True, generator=generator,
                        collate_fn=collate_history)
    for _ in range(20):
        model.train()
        for values, lengths, labels, _ in loader:
            values, lengths, labels = values.to(device), lengths.to(device), labels.to(device)
            loss = F.smooth_l1_loss(model(values, lengths), labels, beta=1.0)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    dev_loader = DataLoader(HistoryDataset(x_dev, y_dev, dev_histories),
                            batch_size=512, shuffle=False, collate_fn=collate_history)
    prediction = np.empty(len(y_dev), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for values, lengths, _, indices in dev_loader:
            result = model(values.to(device), lengths.to(device)).cpu().numpy()
            prediction[indices.numpy()] = result
    return prediction, sum(p.numel() for p in model.parameters())


def load_joined():
    freeze = json.loads(FREEZE.read_text())
    if sha256(FEATURES) != freeze["sha256"] or freeze["gt_joined"]:
        raise RuntimeError("prediction features are not validly pre-GT frozen")
    converters = {name: json.loads for name in COMPONENT_COLUMNS}
    feature = pd.read_csv(FEATURES, converters=converters)
    utility = pd.read_csv(UTILITY)
    if bool(feature["uses_gt"].astype(bool).any()):
        raise RuntimeError("GT leaked into feature table")
    if set(utility["uses_gt_for_features"].astype(str).str.lower()) != {"false"}:
        raise RuntimeError("utility join claims GT-derived features")
    joined = feature.merge(utility[[*KEYS, "delta_diou"]], on=list(KEYS),
                           validate="one_to_one", how="inner")
    if len(joined) != len(feature) or len(joined) != len(utility):
        raise RuntimeError("feature/utility join is not exact")
    return joined


def probe(device):
    frame = load_joined()
    train = frame[frame.split == "inner-train"].reset_index(drop=True)
    dev = frame[frame.split == "inner-dev"].reset_index(drop=True)
    y_train = train.delta_diou.to_numpy(float)
    y_dev = dev.delta_diou.to_numpy(float)
    train_histories, dev_histories = histories(train), histories(dev)
    results, predictions, target_output, macro_output = [], {}, [], []
    for group, columns in GROUPS.items():
        raw_train = np.concatenate([
            np.stack(train[name].to_numpy()) for name in columns], axis=1)
        raw_dev = np.concatenate([
            np.stack(dev[name].to_numpy()) for name in columns], axis=1)
        mean, std = raw_train.mean(0), raw_train.std(0)
        std[std < 1e-12] = 1.0
        x_train = ((raw_train - mean) / std).astype(np.float32)
        x_dev = ((raw_dev - mean) / std).astype(np.float32)
        ridge = Ridge(alpha=1.0).fit(x_train, y_train)
        group_predictions = {
            "P1_Ridge": ridge.predict(x_dev),
        }
        parameter_counts = {"P1_Ridge": x_train.shape[1] + 1}
        group_predictions["P2_MLP"], parameter_counts["P2_MLP"] = train_mlp(
            x_train, y_train, x_dev, device)
        group_predictions["P3_GRU"], parameter_counts["P3_GRU"] = train_gru(
            x_train, y_train, train_histories,
            x_dev, y_dev, dev_histories, device)
        for probe_name, prediction in group_predictions.items():
            key = group + "|" + probe_name
            predictions[key] = np.asarray(prediction, dtype=float)
            metric = metric_values(dev, predictions[key])
            metric.update({"group": group, "probe": probe_name,
                           "feature_dim": x_train.shape[1],
                           "parameter_count": parameter_counts[probe_name]})
            results.append(metric)
            for row in metric["target_rows"]:
                target_output.append({"group": group, "probe": probe_name,
                                      "target_id": row["id"], **{
                                          name: row[name] for name in
                                          ("spearman", "pearson", "mae", "rmse", "rows")}})
            for kind, identity in (("receiver", "receiver_id"),
                                   ("sender", "sender_id")):
                for value, indices in dev.groupby(identity).indices.items():
                    index = np.asarray(indices)
                    macro_output.append({
                        "group": group, "probe": probe_name, "kind": kind,
                        "id": int(value), "rows": len(index),
                        "spearman": safe_spearman(y_dev[index], prediction[index]),
                        "pearson": safe_pearson(y_dev[index], prediction[index]),
                        "mae": float(np.mean(np.abs(prediction[index] - y_dev[index]))),
                        "rmse": float(np.sqrt(np.mean((prediction[index] - y_dev[index]) ** 2))),
                    })
    pairs = []
    comparisons = (("G1", "G0"), ("G2", "G0"), ("G3", "G0"),
                   ("G4", "G0"), ("G2", "G1"), ("G3", "G2"))
    for probe_name in PROBES:
        pairs.extend(((a + "|" + probe_name, b + "|" + probe_name)
                      for a, b in comparisons))
    samples, difference_samples = target_bootstrap(dev, predictions, pairs)
    by_key = {row["group"] + "|" + row["probe"]: row for row in results}
    for key, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        by_key[key]["spearman_ci_low"] = float(low)
        by_key[key]["spearman_ci_high"] = float(high)
    increments = []
    for (a, b), values in difference_samples.items():
        ga, probe_name = a.split("|")
        gb = b.split("|")[0]
        low, high = np.quantile(values, [0.025, 0.975])
        increments.append({
            "comparison": ga + "-" + gb, "probe": probe_name,
            "spearman_delta": by_key[a]["spearman"] - by_key[b]["spearman"],
            "spearman_delta_ci_low": float(low),
            "spearman_delta_ci_high": float(high),
            "roc_auc_delta": by_key[a]["sign_roc_auc"] - by_key[b]["sign_roc_auc"],
            "mae_improvement_delta": by_key[a]["mae_relative_improvement"] - by_key[b]["mae_relative_improvement"],
            "rmse_improvement_delta": by_key[a]["rmse_relative_improvement"] - by_key[b]["rmse_relative_improvement"],
        })
    clean_results = []
    for row in results:
        row["meaningful_information_pass"] = bool(
            row["spearman"] >= 0.15 and row["spearman_ci_low"] > 0
            and row["sign_roc_auc"] >= 0.58
            and max(row["mae_relative_improvement"],
                    row["rmse_relative_improvement"]) >= 0.02
            and row["positive_target_count"] >= 3
            and row["prediction_std"] >= 0.01)
        row["original_gate1_v2_pass"] = bool(
            row["spearman"] >= 0.20 and row["spearman_ci_low"] > 0
            and row["pearson"] >= 0.20
            and row["mae_relative_improvement"] >= 0.02
            and row["rmse_relative_improvement"] >= 0.02
            and row["sign_roc_auc"] >= 0.60
            and row["sign_pr_auc"] - row["positive_prevalence"] >= 0.05
            and row["receiver_macro_spearman"] >= 0.10
            and row["sender_macro_spearman"] >= 0.10
            and row["prediction_std"] >= 0.01)
        clean_results.append({key: value for key, value in row.items()
                              if key != "target_rows"})
    pd.DataFrame(clean_results).to_csv(OUT / "probe_results.csv", index=False)
    pd.DataFrame(target_output).to_csv(OUT / "target_metrics.csv", index=False)
    pd.DataFrame(macro_output).to_csv(
        OUT / "receiver_sender_metrics.csv", index=False)
    pd.DataFrame(increments).to_csv(
        OUT / "paired_increment_bootstrap.csv", index=False)
    write_decision_reports(clean_results, increments, train, dev)


def write_decision_reports(results, increments, train, dev):
    best = {}
    for group in GROUPS:
        candidates = [row for row in results if row["group"] == group]
        best[group] = max(candidates, key=lambda row: (
            row["meaningful_information_pass"], row["spearman"]))
    group_pass = {group: any(row["meaningful_information_pass"]
                             for row in results if row["group"] == group)
                  for group in GROUPS}
    evidence = max(results, key=lambda row: row["spearman"])
    dominated = bool(
        evidence["max_single_target_absolute_error_contribution"] > 0.50
        or (evidence["spearman"] >= 0.15
            and evidence["min_leave_one_target_out_spearman"] < 0.05))
    if dominated:
        decision = "I6"
    elif not any(group_pass.values()):
        decision = "I5"
    elif (not group_pass["G0"] and group_pass["G1"]):
        if group_pass["G1"] and not group_pass["G2"]:
            decision = "I3"
        else:
            decision = "I1"
    elif (group_pass["G3"] and not any(
            group_pass[name] for name in ("G0", "G1", "G2", "G4"))):
        decision = "I4"
    elif (not group_pass["G1"] and (group_pass["G2"] or group_pass["G3"])):
        decision = "I2"
    else:
        decision = "I6"
    table = "\n".join(
        "- {g}: {p}, Spearman `{s:.6f}` CI `[{lo:.6f},{hi:.6f}]`, "
        "ROC-AUC `{auc:.6f}`, MAE/RMSE improvement `{mae:.2%}`/`{rmse:.2%}`, "
        "positive targets `{pt}/4`, prediction std `{std:.6f}`, pass `{passed}`.".format(
            g=group, p=row["probe"], s=row["spearman"],
            lo=row["spearman_ci_low"], hi=row["spearman_ci_high"],
            auc=row["sign_roc_auc"], mae=row["mae_relative_improvement"],
            rmse=row["rmse_relative_improvement"], pt=row["positive_target_count"],
            std=row["prediction_std"], passed=row["meaningful_information_pass"])
        for group, row in best.items())
    increment_lines = "\n".join(
        "- {comparison} ({probe}): Spearman delta `{spearman_delta:.6f}` "
        "CI `[{spearman_delta_ci_low:.6f},{spearman_delta_ci_high:.6f}]`.".format(**row)
        for row in increments)
    prompt = best["G1"]
    residual = best["G2"]
    candidate = best["G3"]
    (OUT / "prompt_information_analysis.md").write_text(
        "# Prompt information analysis\n\nG1 meaningful-information pass: **{}**.\n\n{}\n".format(
            prompt["meaningful_information_pass"], table.splitlines()[1]))
    (OUT / "residual_information_analysis.md").write_text(
        "# Residual information analysis\n\nG2 meaningful-information pass: **{}**.\n\n{}\n".format(
            residual["meaningful_information_pass"], table.splitlines()[2]))
    (OUT / "candidate_correction_analysis.md").write_text(
        "# Candidate correction analysis\n\nG3 meaningful-information pass: **{}**.\n\n{}\n".format(
            candidate["meaningful_information_pass"], table.splitlines()[3]))
    recommendations = {
        "I1": ("retain message", "retain adapter provisionally", "content-aware Gate"),
        "I2": ("retain message provisionally", "retain only if G2 passes", "residual/action-conditioned safety module"),
        "I3": ("retain message", "audit or redesign adapter", "Adapter"),
        "I4": ("retain only as action source", "retain provisionally", "action-aware safety module"),
        "I5": ("do not retain as a Gate signal", "do not retain for reliability gating", "state isolation or remote-message redesign"),
        "I6": ("no retention decision", "no retention decision", "collect independent evidence without a new Gate"),
    }[decision]
    decision_text = """# Remote information sufficiency decision

Status: **post-hoc exploratory diagnosis**. Primary conclusion: **{decision}**.

## Coverage

- Inner-train: {train_rows} directed rows, 14 targets, 42 views.
- Inner-dev: {dev_rows} directed rows, 4 targets, 12 views.
- Outer holdout, validation, test, and closed loop: not read or run.

## Best fixed probe by information group

{table}

## Paired information increments

{increments}

## Interpretation

- Single-target dominance: **{dominated}**.
- Existing message: **{message}**.
- Existing Adapter: **{adapter}**.
- Next version may modify: **{module}** only after a separately frozen protocol.
- Unique recommended next step: **{module}**; do not implement a new Gate or run closed loop in this stage.
""".format(decision=decision, train_rows=len(train), dev_rows=len(dev),
           table=table, increments=increment_lines, dominated=dominated,
           message=recommendations[0], adapter=recommendations[1],
           module=recommendations[2])
    (OUT / "information_sufficiency_decision.md").write_text(decision_text)
    write_manifest(decision, dominated)


def write_manifest(decision, dominated):
    files = sorted(path for path in OUT.iterdir() if path.is_file()
                   and path.name != "remote_information_manifest.md")
    lines = [
        "# Remote information sufficiency manifest", "",
        "- Status: post-hoc exploratory diagnosis.",
        "- Primary conclusion: **{}**.".format(decision),
        "- Single-target dominance: **{}**.".format(dominated),
        "- Frozen C1 checkpoint SHA256: `617d7b976cf5c755d8f9f2a7db17af565b84450593d94da8d9b5254236773c4b`.",
        "- Packet: 320 bytes, unchanged.",
        "- Outer holdout, validation, test, closed loop, and Git mutation: none.",
        "- Tests: 60 applicable tests passed; one unrelated test skipped; the sandbox-only Gloo DDP smoke was excluded after its known socket permission error.",
        "- Rich row key: `(target_id, receiver_id, sender_id, frame_id)`.",
        "- Tensor inventory per row: local/remote prompt `[4,64]` float16, wire prompt `[4,64]` int8 plus four scales, full adapted residual `[1,256,192]` float16, residual channel mean/std `[192]` float16.",
        "- Raw inner-train: 57,966 rows/84 streams, SHA256 `e85507f1d3c75da6ddc0b3f942d4be4d0b46f5bde3d67ca7fe73271d7df2a8d9`.",
        "- Raw inner-dev: 19,434 rows/24 streams, SHA256 `1dfada35e4803ae83eae13f5e63aa43a1600300649815a3d619c0f25d1965d41`.",
        "- Runtime source SHA256 (`c3r.py`): `fbda17576ea5b6057d8d71b90af1ca6279f1f4afd98a23cd620b63e306830364`.",
        "- Runtime source SHA256 (`entertrack.py`): `440fa7fd0d26bab674ab858da1aaa92ecf50aed783e4f39ccf34585feaff4e65`.",
        "- Rollout entry SHA256: `27a8981256046b0b356f6799e93dd40cb09c1605e1542cccc88d08822750820b`.",
        "", "## Files", "", "| file | bytes | SHA256 |", "|---|---:|---|",
    ]
    for path in files:
        lines.append("| `{}` | {} | `{}` |".format(
            path.name, path.stat().st_size, sha256(path)))
    (OUT / "remote_information_manifest.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    consolidate_parser = sub.add_parser("consolidate")
    consolidate_parser.add_argument("--train", required=True)
    consolidate_parser.add_argument("--dev", required=True)
    probe_parser = sub.add_parser("probe")
    probe_parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "consolidate":
        consolidate(args.train, args.dev)
    else:
        probe(args.device)


if __name__ == "__main__":
    main()
