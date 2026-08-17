"""Leakage-guarded contiguous-window dataset for the Temporal Gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from lib.models.entertrack.temporal_gate import (
    TEMPORAL_GATE_INPUT_DIM,
    TEMPORAL_GATE_WINDOW,
    TemporalGate,
)


TARGET_PATTERN = re.compile(r"^(md\d{4})(?:-([123]))?$")
IDENTITY_FIELDS = frozenset(("target_id", "receiver_id", "sender_id", "frame_id"))
FORBIDDEN_FEATURE_TOKENS = (
    "gt", "ground_truth", "visibility", "visible", "iou", "target_id",
    "receiver_id", "sender_id", "view_id", "frame_id", "e0", "label",
    "future", "behavior",
)


def canonical_target(sequence_or_target: str) -> str:
    match = TARGET_PATTERN.fullmatch(str(sequence_or_target).strip())
    if match is None:
        raise ValueError("invalid fold target/view id: {!r}".format(sequence_or_target))
    return match.group(1)


def read_id_file(path: str) -> List[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_inner_split(train_ids: Sequence[str], fold_id: int,
                      protocol_version: str = "v2", dev_targets: int = 4):
    targets = sorted({canonical_target(value) for value in train_ids})
    if len(targets) <= int(dev_targets):
        raise ValueError("outer-training list is too small for the frozen split")
    salt = "temporal-gate-{}|fold{}|".format(
        str(protocol_version), int(fold_id))
    ranked = sorted(
        targets,
        key=lambda target: hashlib.sha256(
            (salt + target).encode("utf-8")).hexdigest(),
    )
    dev = set(ranked[:int(dev_targets)])
    inner_train = [value for value in train_ids if canonical_target(value) not in dev]
    inner_dev = [value for value in train_ids if canonical_target(value) in dev]
    return inner_train, inner_dev, ranked


def build_fold0_inner_split(train_ids: Sequence[str], dev_targets: int = 4):
    return build_inner_split(
        train_ids, fold_id=0, protocol_version="v1",
        dev_targets=dev_targets)


def build_fold1_inner_split(train_ids: Sequence[str], dev_targets: int = 4):
    """Frozen v2 split using target IDs only and SHA256 fixed ordering."""
    return build_inner_split(
        train_ids, fold_id=1, protocol_version="v2",
        dev_targets=dev_targets)


def audit_split(inner_train: Sequence[str], inner_dev: Sequence[str],
                outer_holdout: Sequence[str]) -> Dict[str, object]:
    train_targets = {canonical_target(value) for value in inner_train}
    dev_targets = {canonical_target(value) for value in inner_dev}
    holdout_targets = {canonical_target(value) for value in outer_holdout}
    if train_targets & dev_targets:
        raise AssertionError("inner-train and inner-dev targets overlap")
    if (train_targets | dev_targets) & holdout_targets:
        raise AssertionError("outer holdout leaked into the inner split")
    train_views = set(inner_train)
    dev_views = set(inner_dev)
    holdout_views = set(outer_holdout)
    if train_views & dev_views or train_views & holdout_views or dev_views & holdout_views:
        raise AssertionError("view-level split overlap detected")
    for target, population in (
            (target, inner_train) for target in train_targets):
        views = {value for value in population if canonical_target(value) == target}
        if views != {target + "-1", target + "-2", target + "-3"}:
            raise AssertionError("all A/B/C views must remain grouped")
    for target in dev_targets:
        views = {value for value in inner_dev if canonical_target(value) == target}
        if views != {target + "-1", target + "-2", target + "-3"}:
            raise AssertionError("all A/B/C views must remain grouped")
    return {
        "inner_train_targets": sorted(train_targets),
        "inner_dev_targets": sorted(dev_targets),
        "outer_holdout_targets": sorted(holdout_targets),
        "target_disjoint": True,
        "view_disjoint": True,
    }


def _assert_feature_schema(row: Mapping[str, object]) -> torch.Tensor:
    declared = row.get("model_input_fields", ())
    if declared and tuple(declared) != ("normalized_features",):
        raise ValueError("model input schema may contain only normalized_features")
    for name in declared:
        lower = str(name).lower()
        if any(token in lower for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError("forbidden model input field: {}".format(name))
    features = torch.as_tensor(row["normalized_features"])
    if features.shape != (TEMPORAL_GATE_INPUT_DIM,):
        raise ValueError("normalized_features must have shape [10]")
    features = TemporalGate.normalize_inputs(features).reshape(TEMPORAL_GATE_INPUT_DIM)
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("Temporal Gate input contains non-finite values")
    return features.detach().to(dtype=torch.float32).clone()


def assert_prediction_row(row: Mapping[str, object]) -> None:
    if bool(row.get("uses_gt", False)):
        raise ValueError("prediction table may not use GT")
    forbidden = {
        "gt_bbox", "ground_truth", "visibility", "target_visible", "iou",
        "delta_iou", "label", "label_status",
    }
    present = forbidden.intersection(str(key).lower() for key in row)
    if present:
        raise ValueError("prediction row contains GT/label fields: {}".format(
            sorted(present)))
    _assert_feature_schema(row)


class C3RTemporalGateDataset(Dataset):
    """One valid labeled endpoint per causal contiguous directed-stream window."""

    def __init__(self, rows: Sequence[Mapping[str, object]], window: int = 8,
                 allowed_targets: Iterable[str] = ()) -> None:
        if int(window) != TEMPORAL_GATE_WINDOW:
            raise ValueError("Temporal Gate window is frozen at W=8")
        allowed = {canonical_target(value) for value in allowed_targets}
        streams = defaultdict(list)
        for source in rows:
            row = dict(source)
            target = canonical_target(row["target_id"])
            if allowed and target not in allowed:
                raise ValueError("row target is not authorized by this split")
            if bool(row.get("uses_gt_for_features", False)):
                raise ValueError("GT may not be used for Temporal Gate features")
            features = _assert_feature_schema(row)
            delta_diou = float(row["delta_diou"])
            if not torch.isfinite(torch.tensor(delta_diou)):
                raise ValueError("delta_diou must be finite")
            if not -2.0 <= delta_diou <= 2.0:
                raise ValueError("delta_diou must be within [-2,2]")
            key = (target, int(row["receiver_id"]), int(row["sender_id"]))
            streams[key].append((
                int(row["frame_id"]), features, delta_diou, row))
        self.windows = []
        for key, values in streams.items():
            values.sort(key=lambda item: item[0])
            if len({item[0] for item in values}) != len(values):
                raise ValueError("duplicate frame in directed stream")
            prefix = []
            previous = None
            for frame, features, delta_diou, row in values:
                if previous is not None and frame <= previous:
                    raise ValueError("stream frames must be strictly increasing")
                if previous is None or frame != previous + 1 or not bool(
                        row.get("packet_accepted", True)):
                    prefix = []
                if not bool(row.get("packet_accepted", True)):
                    previous = frame
                    continue
                prefix.append(features)
                prefix = prefix[-TEMPORAL_GATE_WINDOW:]
                if str(row.get("label_status", "valid")) != "valid":
                    raise ValueError(
                        "v2 requires one continuous delta_diou for every row")
                self.windows.append({
                    "history": torch.stack(prefix).detach().clone(),
                    "delta_diou": torch.tensor(
                        delta_diou, dtype=torch.float32),
                    "target_id": key[0],
                    "receiver_id": key[1],
                    "sender_id": key[2],
                    "frame_id": frame,
                })
                previous = frame

    @classmethod
    def from_jsonl(cls, path: str, **kwargs):
        rows = [json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        return cls(rows, **kwargs)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        return self.windows[index]


def collate_temporal_gate(batch: Sequence[Mapping[str, object]]):
    lengths = torch.tensor([item["history"].shape[0] for item in batch],
                           dtype=torch.long)
    padded = torch.zeros(
        len(batch), int(lengths.max().item()), TEMPORAL_GATE_INPUT_DIM,
        dtype=torch.float32)
    for index, item in enumerate(batch):
        padded[index, :item["history"].shape[0]] = item["history"]
    return {
        "history": padded,
        "lengths": lengths,
        "delta_diou": torch.stack([
            item["delta_diou"] for item in batch]),
        "metadata": [
            {name: item[name] for name in IDENTITY_FIELDS}
            for item in batch
        ],
    }
