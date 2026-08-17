"""Standalone, sidecar-only Temporal Gate training entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from lib.models.entertrack.temporal_gate import (
    TEMPORAL_GATE_PARAMETER_COUNT,
    TemporalGate,
    temporal_gate_optimizer_parameters,
)
from lib.train.dataset.c3r_temporal_gate import (
    C3RTemporalGateDataset,
    collate_temporal_gate,
    read_id_file,
)


SEED = 20260719


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


SMOOTH_L1_BETA = 1.0


def smooth_l1_utility_loss(raw_utility: torch.Tensor,
                           delta_diou: torch.Tensor) -> torch.Tensor:
    """Frozen v2 regression objective; beta is not a search parameter."""
    return F.smooth_l1_loss(
        raw_utility.reshape(-1),
        delta_diou.to(dtype=torch.float32).reshape(-1),
        beta=SMOOTH_L1_BETA,
        reduction="mean",
    )


def build_optimizer(model: TemporalGate) -> torch.optim.Optimizer:
    parameters = temporal_gate_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    authorized = {id(parameter) for parameter in parameters}
    observed = {id(parameter) for group in optimizer.param_groups
                for parameter in group["params"]}
    if observed != authorized:
        raise AssertionError("optimizer membership differs from sidecar parameters")
    return optimizer


def freeze_audit(model: TemporalGate,
                 optimizer: torch.optim.Optimizer) -> Dict[str, object]:
    rows = []
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not (name.startswith("gru.") or name.startswith("output.")):
                raise AssertionError("non-sidecar trainable tensor detected")
            raw = parameter.detach().cpu().contiguous().numpy().tobytes()
            digest.update(name.encode("utf-8") + raw)
            rows.append({"name": name, "shape": list(parameter.shape),
                         "count": parameter.numel()})
    count = sum(row["count"] for row in rows)
    if count != TEMPORAL_GATE_PARAMETER_COUNT:
        raise AssertionError("Temporal Gate trainable count is not 1,361")
    authorized = {id(parameter) for parameter in model.parameters()
                  if parameter.requires_grad}
    observed = {id(parameter) for group in optimizer.param_groups
                for parameter in group["params"]}
    if observed != authorized:
        raise AssertionError("optimizer contains unauthorized tensors")
    return {"parameters": rows, "count": count,
            "tensor_sha256": digest.hexdigest(), "passed": True}


def _epoch(model, loader, device, optimizer=None) -> float:
    all_utilities = []
    all_targets = []
    model.train(optimizer is not None)
    for batch in loader:
        history = batch["history"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["delta_diou"].to(device)
        _, raw_utility = model(history, lengths)
        loss = smooth_l1_utility_loss(raw_utility, targets)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite Temporal Gate loss")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not math.isfinite(float(norm)) or float(norm) <= 0.0:
                raise RuntimeError("Temporal Gate gradient must be finite and nonzero")
            optimizer.step()
        all_utilities.append(raw_utility.detach().cpu().reshape(-1))
        all_targets.append(targets.detach().cpu().reshape(-1))
    if not all_utilities:
        raise RuntimeError("empty Temporal Gate loader")
    return float(smooth_l1_utility_loss(
        torch.cat(all_utilities), torch.cat(all_targets)).item())


def _inference_vector(model, loader, device):
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            gate, _ = model(batch["history"].to(device),
                            batch["lengths"].to(device))
            values.append(gate.detach().cpu().reshape(-1))
    return torch.cat(values)


def train(train_jsonl: str, dev_jsonl: str, inner_train_ids: str,
          inner_dev_ids: str, base_checkpoint: str, output: str,
          device: str = "cpu") -> Dict[str, object]:
    set_deterministic_seed()
    base_before = file_digest(base_checkpoint)
    train_set = C3RTemporalGateDataset.from_jsonl(
        train_jsonl, allowed_targets=read_id_file(inner_train_ids))
    dev_set = C3RTemporalGateDataset.from_jsonl(
        dev_jsonl, allowed_targets=read_id_file(inner_dev_ids))
    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True, collate_fn=collate_temporal_gate,
        generator=torch.Generator().manual_seed(SEED))
    dev_loader = DataLoader(
        dev_set, batch_size=256, shuffle=False, collate_fn=collate_temporal_gate)
    model = TemporalGate().to(device)
    optimizer = build_optimizer(model)
    audit = freeze_audit(model, optimizer)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    epochs = []
    for epoch in range(1, 21):
        train_loss = _epoch(model, train_loader, device, optimizer=optimizer)
        with torch.no_grad():
            dev_loss = _epoch(model, dev_loader, device)
        epochs.append({"epoch": epoch, "train_smooth_l1": train_loss,
                       "dev_smooth_l1": dev_loss})
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break
    if best_state is None:
        raise RuntimeError("no finite Temporal Gate checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    replay_a = _inference_vector(model, dev_loader, device)
    replay_b = _inference_vector(model, dev_loader, device)
    replay_max_abs_diff = float((replay_a - replay_b).abs().max().item())
    if replay_max_abs_diff > 1e-6:
        raise RuntimeError("deterministic replay exceeded 1e-6")
    base_after = file_digest(base_checkpoint)
    if base_after != base_before:
        raise RuntimeError("base checkpoint digest changed during sidecar training")
    payload = {
        "state_dict": best_state,
        "model_spec": {"input": 10, "hidden": 16, "layers": 1,
                       "window": 8, "max_gate": 0.25, "parameters": 1361,
                       "output": "raw_utility",
                       "gate_mapping": "0.25*relu(tanh(raw_utility))"},
        "supervision": {"target": "delta_diou", "loss": "SmoothL1Loss",
                        "beta": SMOOTH_L1_BETA},
        "selected_epoch": best_epoch,
        "seed": SEED,
        "base_checkpoint_sha256": base_before,
        "data_digests": {
            "train_jsonl": file_digest(train_jsonl),
            "dev_jsonl": file_digest(dev_jsonl),
            "inner_train_ids": file_digest(inner_train_ids),
            "inner_dev_ids": file_digest(inner_dev_ids),
        },
        "versions": {"python": sys.version.split()[0],
                     "torch": torch.__version__, "numpy": np.__version__},
        "deterministic_replay_max_abs_diff": replay_max_abs_diff,
        "freeze_audit": audit,
        "epochs": epochs,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(str(output_path))
    torch.save(payload, str(output_path))
    return {"checkpoint": str(output_path),
            "checkpoint_sha256": file_digest(str(output_path)),
            "selected_epoch": best_epoch,
            "dev_smooth_l1": best_loss, "base_checkpoint_unchanged": True,
            "deterministic_replay_max_abs_diff": replay_max_abs_diff,
            "freeze_audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--inner-train-ids", required=True)
    parser.add_argument("--inner-dev-ids", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(train(**vars(args)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
