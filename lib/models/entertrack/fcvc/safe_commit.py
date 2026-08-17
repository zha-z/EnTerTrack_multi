import hashlib
import json
from typing import Any, Dict

import torch

from .structures import CandidatePair


def _canonical(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        return {
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def state_digest(state: Dict[str, Any]) -> str:
    payload = json.dumps(_canonical(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SafeCommitRuntime:
    def __init__(self, initial_state: Dict[str, Any]):
        self.state = dict(initial_state)
        self.history = []

    def commit(self, local_candidate: Dict[str, Any], reported_candidate: Dict[str, Any]):
        before = state_digest(self.state)
        if "bbox" in local_candidate:
            self.state["bbox"] = local_candidate["bbox"]
        if "crop" in local_candidate:
            self.state["crop"] = local_candidate["crop"]
        if "sender_source" in local_candidate:
            self.state["sender_source"] = local_candidate["sender_source"]
        after = state_digest(self.state)
        record = CandidatePair(local_candidate, reported_candidate,
                               reported_candidate is not local_candidate)
        self.history.append({"before": before, "after": after, "pair": record})
        return record
