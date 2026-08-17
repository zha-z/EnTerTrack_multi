"""Fixed Temporal Gate sidecar and bounded causal history runtime.

The sidecar is deliberately independent from the EnTeR/C3R checkpoint.  It
accepts only the frozen normalized 10-D reliability vector and never stores a
recurrent hidden state between calls.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


TEMPORAL_GATE_INPUT_DIM = 10
TEMPORAL_GATE_HIDDEN_DIM = 16
TEMPORAL_GATE_WINDOW = 8
TEMPORAL_GATE_MAX_GATE = 0.25
TEMPORAL_GATE_PARAMETER_COUNT = 1361


def temporal_utility_to_gate(raw_utility: torch.Tensor) -> torch.Tensor:
    """Frozen v2 mapping with an exact zero gate for non-positive utility."""
    return TEMPORAL_GATE_MAX_GATE * torch.relu(torch.tanh(raw_utility))


class TemporalGate(nn.Module):
    """Frozen 10 -> one-layer GRU(16) -> Linear(1) architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=TEMPORAL_GATE_INPUT_DIM,
            hidden_size=TEMPORAL_GATE_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.output = nn.Linear(TEMPORAL_GATE_HIDDEN_DIM, 1)
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != TEMPORAL_GATE_PARAMETER_COUNT:
            raise AssertionError("Temporal Gate must contain exactly 1,361 parameters")

    @staticmethod
    def normalize_inputs(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != TEMPORAL_GATE_INPUT_DIM:
            raise ValueError("Temporal Gate input must contain exactly 10 values")
        return torch.nan_to_num(
            inputs.detach().to(dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1.0, 1.0)

    def raw_utility(self, history: torch.Tensor,
                    lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return signed raw utility per causal prefix from zero hidden."""
        if history.dim() == 2:
            history = history.unsqueeze(0)
        if history.dim() != 3 or history.shape[-1] != TEMPORAL_GATE_INPUT_DIM:
            raise ValueError("history must have shape [B,L,10] or [L,10]")
        if history.shape[1] < 1 or history.shape[1] > TEMPORAL_GATE_WINDOW:
            raise ValueError("Temporal Gate history length must be within [1,8]")
        history = self.normalize_inputs(history)
        if lengths is None:
            lengths = torch.full(
                (history.shape[0],), history.shape[1], dtype=torch.long,
                device=history.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=history.device)
            if lengths.shape != (history.shape[0],):
                raise ValueError("lengths must have shape [B]")
            if bool(((lengths < 1) | (lengths > history.shape[1])).any().item()):
                raise ValueError("each prefix length must be within the padded sequence")
        packed = pack_padded_sequence(
            history, lengths.detach().cpu(), batch_first=True,
            enforce_sorted=False)
        zero_hidden = history.new_zeros((1, history.shape[0], TEMPORAL_GATE_HIDDEN_DIM))
        _, final_hidden = self.gru(packed, zero_hidden)
        return self.output(final_hidden[-1])

    def logits(self, history: torch.Tensor,
               lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Backward-compatible alias; v2 interprets this value as utility."""
        return self.raw_utility(history, lengths=lengths)

    def forward(self, history: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        utility = self.raw_utility(history, lengths=lengths)
        gate = temporal_utility_to_gate(utility)
        return gate, utility


class TemporalGateRuntime:
    """Detached W=8 histories isolated by (target, receiver, sender)."""

    def __init__(self, model: TemporalGate) -> None:
        if not isinstance(model, TemporalGate):
            raise TypeError("runtime requires a TemporalGate sidecar")
        self.model = model
        self._histories: Dict[Tuple[str, int, int], Deque[torch.Tensor]] = {}
        self._last_frames: Dict[Tuple[str, int, int], int] = {}

    @property
    def keys(self) -> Tuple[Tuple[str, int, int], ...]:
        return tuple(sorted(self._histories))

    def reset(self, target_id: Optional[str] = None,
              receiver_id: Optional[int] = None,
              sender_id: Optional[int] = None) -> None:
        if target_id is None and receiver_id is None and sender_id is None:
            self._histories.clear()
            self._last_frames.clear()
            return
        doomed = [
            key for key in self._histories
            if (target_id is None or key[0] == str(target_id))
            and (receiver_id is None or key[1] == int(receiver_id))
            and (sender_id is None or key[2] == int(sender_id))
        ]
        for key in doomed:
            self._histories.pop(key, None)
            self._last_frames.pop(key, None)

    def mark_gap(self, target_id: str, receiver_id: int, sender_id: int) -> None:
        self.reset(target_id, receiver_id, sender_id)

    def history(self, target_id: str, receiver_id: int,
                sender_id: int) -> Tuple[torch.Tensor, ...]:
        return tuple(self._histories.get(
            (str(target_id), int(receiver_id), int(sender_id)), ()))

    def gate_for(self, target_id: str, receiver_id: int, sender_id: int,
                 frame_id: int, normalized_input: torch.Tensor) -> torch.Tensor:
        key = (str(target_id), int(receiver_id), int(sender_id))
        frame_id = int(frame_id)
        previous = self._last_frames.get(key)
        if previous is not None and frame_id != previous + 1:
            self.reset(*key)
        vector = TemporalGate.normalize_inputs(
            torch.as_tensor(normalized_input)).reshape(TEMPORAL_GATE_INPUT_DIM)
        vector = vector.detach().to(dtype=torch.float32).clone()
        history = self._histories.setdefault(
            key, deque(maxlen=TEMPORAL_GATE_WINDOW))
        history.append(vector)
        self._last_frames[key] = frame_id
        device = next(self.model.parameters()).device
        window = torch.stack(tuple(history), dim=0).to(device=device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            gate, _ = self.model(window)
        self.model.train(was_training)
        gate = gate.reshape(()).detach()
        if not bool(torch.isfinite(gate).item()) or not 0.0 <= float(gate) <= 0.25:
            raise RuntimeError("Temporal Gate emitted an invalid gate")
        return gate


def temporal_gate_optimizer_parameters(model: TemporalGate) -> Iterable[nn.Parameter]:
    """Audit and return the only tensors authorized for optimization."""
    names = tuple(name for name, parameter in model.named_parameters()
                  if parameter.requires_grad)
    if not names or any(
            not (name.startswith("gru.") or name.startswith("output."))
            for name in names):
        raise AssertionError("optimizer may contain only gru.* and output.*")
    count = sum(parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
    if count != TEMPORAL_GATE_PARAMETER_COUNT:
        raise AssertionError("trainable parameter count must be 1,361")
    return tuple(parameter for parameter in model.parameters()
                 if parameter.requires_grad)


def load_temporal_gate_checkpoint(path: str, expected_sha256: str = "",
                                  map_location: str = "cpu") -> TemporalGate:
    """Load a sidecar-only checkpoint and optionally enforce its digest."""
    import hashlib

    if expected_sha256:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != str(expected_sha256).lower():
            raise RuntimeError("Temporal Gate checkpoint digest mismatch")
    payload = torch.load(path, map_location=map_location)
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    model = TemporalGate()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
