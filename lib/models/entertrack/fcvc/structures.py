"""Shared FCVC data structures; definitions live in this module only."""

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class SenderBundle:
    mid_features: torch.Tensor
    high_features: torch.Tensor
    response_map: torch.Tensor
    confidence_uncertainty: torch.Tensor
    target_prototype: torch.Tensor
    position_grid: torch.Tensor
    crop_affine: torch.Tensor
    image_size: torch.Tensor
    local_bbox: torch.Tensor
    view_id: torch.Tensor
    timestamp: torch.Tensor

    def tensors(self) -> Dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def schema(self) -> Dict[str, Dict[str, object]]:
        output = {}
        for name, value in self.tensors().items():
            output[name] = {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
                "device": str(value.device),
                "detached": not value.requires_grad,
                "estimated_bytes": int(value.numel() * value.element_size()),
            }
        return output

    def estimated_bytes(self) -> int:
        return sum(item["estimated_bytes"] for item in self.schema().values())


@dataclass(frozen=True)
class CandidatePair:
    state_output: Dict[str, Any]
    reported_output: Dict[str, Any]
    used_remote: bool


@dataclass
class TapReplayOutput:
    mid_tokens: torch.Tensor
    high_tokens: torch.Tensor
    replay_tokens: torch.Tensor
    final_tokens: torch.Tensor


@dataclass(frozen=True)
class FrameTrackingResult:
    local_candidate: dict
    collaborative_candidate: dict
    state_output: dict
    reported_output: dict
    local_runtime_payload: dict
    local_diagnostics: dict
    collaborative_diagnostics: dict
