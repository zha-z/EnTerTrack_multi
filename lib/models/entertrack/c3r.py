"""C3R compact, confidence-constrained residual collaboration.

The module is intentionally transport-agnostic.  It implements the frozen
320-byte application packet and a post-local-head collaboration attachment;
it never receives images, templates, annotations, or ground-truth state.
"""

from __future__ import annotations

import base64
import copy
import math
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


C3R_MAGIC = b"C3R1"
C3R_VERSION = 1
C3R_PACKET_BYTES = 320
C3R_PAYLOAD_BYTES = 288
C3R_STATE_BYTES = 24
C3R_SCALE_BYTES = 8
C3R_PROMPT_BYTES = 256
C3R_HEADER_BYTES = 32
C3R_PROMPT_SHAPE = (4, 64)
C3R_RELIABILITY_INPUT_NAMES = (
    "local_response_peak",
    "local_apce_normalized",
    "local_one_minus_entropy",
    "local_top1_top2_margin",
    "remote_response_peak",
    "remote_apce_normalized",
    "remote_one_minus_entropy",
    "remote_top1_top2_margin",
    "local_remote_prompt_cosine",
    "message_age_fraction_of_max",
)


def _encode_diagnostic_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> str:
    """Losslessly encode a detached CPU tensor for default-off diagnostics."""
    raw = tensor.detach().to(device="cpu", dtype=dtype).contiguous().numpy().tobytes()
    return base64.b64encode(zlib.compress(raw, level=6)).decode("ascii")


class PacketValidationError(ValueError):
    """Raised when a wire packet violates the frozen v1 contract."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


@dataclass
class C3RMessage:
    sender_id: int
    sequence_hash: int
    frame_id: int
    timestamp_ms: int
    bbox: torch.Tensor
    bbox_delta: torch.Tensor
    quality: torch.Tensor
    scales: torch.Tensor
    quantized_prompt: torch.Tensor
    prompt: torch.Tensor
    valid: bool = True
    construction_label: int = 1
    fault: str = "clean"
    wire_bytes: Optional[bytes] = None

    def detached_copy(self, detach: bool = True) -> "C3RMessage":
        result = copy.copy(self)
        for name in ("bbox", "bbox_delta", "quality", "scales",
                     "quantized_prompt", "prompt"):
            value = getattr(self, name)
            source = value.detach() if detach else value
            setattr(result, name, source.clone())
        result.wire_bytes = None if self.wire_bytes is None else bytes(self.wire_bytes)
        return result


@dataclass
class PacketDisposition:
    sender_id: Optional[int]
    accepted: bool
    reason: str
    serialized_bytes: int


@dataclass
class MessageAccounting:
    sent_packets: int = 0
    received_packets: int = 0
    accepted_packets: int = 0
    processed_frames: int = 0
    transport_bytes_received: int = 0
    serialized_bytes_sent: int = 0
    serialized_bytes_received: int = 0
    sent_bytes_per_frame: List[int] = field(default_factory=list)
    received_bytes_per_frame: List[int] = field(default_factory=list)
    accepted_bytes_per_frame: List[int] = field(default_factory=list)
    received_bytes_by_peer: Dict[int, int] = field(default_factory=dict)

    @staticmethod
    def field_bytes() -> Dict[str, int]:
        return {
            "header": C3R_HEADER_BYTES,
            "predicted_state_and_quality": C3R_STATE_BYTES,
            "quantization_scales": C3R_SCALE_BYTES,
            "prompt_int8": C3R_PROMPT_BYTES,
            "total": C3R_PACKET_BYTES,
        }

    @staticmethod
    def theoretical_prompt_tensor_bytes() -> int:
        return C3R_PROMPT_SHAPE[0] * C3R_PROMPT_SHAPE[1]

    def record_frame(self, sent: int = 0, received: int = 0,
                     accepted: int = 0,
                     received_by_peer: Optional[Mapping[int, int]] = None) -> None:
        self.processed_frames += 1
        self.sent_packets += int(sent)
        self.received_packets += int(received)
        self.accepted_packets += int(accepted)
        self.serialized_bytes_sent += int(sent) * C3R_PACKET_BYTES
        self.serialized_bytes_received += int(received) * C3R_PACKET_BYTES
        self.transport_bytes_received += int(received) * C3R_PACKET_BYTES
        self.sent_bytes_per_frame.append(int(sent) * C3R_PACKET_BYTES)
        self.received_bytes_per_frame.append(int(received) * C3R_PACKET_BYTES)
        self.accepted_bytes_per_frame.append(int(accepted) * C3R_PACKET_BYTES)
        if received_by_peer is not None:
            for peer_id, packet_count in received_by_peer.items():
                self.received_bytes_by_peer[int(peer_id)] = (
                    self.received_bytes_by_peer.get(int(peer_id), 0)
                    + int(packet_count) * C3R_PACKET_BYTES
                )

    @staticmethod
    def _p90(values: Sequence[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        index = max(0, int(math.ceil(0.90 * len(ordered))) - 1)
        return ordered[index]

    def report(self, peers: int = 2, broadcast: bool = True) -> Dict[str, float]:
        frames = max(self.processed_frames, 1)
        application_unicast_tx = self.serialized_bytes_sent
        if broadcast:
            application_transmitted = self.serialized_bytes_sent
        else:
            application_transmitted = self.serialized_bytes_sent * max(int(peers), 0)
        return {
            **{key + "_bytes": value for key, value in self.field_bytes().items()},
            "theoretical_prompt_tensor_bytes": self.theoretical_prompt_tensor_bytes(),
            "serialized_packet_bytes": C3R_PACKET_BYTES,
            "processed_frames": self.processed_frames,
            "sent_packets": self.sent_packets,
            "received_packets": self.received_packets,
            "accepted_packets": self.accepted_packets,
            "send_frequency_per_frame": self.sent_packets / float(frames),
            "receive_frequency_per_frame": self.received_packets / float(frames),
            "accepted_frequency_per_frame": self.accepted_packets / float(frames),
            "serialized_bytes_sent": self.serialized_bytes_sent,
            "serialized_bytes_received": self.serialized_bytes_received,
            "application_transmitted_bytes": application_transmitted,
            "unicast_replication_bytes": application_unicast_tx * max(int(peers), 0),
            "mean_sent_bytes_per_frame": self.serialized_bytes_sent / float(frames),
            "mean_received_bytes_per_frame": self.serialized_bytes_received / float(frames),
            "p90_sent_bytes_per_frame": self._p90(self.sent_bytes_per_frame),
            "p90_received_bytes_per_frame": self._p90(self.received_bytes_per_frame),
            "p90_accepted_bytes_per_frame": self._p90(self.accepted_bytes_per_frame),
            "received_bytes_by_peer": dict(sorted(self.received_bytes_by_peer.items())),
            "broadcast_accounting": float(bool(broadcast)),
        }


class C3RPacketCodec:
    """Strict serializer/parser for the frozen little-endian C3R v1 packet."""

    @staticmethod
    def _finite_half(values: torch.Tensor, count: int, name: str) -> Tuple[float, ...]:
        flat = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if flat.numel() != count:
            raise ValueError("{} must contain {} values".format(name, count))
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError("{} contains non-finite values".format(name))
        return tuple(float(value) for value in flat)

    def serialize(self, message: C3RMessage) -> bytes:
        packet = bytearray(C3R_PACKET_BYTES)
        flags = 1 if bool(message.valid) else 0
        struct.pack_into(
            "<4sBBHIIQHHI",
            packet,
            0,
            C3R_MAGIC,
            C3R_VERSION,
            flags,
            int(message.sender_id),
            int(message.sequence_hash),
            int(message.frame_id),
            int(message.timestamp_ms),
            C3R_PAYLOAD_BYTES,
            0,
            0,
        )
        struct.pack_into("<4e", packet, 32, *self._finite_half(message.bbox, 4, "bbox"))
        struct.pack_into(
            "<4e", packet, 40,
            *self._finite_half(message.bbox_delta, 4, "bbox_delta")
        )
        struct.pack_into("<4e", packet, 48, *self._finite_half(message.quality, 4, "quality"))
        struct.pack_into("<4e", packet, 56, *self._finite_half(message.scales, 4, "scales"))

        quantized = message.quantized_prompt.detach().to(device="cpu", dtype=torch.int8)
        if tuple(quantized.shape) != C3R_PROMPT_SHAPE:
            raise ValueError("quantized prompt must have shape [4,64]")
        packet[64:320] = bytes((int(value) & 0xFF) for value in quantized.reshape(-1).tolist())
        crc = zlib.crc32(bytes(packet[0:28]) + bytes(packet[32:320])) & 0xFFFFFFFF
        struct.pack_into("<I", packet, 28, crc)
        result = bytes(packet)
        if len(result) != C3R_PACKET_BYTES:
            raise AssertionError("C3R serializer emitted a non-320-byte packet")
        return result

    def parse(self, packet: Union[bytes, bytearray, memoryview]) -> C3RMessage:
        raw = bytes(packet)
        if len(raw) != C3R_PACKET_BYTES:
            raise PacketValidationError("length")
        (magic, version, flags, sender_id, sequence_hash, frame_id,
         timestamp_ms, payload_bytes, reserved, stored_crc) = struct.unpack_from(
            "<4sBBHIIQHHI", raw, 0
        )
        if magic != C3R_MAGIC:
            raise PacketValidationError("magic")
        if version != C3R_VERSION:
            raise PacketValidationError("version")
        if flags & 0xFE:
            raise PacketValidationError("flags")
        if not flags & 0x01:
            raise PacketValidationError("invalid_flag")
        if payload_bytes != C3R_PAYLOAD_BYTES:
            raise PacketValidationError("payload_bytes")
        if reserved != 0:
            raise PacketValidationError("reserved")
        computed_crc = zlib.crc32(raw[0:28] + raw[32:320]) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise PacketValidationError("crc32")

        bbox = torch.tensor(struct.unpack_from("<4e", raw, 32), dtype=torch.float32)
        bbox_delta = torch.tensor(struct.unpack_from("<4e", raw, 40), dtype=torch.float32)
        quality = torch.tensor(struct.unpack_from("<4e", raw, 48), dtype=torch.float32)
        scales = torch.tensor(struct.unpack_from("<4e", raw, 56), dtype=torch.float32)
        if not bool(torch.isfinite(torch.cat((bbox, bbox_delta, quality, scales))).all().item()):
            raise PacketValidationError("non_finite")
        signed = struct.unpack_from("<256b", raw, 64)
        quantized = torch.tensor(signed, dtype=torch.int8).reshape(C3R_PROMPT_SHAPE)
        prompt = quantized.to(torch.float32) * scales[:, None]
        return C3RMessage(
            sender_id=sender_id,
            sequence_hash=sequence_hash,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            bbox=bbox,
            bbox_delta=bbox_delta,
            quality=quality,
            scales=scales,
            quantized_prompt=quantized,
            prompt=prompt,
            valid=True,
            wire_bytes=raw,
        )

    def validate_for_receiver(
        self,
        packet: Union[C3RMessage, bytes, bytearray, memoryview],
        receiver_id: int,
        sequence_hash: int,
        local_frame_id: int,
        local_timestamp_ms: int,
        frame_interval_ms: int,
        last_frame_by_sender: Optional[Mapping[int, int]] = None,
        max_age_intervals: int = 4,
    ) -> C3RMessage:
        original = packet if isinstance(packet, C3RMessage) else None
        raw = self.serialize(packet) if original is not None else bytes(packet)
        parsed = self.parse(raw)
        if parsed.sequence_hash != int(sequence_hash):
            raise PacketValidationError("sequence_hash")
        if parsed.sender_id == int(receiver_id):
            raise PacketValidationError("self_sender")
        previous = -1 if last_frame_by_sender is None else int(
            last_frame_by_sender.get(parsed.sender_id, -1))
        if parsed.frame_id <= previous:
            raise PacketValidationError("replay")
        if parsed.frame_id > int(local_frame_id):
            raise PacketValidationError("future_frame")
        age_ms = int(local_timestamp_ms) - int(parsed.timestamp_ms)
        if age_ms < 0:
            raise PacketValidationError("future_timestamp")
        max_age_ms = max(int(frame_interval_ms), 1) * int(max_age_intervals)
        if age_ms > max_age_ms:
            raise PacketValidationError("stale")
        if original is not None:
            parsed.prompt = original.prompt
            parsed.construction_label = int(original.construction_label)
            parsed.fault = str(original.fault)
        return parsed


class CompactMessageEncoder(nn.Module):
    """Response-weight four regions and project 192-D tokens to 4x64."""

    def __init__(self, token_dim: int = 192, message_dim: int = 64,
                 num_prompts: int = 4):
        super().__init__()
        if int(num_prompts) != 4 or int(message_dim) != 64:
            raise ValueError("C3R v1 freezes four 64-D prompt tokens")
        self.token_dim = int(token_dim)
        self.message_dim = int(message_dim)
        self.num_prompts = int(num_prompts)
        self.projection = nn.Linear(self.token_dim, self.message_dim)
        self.normalization = nn.LayerNorm(self.message_dim)

    @staticmethod
    def message_contract() -> Dict[str, object]:
        return {
            "projected_prompt_shape": [4, 64],
            "projected_prompt_dtype": "float32 before quantization",
            "wire_prompt_shape": [4, 64],
            "wire_prompt_dtype": "int8",
            "serialized_payload_bytes": C3R_PACKET_BYTES,
            "fields": MessageAccounting.field_bytes(),
        }

    @staticmethod
    def response_quality(response: torch.Tensor) -> torch.Tensor:
        if response.dim() == 3:
            response = response.unsqueeze(1)
        if response.dim() != 4:
            raise ValueError("response must have shape [B,1,H,W]")
        values = response.float().flatten(1)
        values = torch.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        peak = values.max(dim=1).values
        minimum = values.min(dim=1).values
        apce = (peak - minimum).square() / (
            (values - minimum[:, None]).square().mean(dim=1) + 1e-12
        )
        apce_norm = (torch.log1p(apce) / math.log1p(500.0)).clamp(0.0, 1.0)
        probabilities = values.clamp_min(0.0) + 1e-12
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        entropy = -(probabilities * probabilities.log()).sum(dim=1)
        entropy = (entropy / math.log(float(values.shape[1]))).clamp(0.0, 1.0)
        top2 = torch.topk(values, k=min(2, values.shape[1]), dim=1).values
        margin = top2[:, 0] if top2.shape[1] == 1 else top2[:, 0] - top2[:, 1]
        return torch.stack((peak, apce_norm, entropy, margin.clamp(0.0, 1.0)), dim=1)

    def pool_project(self, search_tokens: torch.Tensor,
                     response: torch.Tensor) -> torch.Tensor:
        if search_tokens.dim() != 3 or search_tokens.shape[-1] != self.token_dim:
            raise ValueError("search_tokens must have shape [B,HW,{}]".format(self.token_dim))
        if response.dim() == 3:
            response = response.unsqueeze(1)
        batch, _, height, width = response.shape
        if search_tokens.shape[0] != batch or search_tokens.shape[1] != height * width:
            raise ValueError("response grid and search-token count do not match")
        token_grid = search_tokens.reshape(batch, height, width, self.token_dim)
        response_grid = response[:, 0].to(dtype=search_tokens.dtype)
        pooled_rows: List[torch.Tensor] = []
        for batch_index in range(batch):
            maximum = int(response_grid[batch_index].reshape(-1).argmax().item())
            center_y, center_x = divmod(maximum, width)
            start_y = min(max(center_y - 4, 0), max(height - 8, 0))
            start_x = min(max(center_x - 4, 0), max(width - 8, 0))
            window_tokens = token_grid[batch_index, start_y:start_y + 8, start_x:start_x + 8]
            window_response = response_grid[batch_index, start_y:start_y + 8, start_x:start_x + 8]
            regions = ((0, 4, 0, 4), (0, 4, 4, 8),
                       (4, 8, 0, 4), (4, 8, 4, 8))
            rows = []
            for y0, y1, x0, x1 in regions:
                region_tokens = window_tokens[y0:y1, x0:x1].reshape(-1, self.token_dim)
                weights = window_response[y0:y1, x0:x1].reshape(-1).clamp_min(0.0)
                mass = weights.sum()
                weighted = (region_tokens * weights[:, None]).sum(dim=0) / mass.clamp_min(1e-12)
                uniform = region_tokens.mean(dim=0)
                rows.append(torch.where(mass > 1e-12, weighted, uniform))
            pooled_rows.append(torch.stack(rows, dim=0))
        pooled = torch.stack(pooled_rows, dim=0)
        return self.normalization(self.projection(pooled))

    @staticmethod
    def quantize(prompt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prompt.shape[-2:] != C3R_PROMPT_SHAPE:
            raise ValueError("prompt must end with shape [4,64]")
        detached = prompt.detach().float()
        maximum = detached.abs().amax(dim=-1)
        scales = maximum / 127.0
        scales = torch.where(maximum == 0, torch.ones_like(scales), scales)
        quantized = torch.round((detached / scales[..., None]).clamp(-127.0, 127.0)).to(torch.int8)
        reconstructed = quantized.to(prompt.dtype) * scales.to(prompt.dtype)[..., None]
        straight_through = prompt + (reconstructed - prompt).detach()
        return quantized, scales, straight_through

    def forward(
        self,
        search_tokens: torch.Tensor,
        response: torch.Tensor,
        bbox: torch.Tensor,
        previous_bbox: Optional[torch.Tensor],
        sender_ids: Sequence[int],
        sequence_hashes: Sequence[int],
        frame_ids: Sequence[int],
        timestamp_ms: Sequence[int],
    ) -> List[C3RMessage]:
        prompt = self.pool_project(search_tokens, response)
        quantized, scales, reconstructed = self.quantize(prompt)
        quality = self.response_quality(response)
        bbox = bbox.reshape(bbox.shape[0], -1, 4).mean(dim=1).float().clamp(0.0, 1.0)
        if previous_bbox is None:
            previous_bbox = bbox.detach()
        previous_bbox = previous_bbox.reshape(previous_bbox.shape[0], -1, 4).mean(dim=1).float()
        bbox_delta = (bbox - previous_bbox).clamp(-1.0, 1.0)
        batch = search_tokens.shape[0]
        metadata = (sender_ids, sequence_hashes, frame_ids, timestamp_ms)
        if any(len(values) != batch for values in metadata):
            raise ValueError("packet metadata lengths must match the batch")
        messages = []
        for index in range(batch):
            messages.append(C3RMessage(
                sender_id=int(sender_ids[index]),
                sequence_hash=int(sequence_hashes[index]),
                frame_id=int(frame_ids[index]),
                timestamp_ms=int(timestamp_ms[index]),
                bbox=bbox[index],
                bbox_delta=bbox_delta[index],
                quality=quality[index],
                scales=scales[index],
                quantized_prompt=quantized[index],
                prompt=reconstructed[index],
            ))
        return messages


class RemoteMessageAdapter(nn.Module):
    """Shared one-head width-64 cross-attention remote adapter."""

    def __init__(self, token_dim: int = 192, message_dim: int = 64):
        super().__init__()
        self.query = nn.Linear(token_dim, message_dim)
        self.key = nn.Linear(message_dim, message_dim)
        self.value = nn.Linear(message_dim, message_dim)
        self.output = nn.Linear(message_dim, token_dim)
        self.scale = float(message_dim) ** -0.5

    def attend(self, local_tokens: torch.Tensor, remote_prompt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query(local_tokens)
        key = self.key(remote_prompt)
        value = self.value(remote_prompt)
        attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, dim=-1)
        context = torch.matmul(attention, value)
        return context, query

    def forward(self, local_tokens: torch.Tensor,
                remote_prompt: torch.Tensor) -> torch.Tensor:
        context, _ = self.attend(local_tokens, remote_prompt)
        return self.output(context)


class DirectFusionAdapter(nn.Module):
    """A2 bottlenecked concat projection; it replaces rather than adds local tokens."""

    def __init__(self, token_dim: int = 192, message_dim: int = 64):
        super().__init__()
        self.query = nn.Linear(token_dim, message_dim)
        self.key = nn.Linear(message_dim, message_dim)
        self.value = nn.Linear(message_dim, message_dim)
        self.output = nn.Linear(message_dim * 2, token_dim)
        self.scale = float(message_dim) ** -0.5

    def forward(self, local_tokens: torch.Tensor,
                remote_prompt: torch.Tensor) -> torch.Tensor:
        query = self.query(local_tokens)
        key = self.key(remote_prompt)
        value = self.value(remote_prompt)
        attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, dim=-1)
        context = torch.matmul(attention, value)
        return self.output(torch.cat((query, context), dim=-1))


class ReliabilityGate(nn.Module):
    """Exactly the frozen prediction-only 10->32->1 reliability network."""

    def __init__(self, max_gate: float = 0.25):
        super().__init__()
        if float(max_gate) != 0.25:
            raise ValueError("C3R v1 freezes max_gate=0.25")
        self.max_gate = float(max_gate)
        self.network = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    @staticmethod
    def normalize_inputs(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != 10:
            raise ValueError("reliability input must contain exactly 10 values")
        return torch.nan_to_num(
            inputs.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(-1.0, 1.0)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.normalize_inputs(inputs)
        logit = self.network(inputs)
        return self.max_gate * torch.sigmoid(logit), logit

    def detached_instrumentation(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Recompute detached activations without touching the behavior graph."""
        normalized = self.normalize_inputs(inputs.detach())
        with torch.no_grad():
            hidden_pre = self.network[0](normalized)
            hidden_post = self.network[1](hidden_pre)
            output_pre = self.network[2](hidden_post)
            sigmoid_activation = torch.sigmoid(output_pre)
            gate = self.max_gate * sigmoid_activation
        return {
            "normalized_input": normalized.detach(),
            "hidden_pre_activation": hidden_pre.detach(),
            "hidden_post_activation": hidden_post.detach(),
            "output_pre_sigmoid": output_pre.detach(),
            "sigmoid_activation": sigmoid_activation.detach(),
            "gate": gate.detach(),
        }


def _relative_cap(residual: torch.Tensor, local_tokens: torch.Tensor,
                  cap: float, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_norm = residual.float().norm().clamp_min(eps)
    local_norm = local_tokens.float().norm().clamp_min(eps)
    ratio = residual_norm / local_norm
    scale = torch.clamp(torch.as_tensor(float(cap), device=residual.device) / ratio, max=1.0)
    capped = residual * scale.to(dtype=residual.dtype)
    excess = torch.relu(ratio - float(cap))
    return capped, ratio, excess


class LocalFirstResidualFusion(nn.Module):
    """Frozen C0/C1/A1 residual rules and A2 direct control."""

    VALID_VARIANTS = ("c0", "c1", "a1", "a2")

    def __init__(self, variant: str = "c1", peer_cap: float = 0.25,
                 aggregate_cap: float = 0.35):
        super().__init__()
        variant = str(variant).lower()
        if variant not in self.VALID_VARIANTS:
            raise ValueError("Unsupported C3R variant: {}".format(variant))
        if float(peer_cap) != 0.25 or float(aggregate_cap) != 0.35:
            raise ValueError("C3R v1 freezes residual caps at 0.25 and 0.35")
        self.variant = variant
        self.peer_cap = float(peer_cap)
        self.aggregate_cap = float(aggregate_cap)

    def forward(
        self,
        local_tokens: torch.Tensor,
        peer_outputs: Sequence[torch.Tensor],
        gates: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not peer_outputs:
            zero = local_tokens.new_zeros(())
            return local_tokens, {
                "preclamp_excess": zero,
                "aggregate_ratio": zero,
                "residual_budget_loss": zero,
            }
        if self.variant == "a2":
            weights = torch.stack([gate.reshape(()) for gate in gates])
            weights = weights / weights.sum().clamp_min(1e-12)
            direct = sum(weight * output for weight, output in zip(weights, peer_outputs))
            zero = local_tokens.new_zeros(())
            return direct, {
                "preclamp_excess": zero,
                "aggregate_ratio": (direct.float().norm() / local_tokens.float().norm().clamp_min(1e-12)),
                "residual_budget_loss": zero,
            }
        if self.variant == "c0":
            residual = sum(peer_outputs) / float(len(peer_outputs))
            ratio = residual.float().norm() / local_tokens.float().norm().clamp_min(1e-12)
            zero = local_tokens.new_zeros(())
            return local_tokens + residual, {
                "preclamp_excess": zero,
                "aggregate_ratio": ratio,
                "residual_budget_loss": zero,
            }

        capped_peers = []
        excesses = []
        ratios = []
        for output, gate in zip(peer_outputs, gates):
            gated = output * gate.to(dtype=output.dtype)
            capped, ratio, excess = _relative_cap(gated, local_tokens, self.peer_cap)
            capped_peers.append(capped)
            ratios.append(ratio)
            excesses.append(excess)
        residual = sum(capped_peers)
        residual, aggregate_ratio_before, aggregate_excess = _relative_cap(
            residual, local_tokens, self.aggregate_cap)
        excesses.append(aggregate_excess)
        budget_loss = torch.stack(excesses).mean()
        aggregate_ratio = residual.float().norm() / local_tokens.float().norm().clamp_min(1e-12)
        return local_tokens + residual, {
            "preclamp_excess": torch.stack(excesses),
            "peer_ratio_before": torch.stack(ratios),
            "aggregate_ratio_before": aggregate_ratio_before,
            "aggregate_ratio": aggregate_ratio,
            "residual_budget_loss": budget_loss,
        }


class CommunicationPerturbation:
    """Remote-only frozen fault injector.  Local inputs are never accepted."""

    def __init__(
        self,
        enabled: bool = False,
        dropout_probability: float = 0.25,
        delays: Sequence[int] = (0, 1, 2, 4),
        delay_probabilities: Sequence[float] = (0.50, 0.20, 0.20, 0.10),
        corruption_probability: float = 0.15,
        wrong_remote_probability: float = 0.20,
        conflict_probability: float = 0.15,
        seed: int = 20260716,
    ):
        self.enabled = bool(enabled)
        self.dropout_probability = float(dropout_probability)
        self.delays = tuple(int(value) for value in delays)
        self.delay_probabilities = tuple(float(value) for value in delay_probabilities)
        self.corruption_probability = float(corruption_probability)
        self.wrong_remote_probability = float(wrong_remote_probability)
        self.conflict_probability = float(conflict_probability)
        self.random = random.Random(int(seed))
        if len(self.delays) != len(self.delay_probabilities):
            raise ValueError("delay values and probabilities must align")
        if abs(sum(self.delay_probabilities) - 1.0) > 1e-8:
            raise ValueError("delay probabilities must sum to one")

    def _sample_delay(self) -> int:
        value = self.random.random()
        cumulative = 0.0
        for delay, probability in zip(self.delays, self.delay_probabilities):
            cumulative += probability
            if value <= cumulative:
                return delay
        return self.delays[-1]

    def apply(
        self,
        messages: Sequence[C3RMessage],
        frame_interval_ms: int,
        wrong_pool: Optional[Sequence[C3RMessage]] = None,
        force_fault: Optional[str] = None,
    ) -> List[C3RMessage]:
        copied = [message.detached_copy(detach=False) for message in messages]
        if not self.enabled and force_fault is None:
            return copied
        output: List[C3RMessage] = []
        for index, message in enumerate(copied):
            fault = force_fault
            if fault is None and self.random.random() < self.dropout_probability:
                fault = "dropout"
            if fault == "dropout":
                continue
            if fault is None:
                delay = self._sample_delay()
                if delay:
                    message.frame_id = max(0, message.frame_id - delay)
                    message.timestamp_ms = max(0, message.timestamp_ms - delay * int(frame_interval_ms))
                    message.fault = "delay"
                    if delay >= 4:
                        message.construction_label = 0
            elif fault in ("delay", "stale"):
                delay = 1 if fault == "delay" else 5
                message.frame_id = max(0, message.frame_id - delay)
                message.timestamp_ms = max(0, message.timestamp_ms - delay * int(frame_interval_ms))
                message.fault = fault
                message.construction_label = 0 if fault == "stale" else message.construction_label

            wrong_selected = fault == "wrong_remote" or (
                fault is None and wrong_pool and self.random.random() < self.wrong_remote_probability)
            if wrong_selected:
                if wrong_pool:
                    replacement = wrong_pool[index % len(wrong_pool)].detached_copy(detach=False)
                    replacement.sender_id = message.sender_id
                    replacement.sequence_hash = message.sequence_hash
                    replacement.frame_id = message.frame_id
                    replacement.timestamp_ms = message.timestamp_ms
                    message = replacement
                else:
                    message.prompt = torch.roll(message.prompt, shifts=1, dims=0)
                    message.quantized_prompt = torch.roll(message.quantized_prompt, shifts=1, dims=0)
                message.construction_label = 0
                message.fault = "wrong_remote"

            corrupt_selected = fault == "corrupt" or (
                fault is None and self.random.random() < self.corruption_probability)
            if corrupt_selected:
                flat = message.quantized_prompt.reshape(-1).clone()
                count = max(1, int(round(flat.numel() * 0.10)))
                indices = list(range(flat.numel()))
                self.random.shuffle(indices)
                selected = indices[:count]
                flat[selected] = 0
                noise = torch.tensor(
                    [self.random.randint(-3, 3) for _ in selected],
                    dtype=torch.int16, device=flat.device)
                values = flat[selected].to(torch.int16) + noise
                flat[selected] = values.clamp(-127, 127).to(torch.int8)
                message.quantized_prompt = flat.reshape(C3R_PROMPT_SHAPE)
                message.prompt = message.quantized_prompt.float() * message.scales[:, None]
                message.construction_label = 0
                message.fault = "corrupt"

            if fault == "zero":
                message.quantized_prompt = torch.zeros_like(message.quantized_prompt)
                message.prompt = torch.zeros_like(message.prompt)
                message.scales = torch.ones_like(message.scales)
                message.construction_label = 0
                message.fault = "zero"
            message.wire_bytes = None
            output.append(message)

        if len(output) >= 2 and (
                force_fault == "one_bad" or
                (force_fault is None and self.random.random() < self.conflict_probability)):
            bad = output[-1]
            bad.prompt = torch.roll(bad.prompt, shifts=1, dims=0)
            bad.quantized_prompt = torch.roll(bad.quantized_prompt, shifts=1, dims=0)
            bad.construction_label = 0
            bad.fault = "one_bad"
            bad.wire_bytes = None
        return output


def gate_ranking_loss(logits: torch.Tensor, labels: torch.Tensor,
                      margin: float = 0.0) -> torch.Tensor:
    """Rank construction-positive messages above synthetic negatives."""
    logits = logits.reshape(-1)
    labels = labels.reshape(-1).to(device=logits.device)
    positives = logits[labels > 0]
    negatives = logits[labels <= 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.sum() * 0.0
    differences = positives[:, None] - negatives[None, :]
    return F.softplus(float(margin) - differences).mean()


class C3R(nn.Module):
    """Transport-neutral C3R coordinator with explicit local fallback."""

    def __init__(
        self,
        token_dim: int = 192,
        message_dim: int = 64,
        num_prompts: int = 4,
        variant: str = "c1",
        max_gate: float = 0.25,
        peer_cap: float = 0.25,
        aggregate_cap: float = 0.35,
        max_age_intervals: int = 4,
    ):
        super().__init__()
        self.variant = str(variant).lower()
        self.max_age_intervals = int(max_age_intervals)
        if self.max_age_intervals != 4:
            raise ValueError("C3R v1 freezes max age at four frame intervals")
        self.encoder = CompactMessageEncoder(token_dim, message_dim, num_prompts)
        self.adapter = (
            DirectFusionAdapter(token_dim, message_dim)
            if self.variant == "a2" else RemoteMessageAdapter(token_dim, message_dim)
        )
        self.reliability = ReliabilityGate(max_gate=max_gate)
        self.fusion = LocalFirstResidualFusion(
            variant=self.variant, peer_cap=peer_cap, aggregate_cap=aggregate_cap)
        self.codec = C3RPacketCodec()

    @staticmethod
    def _semantic_cosine(local_prompt: torch.Tensor,
                         remote_prompt: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(
            local_prompt.reshape(1, -1).float(),
            remote_prompt.reshape(1, -1).float(),
            dim=1,
            eps=1e-12,
        ).clamp(-1.0, 1.0)

    def _reliability_input_parts(
        self,
        local_quality: torch.Tensor,
        remote: C3RMessage,
        local_prompt: torch.Tensor,
        age_intervals: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        remote_quality = remote.quality.to(device=local_quality.device, dtype=torch.float32)
        semantic = self._semantic_cosine(
            local_prompt,
            remote.prompt.to(device=local_prompt.device, dtype=local_prompt.dtype),
        )
        raw_values = torch.stack((
            local_quality[0], local_quality[1], 1.0 - local_quality[2], local_quality[3],
            remote_quality[0], remote_quality[1], 1.0 - remote_quality[2], remote_quality[3],
            semantic[0], local_quality.new_tensor(float(age_intervals) / 4.0),
        ))
        normalized = torch.nan_to_num(
            raw_values, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return raw_values, normalized

    def _reliability_input(
        self,
        local_quality: torch.Tensor,
        remote: C3RMessage,
        local_prompt: torch.Tensor,
        age_intervals: float,
    ) -> torch.Tensor:
        return self._reliability_input_parts(
            local_quality, remote, local_prompt, age_intervals)[1]

    def collaborate(
        self,
        local_tokens: torch.Tensor,
        local_response: torch.Tensor,
        packets: Sequence[Union[C3RMessage, bytes, bytearray, memoryview]],
        receiver_id: int,
        sequence_hash: int,
        local_frame_id: int,
        local_timestamp_ms: int,
        frame_interval_ms: int,
        last_frame_by_sender: Optional[MutableMapping[int, int]] = None,
        instrumentation: bool = False,
        remote_information_diagnostics: bool = False,
        gate_provider: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, object]:
        if remote_information_diagnostics and not instrumentation:
            raise ValueError(
                "remote-information diagnostics require C3R instrumentation")
        if local_tokens.dim() == 2:
            local_tokens = local_tokens.unsqueeze(0)
        if local_response.dim() == 3:
            local_response = local_response.unsqueeze(0)
        if local_tokens.shape[0] != 1 or local_response.shape[0] != 1:
            raise ValueError("collaborate handles one receiver record at a time")
        if not packets:
            zero = local_tokens.new_zeros(())
            return {
                "search_tokens": local_tokens,
                "used_remote": False,
                "accepted_count": 0,
                "dispositions": [],
                "gate_logits": local_tokens.new_empty((0,)),
                "gate_labels": local_tokens.new_empty((0,), dtype=torch.long),
                "gates": local_tokens.new_empty((0,)),
                "residual_budget_loss": zero,
                "aggregate_ratio": zero,
            }

        accepted: List[C3RMessage] = []
        dispositions: List[PacketDisposition] = []
        seen_senders = set()
        for packet in packets:
            sender_hint = packet.sender_id if isinstance(packet, C3RMessage) else None
            try:
                message = self.codec.validate_for_receiver(
                    packet,
                    receiver_id=receiver_id,
                    sequence_hash=sequence_hash,
                    local_frame_id=local_frame_id,
                    local_timestamp_ms=local_timestamp_ms,
                    frame_interval_ms=frame_interval_ms,
                    last_frame_by_sender=last_frame_by_sender,
                    max_age_intervals=self.max_age_intervals,
                )
                if message.sender_id in seen_senders:
                    raise PacketValidationError("duplicate_sender")
                seen_senders.add(message.sender_id)
                accepted.append(message)
                dispositions.append(PacketDisposition(
                    message.sender_id, True, "accepted", C3R_PACKET_BYTES))
            except (PacketValidationError, ValueError, struct.error) as error:
                reason = error.reason if isinstance(error, PacketValidationError) else "malformed"
                dispositions.append(PacketDisposition(
                    sender_hint, False, reason,
                    C3R_PACKET_BYTES if isinstance(packet, C3RMessage) else len(bytes(packet))))

        accepted.sort(key=lambda message: message.sender_id)
        if not accepted:
            zero = local_tokens.new_zeros(())
            return {
                "search_tokens": local_tokens,
                "used_remote": False,
                "accepted_count": 0,
                "dispositions": dispositions,
                "gate_logits": local_tokens.new_empty((0,)),
                "gate_labels": local_tokens.new_empty((0,), dtype=torch.long),
                "gates": local_tokens.new_empty((0,)),
                "residual_budget_loss": zero,
                "aggregate_ratio": zero,
            }

        local_prompt = self.encoder.pool_project(local_tokens, local_response)[0]
        local_quality = self.encoder.response_quality(local_response)[0]
        peer_outputs: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []
        labels: List[int] = []
        reliability_inputs: List[torch.Tensor] = []
        instrumentation_rows: List[Dict[str, object]] = []
        for message in accepted:
            remote_prompt = message.prompt.to(device=local_tokens.device, dtype=local_tokens.dtype).unsqueeze(0)
            peer_output = self.adapter(local_tokens, remote_prompt)
            peer_outputs.append(peer_output)
            age_ms = max(int(local_timestamp_ms) - int(message.timestamp_ms), 0)
            age_intervals = age_ms / float(max(int(frame_interval_ms), 1))
            reliability_input_raw, reliability_input = self._reliability_input_parts(
                local_quality, message, local_prompt, age_intervals)
            reliability_inputs.append(reliability_input)
            if gate_provider is not None:
                if self.variant != "c1":
                    raise RuntimeError("Temporal gate overrides are authorized only for C1")
                gate = torch.as_tensor(
                    gate_provider(int(message.sender_id), reliability_input.detach()),
                    device=local_tokens.device,
                    dtype=torch.float32,
                ).reshape(())
                if not bool(torch.isfinite(gate).item()) or not 0.0 <= float(gate) <= 0.25:
                    raise RuntimeError("gate override must be finite and within [0,0.25]")
                probability = (gate / 0.25).clamp(1e-7, 1.0 - 1e-7)
                logit = torch.logit(probability).reshape(1, 1)
            elif self.variant in ("c0", "a1"):
                gate = local_tokens.new_tensor([[0.25]])
                logit = local_tokens.new_zeros((1, 1))
            else:
                learned_gate, logit = self.reliability(reliability_input.unsqueeze(0))
                gate = learned_gate
            gates.append(gate.reshape(()))
            logits.append(logit.reshape(()))
            labels.append(int(message.construction_label))

            if instrumentation:
                gate_trace = self.reliability.detached_instrumentation(
                    reliability_input.unsqueeze(0))
                local_detached = local_tokens.detach().float()
                remote_detached = remote_prompt.detach().float()
                residual_detached = peer_output.detach().float()
                gate_detached = gate.detach().float().reshape(())
                gated_residual = residual_detached * gate_detached
                capped_residual, peer_ratio_before, _ = _relative_cap(
                    gated_residual, local_detached, self.fusion.peer_cap)
                local_norm = local_detached.norm()
                remote_norm = remote_detached.norm()
                residual_norm = residual_detached.norm()
                gated_norm = gated_residual.norm()
                capped_norm = capped_residual.detach().float().norm()
                residual_ratio = residual_norm / local_norm.clamp_min(1e-12)
                residual_cosine = F.cosine_similarity(
                    residual_detached.reshape(1, -1),
                    local_detached.reshape(1, -1), dim=1, eps=1e-12)[0]
                finite = bool(torch.isfinite(torch.stack((
                    local_norm, remote_norm, residual_norm, gated_norm,
                    capped_norm, residual_ratio, residual_cosine,
                ))).all().item())
                instrumentation_row = {
                    "sender_id": int(message.sender_id),
                    "message_frame_id": int(message.frame_id),
                    "message_timestamp_ms": int(message.timestamp_ms),
                    "message_age_ms": int(age_ms),
                    "message_age_intervals": float(age_intervals),
                    "valid": bool(message.valid),
                    "dropped": False,
                    "stale": bool(age_intervals > float(self.max_age_intervals)),
                    "reliability_input_raw": reliability_input_raw.detach().float().cpu().tolist(),
                    "reliability_input_normalized": gate_trace[
                        "normalized_input"].reshape(-1).float().cpu().tolist(),
                    "hidden_pre_activation": gate_trace[
                        "hidden_pre_activation"].reshape(-1).float().cpu().tolist(),
                    "hidden_post_activation": gate_trace[
                        "hidden_post_activation"].reshape(-1).float().cpu().tolist(),
                    "output_pre_sigmoid_logit": float(
                        gate_trace["output_pre_sigmoid"].reshape(-1)[0].cpu().item()),
                    "sigmoid_activation": float(
                        gate_trace["sigmoid_activation"].reshape(-1)[0].cpu().item()),
                    "final_gate": float(gate_detached.cpu().item()),
                    "diagnostic_recomputed_gate": float(
                        gate_trace["gate"].reshape(-1)[0].cpu().item()),
                    "remote_message_l2": float(remote_norm.cpu().item()),
                    "adapted_residual_l2": float(residual_norm.cpu().item()),
                    "local_feature_l2": float(local_norm.cpu().item()),
                    "adapted_residual_local_ratio": float(residual_ratio.cpu().item()),
                    "adapted_residual_local_cosine": float(residual_cosine.cpu().item()),
                    "gate_times_residual_l2": float(gated_norm.cpu().item()),
                    "peer_ratio_before_cap": float(peer_ratio_before.detach().float().cpu().item()),
                    "capped_gated_residual_l2": float(capped_norm.cpu().item()),
                    "zero_residual": bool(residual_norm.cpu().item() == 0.0),
                    "abnormal_residual": not finite,
                    "remote_quality": message.quality.detach().float().cpu().tolist(),
                    "remote_bbox_normalized_cxcywh": message.bbox.detach().float().cpu().tolist(),
                }
                if remote_information_diagnostics:
                    residual_flat = residual_detached.reshape(
                        -1, residual_detached.shape[-1])
                    residual_channel_mean = residual_flat.mean(dim=0)
                    residual_channel_std = residual_flat.std(
                        dim=0, unbiased=False)
                    local_prompt_detached = local_prompt.detach().float()
                    remote_prompt_detached = remote_prompt[0].detach().float()
                    prompt_difference = (
                        remote_prompt_detached - local_prompt_detached)
                    instrumentation_row.update({
                        "remote_information_diagnostics": True,
                        "prompt_shape": list(local_prompt_detached.shape),
                        "local_prompt_dtype": "float16",
                        "remote_prompt_dtype": "float16_strict_dequantized_wire",
                        "local_prompt_f16_zlib_b64": _encode_diagnostic_tensor(
                            local_prompt_detached, torch.float16),
                        "remote_prompt_f16_zlib_b64": _encode_diagnostic_tensor(
                            remote_prompt_detached, torch.float16),
                        "remote_quantized_prompt_i8_zlib_b64":
                            _encode_diagnostic_tensor(
                                message.quantized_prompt, torch.int8),
                        "remote_prompt_quantization_scales":
                            message.scales.detach().float().cpu().tolist(),
                        "local_prompt_l2": float(
                            local_prompt_detached.norm().cpu().item()),
                        "remote_prompt_l2": float(
                            remote_prompt_detached.norm().cpu().item()),
                        "remote_local_prompt_difference_l2": float(
                            prompt_difference.norm().cpu().item()),
                        "adapted_residual_shape": list(
                            residual_detached.shape),
                        "adapted_residual_dtype": "float16",
                        "adapted_residual_f16_zlib_b64":
                            _encode_diagnostic_tensor(
                                residual_detached, torch.float16),
                        "adapted_residual_mean": float(
                            residual_flat.mean().cpu().item()),
                        "adapted_residual_std": float(
                            residual_flat.std(unbiased=False).cpu().item()),
                        "adapted_residual_max_abs": float(
                            residual_flat.abs().max().cpu().item()),
                        "adapted_residual_channel_mean_shape": list(
                            residual_channel_mean.shape),
                        "adapted_residual_channel_mean_f16_zlib_b64":
                            _encode_diagnostic_tensor(
                                residual_channel_mean, torch.float16),
                        "adapted_residual_channel_std_shape": list(
                            residual_channel_std.shape),
                        "adapted_residual_channel_std_f16_zlib_b64":
                            _encode_diagnostic_tensor(
                                residual_channel_std, torch.float16),
                    })
                instrumentation_rows.append(instrumentation_row)

        fused, fusion_diagnostics = self.fusion(local_tokens, peer_outputs, gates)
        instrumentation_aggregate = None
        if instrumentation:
            local_detached = local_tokens.detach().float()
            fused_detached = fused.detach().float()
            aggregate_residual = fused_detached - local_detached
            local_norm = local_detached.norm()
            aggregate_norm = aggregate_residual.norm()
            aggregate_ratio = aggregate_norm / local_norm.clamp_min(1e-12)
            before_norm = local_norm
            after_norm = fused_detached.norm()
            aggregate_cosine = F.cosine_similarity(
                aggregate_residual.reshape(1, -1),
                local_detached.reshape(1, -1), dim=1, eps=1e-12)[0]
            gate_values = torch.stack([item.detach().float().reshape(()) for item in gates])
            normalized_weights = gate_values / gate_values.sum().clamp_min(1e-12)
            for row, weight in zip(instrumentation_rows, normalized_weights):
                row["multi_remote_normalized_weight"] = float(weight.cpu().item())
                row["aggregation_gate_mean"] = float(gate_values.mean().cpu().item())
                row["aggregation_gate_sum"] = float(gate_values.sum().cpu().item())
                row["aggregate_residual_l2"] = float(aggregate_norm.cpu().item())
                row["aggregate_residual_local_ratio"] = float(aggregate_ratio.cpu().item())
            instrumentation_aggregate = {
                "sender_count": len(accepted),
                "aggregation_gate_mean": float(gate_values.mean().cpu().item()),
                "aggregation_gate_sum": float(gate_values.sum().cpu().item()),
                "aggregate_residual_l2": float(aggregate_norm.cpu().item()),
                "aggregate_residual_local_ratio": float(aggregate_ratio.cpu().item()),
                "aggregate_residual_local_cosine": float(aggregate_cosine.cpu().item()),
                "feature_norm_before_fusion": float(before_norm.cpu().item()),
                "feature_norm_after_fusion": float(after_norm.cpu().item()),
                "zero_aggregate_residual": bool(aggregate_norm.cpu().item() == 0.0),
                "abnormal_aggregate_residual": not bool(torch.isfinite(torch.stack((
                    aggregate_norm, aggregate_ratio, aggregate_cosine,
                    before_norm, after_norm))).all().item()),
            }
        if last_frame_by_sender is not None:
            for message in accepted:
                last_frame_by_sender[message.sender_id] = message.frame_id
        result = {
            "search_tokens": fused,
            "used_remote": True,
            "accepted_count": len(accepted),
            "accepted_sender_ids": [message.sender_id for message in accepted],
            "dispositions": dispositions,
            "gate_logits": torch.stack(logits),
            "gate_labels": torch.tensor(labels, device=local_tokens.device, dtype=torch.long),
            "gates": torch.stack(gates),
            "reliability_inputs": torch.stack(reliability_inputs),
            **fusion_diagnostics,
        }
        if instrumentation:
            result["instrumentation_source_rows"] = instrumentation_rows
            result["instrumentation_aggregate"] = instrumentation_aggregate
        return result


def build_c3r(cfg, token_dim: int = 192) -> C3R:
    node = cfg.MODEL.C3R
    return C3R(
        token_dim=token_dim,
        message_dim=int(node.MESSAGE_DIM),
        num_prompts=int(node.NUM_PROMPTS),
        variant=str(node.VARIANT).lower(),
        max_gate=float(node.MAX_GATE),
        peer_cap=float(node.PEER_NORM_CAP),
        aggregate_cap=float(node.AGGREGATE_NORM_CAP),
        max_age_intervals=int(node.MAX_AGE_INTERVALS),
    )
