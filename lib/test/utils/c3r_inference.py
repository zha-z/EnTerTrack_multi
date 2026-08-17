"""Typed C3R inference orchestration contracts.

This module moves only prediction-derived records and serialized C3R v1
packets. It has no dataset annotation, image, or global-state API.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from lib.models.entertrack.c3r import (
    C3R,
    C3R_PACKET_BYTES,
    C3RPacketCodec,
)


TARGET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
C3R_DIAGNOSTIC_COLUMNS = [
    "target_id",
    "receiver_id",
    "frame_id",
    "timestamp_ms",
    "sequence_hash",
    "uses_gt",
    "packet_bytes",
    "sent_packets",
    "sent_bytes",
    "received_packets",
    "received_bytes",
    "accepted_packets",
    "accepted_bytes",
    "used_remote",
    "accepted_sender_ids",
    "gates",
    "aggregate_ratio",
    "dispositions",
]


def validate_target_id(target_id: str) -> str:
    value = str(target_id)
    if TARGET_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid C3R target id: {!r}".format(value))
    return value


def target_session_hash(target_id: str) -> int:
    """Stable target-session hash independent of runid and model behavior."""
    target_id = validate_target_id(target_id)
    return zlib.crc32(("C3R1:" + target_id).encode("utf-8")) & 0xFFFFFFFF


@dataclass
class C3RReceiverContext:
    target_id: str
    receiver_id: int
    sequence_hash: int
    frame_id: int
    timestamp_ms: int
    frame_interval_ms: int = 33
    last_frame_by_sender: MutableMapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_id = validate_target_id(self.target_id)
        self.receiver_id = int(self.receiver_id)
        self.sequence_hash = int(self.sequence_hash)
        self.frame_id = int(self.frame_id)
        self.timestamp_ms = int(self.timestamp_ms)
        self.frame_interval_ms = int(self.frame_interval_ms)
        if not 0 <= self.receiver_id <= 0xFFFF:
            raise ValueError("receiver_id is outside uint16 range")
        if not 0 <= self.sequence_hash <= 0xFFFFFFFF:
            raise ValueError("sequence_hash is outside uint32 range")
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ValueError("frame and timestamp must be non-negative")
        if self.frame_interval_ms <= 0:
            raise ValueError("frame_interval_ms must be positive")
        expected = target_session_hash(self.target_id)
        if self.sequence_hash != expected:
            raise ValueError("target id and sequence hash disagree")

    @classmethod
    def for_frame(cls, target_id: str, receiver_id: int, frame_id: int,
                  frame_interval_ms: int = 33,
                  last_frame_by_sender: Optional[MutableMapping[int, int]] = None):
        return cls(
            target_id=target_id,
            receiver_id=receiver_id,
            sequence_hash=target_session_hash(target_id),
            frame_id=frame_id,
            timestamp_ms=int(frame_id) * int(frame_interval_ms),
            frame_interval_ms=frame_interval_ms,
            last_frame_by_sender=(
                {} if last_frame_by_sender is None else last_frame_by_sender),
        )


@dataclass(frozen=True)
class C3RPacketRecord:
    target_id: str
    sender_id: int
    sequence_hash: int
    frame_id: int
    timestamp_ms: int
    payload: bytes

    def __post_init__(self) -> None:
        target_id = validate_target_id(self.target_id)
        payload = bytes(self.payload)
        if len(payload) != C3R_PACKET_BYTES:
            raise ValueError("C3R packet record must contain exactly 320 bytes")
        parsed = C3RPacketCodec().parse(payload)
        expected = target_session_hash(target_id)
        if int(self.sequence_hash) != expected:
            raise ValueError("record target and sequence hash disagree")
        fields = (
            ("sender_id", int(self.sender_id), int(parsed.sender_id)),
            ("sequence_hash", int(self.sequence_hash), int(parsed.sequence_hash)),
            ("frame_id", int(self.frame_id), int(parsed.frame_id)),
            ("timestamp_ms", int(self.timestamp_ms), int(parsed.timestamp_ms)),
        )
        for name, declared, encoded in fields:
            if declared != encoded:
                raise ValueError("record {} disagrees with packet".format(name))
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "payload", payload)


@dataclass(frozen=True)
class C3RFrameExchange:
    target_id: str
    sequence_hash: int
    frame_id: int
    timestamp_ms: int
    records: Tuple[C3RPacketRecord, ...]
    frame_interval_ms: int = 33

    def __post_init__(self) -> None:
        target_id = validate_target_id(self.target_id)
        expected = target_session_hash(target_id)
        if int(self.frame_interval_ms) <= 0:
            raise ValueError("frame_interval_ms must be positive")
        if int(self.sequence_hash) != expected:
            raise ValueError("exchange target and sequence hash disagree")
        ordered = tuple(sorted(tuple(self.records), key=lambda item: item.sender_id))
        sender_ids = [int(record.sender_id) for record in ordered]
        if len(sender_ids) != len(set(sender_ids)):
            raise ValueError("duplicate sender in one C3R frame exchange")
        for record in ordered:
            if record.target_id != target_id:
                raise ValueError("mixed target in C3R frame exchange")
            if int(record.sequence_hash) != int(self.sequence_hash):
                raise ValueError("mixed session in C3R frame exchange")
            frame_age = int(self.frame_id) - int(record.frame_id)
            timestamp_age = int(self.timestamp_ms) - int(record.timestamp_ms)
            if frame_age < 0 or timestamp_age < 0:
                raise ValueError("future packet in C3R frame exchange")
            if frame_age > 4 or timestamp_age > 4 * int(self.frame_interval_ms):
                raise ValueError("stale packet in C3R frame exchange")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "records", ordered)

    def packets_for(self, receiver_id: int) -> Tuple[bytes, ...]:
        receiver_id = int(receiver_id)
        return tuple(
            record.payload for record in self.records
            if int(record.sender_id) != receiver_id
        )

    def sender_ids_for(self, receiver_id: int) -> Tuple[int, ...]:
        receiver_id = int(receiver_id)
        return tuple(
            int(record.sender_id) for record in self.records
            if int(record.sender_id) != receiver_id
        )


@dataclass
class C3RHeadResult:
    output: Mapping[str, object]
    collaboration: Mapping[str, object]
    used_remote: bool


def _normalized_cxcywh(bbox_xywh, image_height: int, image_width: int,
                       device: torch.device) -> torch.Tensor:
    values = torch.as_tensor(
        bbox_xywh, device=device, dtype=torch.float32).reshape(4)
    x, y, width, height = values.unbind()
    result = torch.stack((
        (x + 0.5 * width) / max(float(image_width), 1.0),
        (y + 0.5 * height) / max(float(image_height), 1.0),
        width / max(float(image_width), 1.0),
        height / max(float(image_height), 1.0),
    )).clamp(0.0, 1.0)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("predicted bbox produced non-finite packet state")
    return result.unsqueeze(0)


def build_packet_record(c3r: C3R, feat_len_s: int,
                        candidate: Mapping[str, object], target_id: str,
                        sender_id: int, frame_id: int, timestamp_ms: int,
                        image_height: int, image_width: int) -> C3RPacketRecord:
    """Build one wire packet from a local prediction candidate."""
    out = candidate["out_dict"]
    feature = out["backbone_feat"]
    search_tokens = feature[:, -int(feat_len_s):]
    response = out["score_map"]
    bbox = _normalized_cxcywh(
        candidate["target_bbox"], image_height, image_width, feature.device)
    previous_bbox = _normalized_cxcywh(
        candidate["prev_bbox"], image_height, image_width, feature.device)
    sequence_hash = target_session_hash(target_id)
    message = c3r.encoder(
        search_tokens=search_tokens,
        response=response,
        bbox=bbox,
        previous_bbox=previous_bbox,
        sender_ids=[int(sender_id)],
        sequence_hashes=[sequence_hash],
        frame_ids=[int(frame_id)],
        timestamp_ms=[int(timestamp_ms)],
    )[0]
    payload = c3r.codec.serialize(message)
    return C3RPacketRecord(
        target_id=target_id,
        sender_id=int(sender_id),
        sequence_hash=sequence_hash,
        frame_id=int(frame_id),
        timestamp_ms=int(timestamp_ms),
        payload=payload,
    )


def collaborate_local_candidate(c3r: C3R, forward_head, feat_len_s: int,
                                candidate: Mapping[str, object],
                                packets: Sequence[bytes],
                                context: C3RReceiverContext,
                                instrumentation: bool = False,
                                remote_information_diagnostics: bool = False,
                                gate_provider: Optional[Callable[
                                    [int, torch.Tensor], torch.Tensor]] = None,
                                ) -> C3RHeadResult:
    """Apply existing C3R logic after one retained local backbone forward."""
    out = candidate["out_dict"]
    feature = out["backbone_feat"]
    search_tokens = feature[:, -int(feat_len_s):]
    collaboration = c3r.collaborate(
        local_tokens=search_tokens,
        local_response=out["score_map"],
        packets=tuple(packets),
        receiver_id=context.receiver_id,
        sequence_hash=context.sequence_hash,
        local_frame_id=context.frame_id,
        local_timestamp_ms=context.timestamp_ms,
        frame_interval_ms=context.frame_interval_ms,
        last_frame_by_sender=context.last_frame_by_sender,
        instrumentation=bool(instrumentation),
        remote_information_diagnostics=bool(
            remote_information_diagnostics),
        gate_provider=gate_provider,
    )
    if not bool(collaboration["used_remote"]):
        return C3RHeadResult(
            output=out, collaboration=collaboration, used_remote=False)
    fused_feature = torch.cat((
        feature[:, :-int(feat_len_s)], collaboration["search_tokens"]), dim=1)
    collaborative_out = forward_head(fused_feature, None)
    collaborative_out["c3r"] = collaboration
    collaborative_out["local_output"] = out
    collaborative_out["local_search_tokens"] = search_tokens
    collaborative_out["backbone_feat"] = feature
    return C3RHeadResult(
        output=collaborative_out, collaboration=collaboration, used_remote=True)


def _finite_scalar(value, default=0.0) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        value = value.detach().float().reshape(-1)[0].cpu().item()
    value = float(value)
    return value if math.isfinite(value) else float(default)


def diagnostic_row(target_id: str, receiver_id: int,
                   context: C3RReceiverContext,
                   sent_packets: int, received_packets: int,
                   collaboration: Mapping[str, object]) -> Dict[str, object]:
    dispositions = collaboration.get("dispositions", ())
    disposition_rows = [
        {
            "sender_id": item.sender_id,
            "accepted": bool(item.accepted),
            "reason": str(item.reason),
            "serialized_bytes": int(item.serialized_bytes),
        }
        for item in dispositions
    ]
    gates = collaboration.get("gates", torch.empty(0))
    gate_values = (
        gates.detach().float().cpu().tolist()
        if torch.is_tensor(gates) else list(gates or ()))
    accepted = int(collaboration.get("accepted_count", 0))
    return {
        "target_id": validate_target_id(target_id),
        "receiver_id": int(receiver_id),
        "frame_id": int(context.frame_id),
        "timestamp_ms": int(context.timestamp_ms),
        "sequence_hash": int(context.sequence_hash),
        "uses_gt": False,
        "packet_bytes": C3R_PACKET_BYTES,
        "sent_packets": int(sent_packets),
        "sent_bytes": int(sent_packets) * C3R_PACKET_BYTES,
        "received_packets": int(received_packets),
        "received_bytes": int(received_packets) * C3R_PACKET_BYTES,
        "accepted_packets": accepted,
        "accepted_bytes": accepted * C3R_PACKET_BYTES,
        "used_remote": bool(collaboration.get("used_remote", False)),
        "accepted_sender_ids": json.dumps(
            collaboration.get("accepted_sender_ids", ()), separators=(",", ":")),
        "gates": json.dumps(gate_values, separators=(",", ":")),
        "aggregate_ratio": _finite_scalar(
            collaboration.get("aggregate_ratio", 0.0)),
        "dispositions": json.dumps(
            disposition_rows, sort_keys=True, separators=(",", ":")),
    }
