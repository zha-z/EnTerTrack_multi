from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import random
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence


@dataclass
class Message:
    src: int
    dst: int
    payload: Dict[str, Any]
    frame_idx: int
    deliver_frame: int
    size_bytes: int
    priority: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationStats:
    sent: int = 0
    delivered: int = 0
    dropped_by_loss: int = 0
    dropped_by_bandwidth: int = 0
    bytes_sent: int = 0
    bytes_delivered: int = 0
    total_delay_frames: int = 0

    def as_dict(self, num_frames: int = 0) -> Dict[str, float]:
        attempted = self.sent + self.dropped_by_loss + self.dropped_by_bandwidth
        return {
            "sent": self.sent,
            "delivered": self.delivered,
            "dropped_by_loss": self.dropped_by_loss,
            "dropped_by_bandwidth": self.dropped_by_bandwidth,
            "bytes_sent": self.bytes_sent,
            "bytes_delivered": self.bytes_delivered,
            "delivery_ratio": self.delivered / max(1, self.sent),
            "attempted": attempted,
            "bytes_per_frame": self.bytes_sent / max(1, num_frames),
            "average_delay_frames": self.total_delay_frames / max(1, self.delivered),
        }


def estimate_payload_size_bytes(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, bool):
        return 1
    if isinstance(payload, int):
        return 4
    if isinstance(payload, float):
        return 4
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    if isinstance(payload, bytes):
        return len(payload)
    if isinstance(payload, dict):
        return sum(estimate_payload_size_bytes(k) + estimate_payload_size_bytes(v) for k, v in payload.items())
    if isinstance(payload, (list, tuple)):
        return sum(estimate_payload_size_bytes(v) for v in payload)

    shape = getattr(payload, "shape", None)
    dtype = str(getattr(payload, "dtype", "float32"))
    if shape is not None:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        if "int8" in dtype or "uint8" in dtype:
            bytes_per_value = 1
        elif "float16" in dtype or "bfloat16" in dtype:
            bytes_per_value = 2
        elif "float64" in dtype or "int64" in dtype:
            bytes_per_value = 8
        else:
            bytes_per_value = 4
        return int(numel * bytes_per_value)

    return 64


class CommunicationSimulator:
    """Frame-level communication simulator for cooperative tracking evaluation."""

    def __init__(
        self,
        num_agents: int,
        send_interval: int = 1,
        bandwidth_limit_bytes_per_frame: Optional[int] = None,
        packet_loss: float = 0.0,
        delay_frames: int = 0,
        max_neighbors: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if send_interval <= 0:
            raise ValueError("send_interval must be positive")
        if not 0.0 <= packet_loss <= 1.0:
            raise ValueError("packet_loss must be in [0, 1]")
        if delay_frames < 0:
            raise ValueError("delay_frames must be non-negative")

        self.num_agents = num_agents
        self.send_interval = send_interval
        self.bandwidth_limit_bytes_per_frame = bandwidth_limit_bytes_per_frame
        self.packet_loss = packet_loss
        self.delay_frames = delay_frames
        self.max_neighbors = max_neighbors
        self.rng = random.Random(seed)
        self.queue: DefaultDict[int, List[Message]] = defaultdict(list)
        self.stats = CommunicationStats()
        self._used_bandwidth: DefaultDict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def should_send(self, frame_idx: int, event_triggered: bool = False) -> bool:
        return event_triggered or frame_idx % self.send_interval == 0

    def select_neighbors(
        self,
        src: int,
        candidates: Sequence[int],
        scores: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        neighbors = [agent_id for agent_id in candidates if agent_id != src]
        if scores:
            neighbors.sort(key=lambda agent_id: scores.get(agent_id, 0.0), reverse=True)
        if self.max_neighbors is not None:
            neighbors = neighbors[: self.max_neighbors]
        return neighbors

    def send(
        self,
        frame_idx: int,
        src: int,
        dsts: Iterable[int],
        payload: Dict[str, Any],
        priority: float = 0.0,
        event_triggered: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        if not self.should_send(frame_idx, event_triggered=event_triggered):
            return []

        size_bytes = estimate_payload_size_bytes(payload)
        accepted: List[Message] = []
        for dst in dsts:
            if dst == src:
                continue

            used = self._used_bandwidth[frame_idx][src]
            if self.bandwidth_limit_bytes_per_frame is not None and used + size_bytes > self.bandwidth_limit_bytes_per_frame:
                self.stats.dropped_by_bandwidth += 1
                continue

            if self.rng.random() < self.packet_loss:
                self.stats.dropped_by_loss += 1
                continue

            msg = Message(
                src=src,
                dst=dst,
                payload=payload,
                frame_idx=frame_idx,
                deliver_frame=frame_idx + self.delay_frames,
                size_bytes=size_bytes,
                priority=priority,
                metadata=metadata or {},
            )
            self.queue[msg.deliver_frame].append(msg)
            self._used_bandwidth[frame_idx][src] += size_bytes
            self.stats.sent += 1
            self.stats.bytes_sent += size_bytes
            accepted.append(msg)

        return accepted

    def deliver(self, frame_idx: int, dst: Optional[int] = None) -> List[Message]:
        ready = self.queue.pop(frame_idx, [])
        if dst is None:
            delivered = ready
            remaining: List[Message] = []
        else:
            delivered = [msg for msg in ready if msg.dst == dst]
            remaining = [msg for msg in ready if msg.dst != dst]
            if remaining:
                self.queue[frame_idx].extend(remaining)

        self.stats.delivered += len(delivered)
        self.stats.bytes_delivered += sum(msg.size_bytes for msg in delivered)
        self.stats.total_delay_frames += sum(msg.deliver_frame - msg.frame_idx for msg in delivered)
        return delivered

    def flush_until(self, last_frame_idx: int) -> List[Message]:
        messages: List[Message] = []
        for frame_idx in sorted(list(self.queue.keys())):
            if frame_idx <= last_frame_idx:
                messages.extend(self.deliver(frame_idx))
        return messages
