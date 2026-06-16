import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.test.coop.communication_simulator import CommunicationSimulator


def run_demo():
    sim = CommunicationSimulator(
        num_agents=3,
        send_interval=2,
        bandwidth_limit_bytes_per_frame=64,
        packet_loss=0.0,
        delay_frames=1,
        seed=0,
    )

    for frame_idx in range(5):
        sim.send(frame_idx, src=0, dsts=[1, 2], payload={"bbox": [1, 2, 3, 4], "score": 0.9})
        sim.deliver(frame_idx)

    sim.flush_until(6)
    stats = sim.stats.as_dict(num_frames=5)
    assert stats["sent"] == 6
    assert stats["delivered"] == 6
    assert stats["average_delay_frames"] == 1.0
    return stats


if __name__ == "__main__":
    print(run_demo())
