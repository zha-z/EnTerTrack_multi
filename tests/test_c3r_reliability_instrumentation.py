import copy
import inspect
import unittest

import numpy as np
import torch

from lib.models.entertrack.c3r import C3R, C3R_RELIABILITY_INPUT_NAMES
from lib.test.utils.c3r_inference import (
    C3RFrameExchange,
    C3RReceiverContext,
    build_packet_record,
    collaborate_local_candidate,
    target_session_hash,
)


FEAT_LEN = 16 * 16


def candidate(seed):
    generator = torch.Generator().manual_seed(seed)
    feature = torch.randn(1, 4 + FEAT_LEN, 8, generator=generator)
    score = torch.sigmoid(torch.randn(1, 1, 16, 16, generator=generator))
    return {
        "out_dict": {"backbone_feat": feature, "score_map": score},
        "target_bbox": [16.0, 12.0, 20.0, 18.0],
        "prev_bbox": [15.0, 12.0, 20.0, 18.0],
        "image": np.zeros((64, 80, 3), dtype=np.uint8),
    }


def forward_head(feature, _):
    search = feature[:, -FEAT_LEN:]
    score = torch.sigmoid(search.mean(dim=-1)).reshape(1, 1, 16, 16)
    box = torch.stack((
        score.mean(), score.amax(), score.amin(), score.std(unbiased=False)
    )).reshape(1, 1, 4)
    return {"score_map": score, "backbone_feat": feature, "pred_boxes": box}


class ReliabilityInstrumentationIdentityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260716)
        self.model = C3R(token_dim=8, variant="c1")
        self.target = "md3005"
        self.frame = 3
        self.timestamp = 99
        records = []
        for sender in (1, 2):
            records.append(build_packet_record(
                self.model, FEAT_LEN, candidate(sender), self.target,
                sender, self.frame, self.timestamp, 64, 80))
        self.exchange = C3RFrameExchange(
            self.target, target_session_hash(self.target), self.frame,
            self.timestamp, tuple(records))

    def _run(self, enabled):
        replay_state = {}
        context = C3RReceiverContext.for_frame(
            self.target, 0, self.frame, last_frame_by_sender=replay_state)
        result = collaborate_local_candidate(
            self.model, forward_head, FEAT_LEN, candidate(0),
            self.exchange.packets_for(0), context,
            instrumentation=enabled)
        return result, replay_state

    def test_default_is_disabled(self):
        parameter = inspect.signature(collaborate_local_candidate).parameters[
            "instrumentation"]
        self.assertFalse(parameter.default)
        result, _ = self._run(False)
        self.assertNotIn("instrumentation_source_rows", result.collaboration)
        self.assertNotIn("instrumentation_aggregate", result.collaboration)

    def test_enabled_is_bitwise_behavior_identity(self):
        disabled, disabled_state = self._run(False)
        enabled, enabled_state = self._run(True)

        checks = {
            "bbox": torch.equal(
                disabled.output["pred_boxes"], enabled.output["pred_boxes"]),
            "score_map": torch.equal(
                disabled.output["score_map"], enabled.output["score_map"]),
            "final_feature": torch.equal(
                disabled.output["backbone_feat"], enabled.output["backbone_feat"]),
            "gate": torch.equal(
                disabled.collaboration["gates"], enabled.collaboration["gates"]),
            "packet": self.exchange.packets_for(0) == self.exchange.packets_for(0),
            "tracker_state": disabled_state == enabled_state,
        }
        self.assertEqual(checks, {name: True for name in checks})
        self.assertTrue(torch.equal(
            disabled.collaboration["search_tokens"],
            enabled.collaboration["search_tokens"]))
        self.assertTrue(torch.equal(
            disabled.collaboration["gate_logits"],
            enabled.collaboration["gate_logits"]))

        rows = enabled.collaboration["instrumentation_source_rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(C3R_RELIABILITY_INPUT_NAMES), 10)
        for row, gate in zip(rows, enabled.collaboration["gates"]):
            self.assertEqual(len(row["reliability_input_raw"]), 10)
            self.assertEqual(len(row["reliability_input_normalized"]), 10)
            self.assertEqual(len(row["hidden_pre_activation"]), 32)
            self.assertEqual(len(row["hidden_post_activation"]), 32)
            self.assertEqual(row["final_gate"], float(gate.detach().cpu().item()))
            self.assertEqual(row["final_gate"], row["diagnostic_recomputed_gate"])
            self.assertTrue(all(not torch.is_tensor(value) for value in row.values()))

        aggregate = enabled.collaboration["instrumentation_aggregate"]
        self.assertEqual(aggregate["sender_count"], 2)
        self.assertFalse(aggregate["abnormal_aggregate_residual"])

    def test_instrumentation_does_not_mutate_packet(self):
        before = tuple(bytes(packet) for packet in self.exchange.packets_for(0))
        self._run(True)
        after = tuple(bytes(packet) for packet in self.exchange.packets_for(0))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
