import copy
import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lib.config.entertrack.config import cfg, update_config_from_file
from lib.models.entertrack.c3r import C3R, C3R_PACKET_BYTES, CommunicationPerturbation
from lib.test.evaluation.tracker import Tracker, trackerlist
from lib.test.evaluation.run_id import (
    parse_run_id_argument,
    read_run_identity,
    reserve_run_directory,
    result_directory,
    result_directory_name,
    validate_run_id,
)
from lib.test.utils.c3r_inference import (
    C3RFrameExchange,
    C3RPacketRecord,
    C3RReceiverContext,
    build_packet_record,
    collaborate_local_candidate,
    diagnostic_row,
    target_session_hash,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments" / "entertrack"
HOLDOUT = ROOT / "lib/train/data_specs/threemdot/c3r_f0_holdout.txt"
REGISTRY = ROOT / "output/multi_agent_collaboration_clean/formal/evaluation_registry.csv"
FEAT_LEN = 16 * 16


def resolved(name):
    update_config_from_file(EXPERIMENTS / (name + ".yaml"))
    return copy.deepcopy(cfg)


def synthetic_candidate(seed=0):
    generator = torch.Generator().manual_seed(seed)
    feature = torch.randn(1, 4 + FEAT_LEN, 8, generator=generator)
    score = torch.sigmoid(torch.randn(1, 1, 16, 16, generator=generator))
    return {
        "out_dict": {
            "backbone_feat": feature,
            "score_map": score,
        },
        "target_bbox": [16.0, 12.0, 20.0, 18.0],
        "prev_bbox": [15.0, 12.0, 20.0, 18.0],
        "image": np.zeros((64, 80, 3), dtype=np.uint8),
    }


def fake_forward_head(feature, _):
    search = feature[:, -FEAT_LEN:]
    score = torch.sigmoid(search.mean(dim=-1)).reshape(1, 1, 16, 16)
    return {"score_map": score, "backbone_feat": feature}


class FormalConfigTest(unittest.TestCase):
    def test_three_configs_and_holdout(self):
        e0 = resolved("entertrack_c3r_e0_f0")
        c0 = resolved("entertrack_c3r_c0_f0")
        c1 = resolved("entertrack_c3r_c1_f0")
        self.assertFalse(e0.MODEL.C3R.ENABLED)
        self.assertTrue(c0.MODEL.C3R.ENABLED)
        self.assertTrue(c1.MODEL.C3R.ENABLED)
        self.assertEqual(c0.MODEL.C3R.VARIANT, "c0")
        self.assertEqual(c1.MODEL.C3R.VARIANT, "c1")
        for field in (
            "NUM_PROMPTS", "MESSAGE_DIM", "PACKET_VERSION", "MAX_GATE",
            "PEER_NORM_CAP", "AGGREGATE_NORM_CAP", "MAX_AGE_INTERVALS",
        ):
            self.assertEqual(getattr(c0.MODEL.C3R, field), getattr(c1.MODEL.C3R, field))
        self.assertEqual(c0.DATA.TRAIN.SPLIT_FILE, c1.DATA.TRAIN.SPLIT_FILE)
        self.assertEqual(c0.TRAIN.C3R.SEED, c1.TRAIN.C3R.SEED)
        self.assertEqual(c0.TRAIN.EPOCH, c1.TRAIN.EPOCH)
        sequences = HOLDOUT.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(sequences), 15)
        targets = {}
        for sequence in sequences:
            target, view = sequence.rsplit("-", 1)
            targets.setdefault(target, set()).add(int(view))
        self.assertEqual(len(targets), 5)
        self.assertTrue(all(views == {1, 2, 3} for views in targets.values()))


class RunIdContractTest(unittest.TestCase):
    def test_numeric_and_string_paths(self):
        self.assertEqual(parse_run_id_argument("14"), 14)
        self.assertEqual(parse_run_id_argument("c3r_c1_f0"), "c3r_c1_f0")
        self.assertEqual(result_directory_name("cfg", 0), "cfg_000")
        self.assertEqual(result_directory_name("cfg", 14), "cfg_014")
        self.assertEqual(result_directory_name("cfg", "c3r_c1_f0"), "cfg_c3r_c1_f0")

    def test_invalid_and_empty_strings(self):
        for value in ("", "../x", "a/b", "a b", "/tmp/x", "x;rm"):
            with self.assertRaises((TypeError, ValueError)):
                validate_run_id(value)

    def test_reservation_duplicate_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "entertrack" / "cfg_c3r_c1_f0"
            identity = {
                "tracker_name": "entertrack",
                "parameter_name": "cfg",
                "dataset_name": "threemdot_cv",
                "runid": "c3r_c1_f0",
            }
            marker = reserve_run_directory(path, identity)
            self.assertTrue(marker.is_file())
            self.assertEqual(read_run_identity(path)["runid"], "c3r_c1_f0")
            with self.assertRaises(FileExistsError):
                reserve_run_directory(path, identity)

    def test_registry_path_round_trip(self):
        with REGISTRY.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            sorted(row["experiment_id"] for row in rows),
            sorted("{}_F{}".format(system, fold)
                   for fold in range(5) for system in ("C0", "C1", "E0")),
        )
        for row in rows:
            expected = result_directory(
                ROOT / "output/test/tracking_results",
                "entertrack", row["config"], row["runid"])
            self.assertEqual(expected, Path(row["output_dir"]))

    def test_legacy_tracker_wrapper_and_numeric_result_path(self):
        tracker = Tracker(
            "entertrack", "entertrack_c3r_e0_f0", "threemdot_cv", 14)
        self.assertTrue(tracker.results_dir.endswith(
            "entertrack/entertrack_c3r_e0_f0_014"))
        params = tracker.get_parameters()
        self.assertEqual(params.run_id, 14)
        self.assertFalse(params.no_gt_inference)
        self.assertEqual(
            trackerlist(
                "entertrack", "entertrack_c3r_e0_f0", "threemdot_cv",
                run_ids="c3r_e0_f0")[0].run_id,
            "c3r_e0_f0",
        )


class PacketContextIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.c0 = C3R(token_dim=8, variant="c0")
        self.c1 = C3R(token_dim=8, variant="c1")
        self.target = "md3005"
        self.frame = 3
        self.timestamp = 99

    def record(self, model, sender, seed=None, frame=None, timestamp=None):
        candidate = synthetic_candidate(sender if seed is None else seed)
        return build_packet_record(
            model,
            feat_len_s=FEAT_LEN,
            candidate=candidate,
            target_id=self.target,
            sender_id=sender,
            frame_id=self.frame if frame is None else frame,
            timestamp_ms=self.timestamp if timestamp is None else timestamp,
            image_height=64,
            image_width=80,
        )

    def test_packet_context_and_two_remote_aggregation(self):
        records = tuple(self.record(self.c1, sender) for sender in (2, 0, 1))
        exchange = C3RFrameExchange(
            self.target, target_session_hash(self.target), self.frame,
            self.timestamp, records)
        self.assertEqual(tuple(record.sender_id for record in exchange.records), (0, 1, 2))
        self.assertEqual(exchange.sender_ids_for(0), (1, 2))
        self.assertTrue(all(len(packet) == 320 for packet in exchange.packets_for(0)))
        context = C3RReceiverContext.for_frame(
            self.target, 0, self.frame, last_frame_by_sender={})
        result = collaborate_local_candidate(
            self.c1, fake_forward_head, FEAT_LEN, synthetic_candidate(0),
            exchange.packets_for(0), context)
        self.assertTrue(result.used_remote)
        self.assertEqual(result.collaboration["accepted_count"], 2)
        self.assertTrue(torch.isfinite(result.output["score_map"]).all())
        row = diagnostic_row(
            self.target, 0, context, 1, 2, result.collaboration)
        self.assertFalse(row["uses_gt"])
        self.assertEqual(row["packet_bytes"], C3R_PACKET_BYTES)
        self.assertEqual(row["received_bytes"], 640)

    def test_remote_missing_and_replay_identity(self):
        candidate = synthetic_candidate(0)
        context = C3RReceiverContext.for_frame(
            self.target, 0, self.frame, last_frame_by_sender={})
        missing = collaborate_local_candidate(
            self.c1, fake_forward_head, FEAT_LEN, candidate, (), context)
        self.assertFalse(missing.used_remote)
        self.assertIs(missing.output, candidate["out_dict"])

        record = self.record(self.c1, 1)
        first = collaborate_local_candidate(
            self.c1, fake_forward_head, FEAT_LEN, candidate, (record.payload,), context)
        self.assertTrue(first.used_remote)
        replay = collaborate_local_candidate(
            self.c1, fake_forward_head, FEAT_LEN, candidate, (record.payload,), context)
        self.assertFalse(replay.used_remote)
        self.assertIs(replay.output, candidate["out_dict"])

    def test_delay_wrong_target_and_frame_isolation(self):
        delayed = self.record(self.c1, 1, frame=2, timestamp=66)
        exchange = C3RFrameExchange(
            self.target, target_session_hash(self.target), 3, 99, (delayed,))
        context = C3RReceiverContext.for_frame(
            self.target, 0, 3, last_frame_by_sender={})
        result = collaborate_local_candidate(
            self.c1, fake_forward_head, FEAT_LEN, synthetic_candidate(0),
            exchange.packets_for(0), context)
        self.assertTrue(result.used_remote)
        with self.assertRaises(ValueError):
            C3RFrameExchange(
                "md3019", target_session_hash("md3019"), 3, 99, (delayed,))
        future = self.record(self.c1, 2, frame=4, timestamp=132)
        with self.assertRaises(ValueError):
            C3RFrameExchange(
                self.target, target_session_hash(self.target), 3, 99, (future,))

    def test_faults_one_bad_conflict_and_schema_parity(self):
        candidate = synthetic_candidate(0)
        feature = candidate["out_dict"]["backbone_feat"][:, -FEAT_LEN:]
        response = candidate["out_dict"]["score_map"]
        messages = self.c1.encoder(
            feature.repeat(2, 1, 1), response.repeat(2, 1, 1, 1),
            torch.tensor([[0.5, 0.5, 0.2, 0.2]]).repeat(2, 1), None,
            [1, 2], [target_session_hash(self.target)] * 2,
            [self.frame] * 2, [self.timestamp] * 2)
        faulty = CommunicationPerturbation(enabled=False).apply(
            messages, frame_interval_ms=33, force_fault="one_bad")
        self.assertEqual([item.construction_label for item in faulty], [1, 0])
        context_a = C3RReceiverContext.for_frame(
            self.target, 0, self.frame, last_frame_by_sender={})
        context_b = C3RReceiverContext.for_frame(
            self.target, 0, self.frame, last_frame_by_sender={})
        forward = collaborate_local_candidate(
            self.c1, fake_forward_head, FEAT_LEN, candidate,
            [self.c1.codec.serialize(item) for item in faulty], context_a)
        reverse = collaborate_local_candidate(
            self.c1, fake_forward_head, FEAT_LEN, candidate,
            [self.c1.codec.serialize(item) for item in reversed(faulty)], context_b)
        self.assertTrue(torch.equal(forward.output["score_map"], reverse.output["score_map"]))
        self.assertEqual(
            self.c0.encoder.message_contract(),
            self.c1.encoder.message_contract())
        self.assertEqual(self.c1.encoder.message_contract()["serialized_payload_bytes"], 320)

    def test_no_gt_api_and_e0_disabled(self):
        signature = inspect.signature(build_packet_record)
        forbidden = ("gt", "visible", "annotation", "oracle", "iou")
        names = " ".join(signature.parameters)
        self.assertFalse(any(token in names.lower() for token in forbidden))
        e0 = resolved("entertrack_c3r_e0_f0")
        self.assertFalse(e0.MODEL.C3R.ENABLED)
        self.assertFalse(e0.TEST.C3R.ENABLED)


if __name__ == "__main__":
    unittest.main()
