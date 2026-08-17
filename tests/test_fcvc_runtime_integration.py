import copy
import math
import unittest
from types import SimpleNamespace

import torch

from lib.test.tracker.entertrack import (
    EnTeRTrack,
    FCVC_PERSISTENT_STATE_REGISTRY,
    FrameTrackingResult,
)


class FakeMotionStateManager:
    def __init__(self):
        self.history = []

    def update_prediction_only(self, frame_id, predicted_bbox, max_score, apce,
                               response, image_size, remote_quality,
                               remote_weight_entropy, remote_max_weight,
                               valid_remote_count):
        record = {
            "frame_id": int(frame_id),
            "bbox": [float(v) for v in predicted_bbox],
            "score": float(max_score.detach().reshape(-1)[0].cpu().item())
                if torch.is_tensor(max_score) else float(max_score),
            "apce": float(apce.detach().reshape(-1)[0].cpu().item())
                if torch.is_tensor(apce) else float(apce),
        }
        self.history.append(record)
        return copy.deepcopy(record)


def make_tracker():
    tracker = object.__new__(EnTeRTrack)
    tracker.params = SimpleNamespace(search_factor=4.0, search_size=256)
    tracker.state = [0.0, 0.0, 20.0, 20.0]
    tracker.frame_id = 0
    tracker.z_dict1 = SimpleNamespace(tensors=torch.ones(1, 3, 128, 128))
    tracker.z_patch_arr = [[1, 2], [3, 4]]
    tracker.box_mask_z = torch.ones(1, 64, dtype=torch.bool)
    tracker.last_pcum_diagnostic = None
    tracker.c3r_last_frame_by_sender = {}
    tracker.c3r_message_accounting = SimpleNamespace(frames=[])
    tracker.temporal_gate_runtime = None
    tracker.mcr_manager = SimpleNamespace(history=[])
    tracker.mcr_updates_allowed = True
    tracker.motion_state_manager = FakeMotionStateManager()
    tracker.motion_state_log_enabled = True
    tracker.output_window = torch.ones(16, 16)
    tracker._pcum_diagnostic_hooks = SimpleNamespace(alignment={}, fusion={})
    tracker.debug = False
    tracker.save_all_boxes = False
    return tracker


def candidate(frame, bbox=None, score=0.5, apce=10.0):
    bbox = bbox or [float(frame), float(frame + 1), 20.0, 20.0]
    response = torch.full((1, 1, 16, 16), float(score))
    return {
        "output": {"target_bbox": list(bbox), "max_score": float(score)},
        "target_bbox": list(bbox),
        "prev_bbox": [float(frame - 1), float(frame), 20.0, 20.0],
        "crop_bbox": [float(frame - 1), float(frame), 20.0, 20.0],
        "resize_factor": 2.0,
        "search_factor": 4.0,
        "max_score": torch.tensor([float(score)]),
        "apce": torch.tensor([[float(apce)]]),
        "response": response,
        "out_dict": {"score_map": response.clone()},
        "pred_boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
        "x_patch_arr": [[frame]],
        "image": torch.zeros(32, 32, 3),
        "local_prompt": torch.full((1, 4), float(frame)),
        "aligned_prompt": None,
        "align_gate": None,
        "remote_states": None,
        "template_update_decision": frame % 2 == 0,
        "memory_update_decision": frame % 3 == 0,
    }


class FCVCRuntimeIntegrationTest(unittest.TestCase):
    def test_persistent_registry_completeness(self):
        names = {row["name"] for row in FCVC_PERSISTENT_STATE_REGISTRY}
        required = {
            "state", "frame_id", "z_dict1", "z_patch_arr", "box_mask_z",
            "_pcum_diagnostic_hooks", "last_pcum_diagnostic",
            "c3r_last_frame_by_sender", "c3r_message_accounting",
            "temporal_gate_runtime", "mcr_manager", "mcr_updates_allowed",
            "motion_state_manager", "output_window", "candidate.crop_bbox",
            "candidate.resize_factor", "candidate.search_factor",
            "candidate.max_score", "candidate.apce", "candidate.response",
            "candidate.out_dict", "candidate.local_prompt",
            "candidate.remote_states", "per-view tracker objects",
            "sender packet/bundle queue",
        }
        self.assertTrue(required.issubset(names))
        self.assertEqual(len(FCVC_PERSISTENT_STATE_REGISTRY), 25)

    def test_split_api_contract_and_report_state_immutability(self):
        tracker = make_tracker()
        pre = tracker.fcvc_persistent_state_digest()
        result = tracker.fcvc_predict_frame(
            local_candidate=candidate(1),
            collaborative_candidate=candidate(1, bbox=[999.0, 999.0, 1.0, 1.0]),
            debug_assertions=True,
        )
        self.assertIsInstance(result, FrameTrackingResult)
        self.assertEqual(tracker.fcvc_persistent_state_digest(), pre)
        report = tracker.fcvc_emit_report(result.reported_output,
                                          debug_assertions=True)
        self.assertEqual(report["target_bbox"], [999.0, 999.0, 1.0, 1.0])
        self.assertEqual(tracker.fcvc_persistent_state_digest(), pre)
        with self.assertRaisesRegex(AssertionError, "collaborative provenance"):
            tracker.fcvc_commit_state(
                result.reported_output, result.local_runtime_payload,
                debug_assertions=True)

    def test_local_payload_alias_audit(self):
        tracker = make_tracker()
        local = candidate(2)
        collab = candidate(2, bbox=[1000.0, -1000.0, 5.0, 5.0], score=1.0)
        result = tracker.fcvc_predict_frame(
            local_candidate=local,
            collaborative_candidate=collab,
            debug_assertions=True,
        )
        local["target_bbox"][0] = -12345.0
        local["response"].fill_(float("nan"))
        collab["target_bbox"][1] = 77777.0
        self.assertEqual(result.local_runtime_payload["target_bbox"],
                         [2.0, 3.0, 20.0, 20.0])
        self.assertTrue(torch.isfinite(result.local_runtime_payload["response"]).all())

    def test_ten_frame_state_identity_and_extreme_report_isolation(self):
        e0 = make_tracker()
        fcvc = make_tracker()
        extremes = [
            [10000.0, 10000.0, 10.0, 10.0],
            [0.0, 0.0, 100000.0, 100000.0],
            [5.0, 5.0, 1e-6, 1e-6],
            [8.0, 8.0, 3.0, 3.0],
            [9.0, 9.0, 4.0, 4.0],
            [-1000.0, 50.0, 5.0, 5.0],
            [50.0, -1000.0, 5.0, 5.0],
            [9999.0, -9999.0, 6.0, 6.0],
            [2.0, 3.0, 7.0, 7.0],
            [4.0, 5.0, 8.0, 8.0],
        ]
        for frame in range(10):
            e0.frame_id += 1
            fcvc.frame_id += 1
            local = candidate(frame, score=0.0 if frame == 3 else 1.0 if frame == 4 else 0.5)
            e0_output = e0._commit_candidate(copy.deepcopy(local))
            e0_output = e0._attach_motion_shadow_diagnostics(local, e0_output)
            collab = candidate(frame, bbox=extremes[frame],
                               score=0.0 if frame == 3 else 1.0)
            result = fcvc.fcvc_predict_frame(
                local_candidate=copy.deepcopy(local),
                collaborative_candidate=collab,
                debug_assertions=True,
            )
            fcvc_report = fcvc.fcvc_commit_frame_result(
                result, debug_assertions=True)
            self.assertEqual(e0.fcvc_persistent_state_digest(),
                             fcvc.fcvc_persistent_state_digest())
            self.assertEqual(e0.fcvc_next_crop_digest(),
                             fcvc.fcvc_next_crop_digest())
            self.assertEqual(e0.state, fcvc.state)
            self.assertEqual(e0.motion_state_manager.history,
                             fcvc.motion_state_manager.history)
            self.assertEqual(e0_output["motion_state_diagnostics"],
                             fcvc_report["motion_state_diagnostics"])
            self.assertEqual(result.local_runtime_payload["sender_bundle_source"],
                             "local")
            if frame not in ():
                self.assertEqual(fcvc_report["target_bbox"], extremes[frame])

    def test_invalid_collaborative_output_falls_back_to_local(self):
        tracker = make_tracker()
        tracker.frame_id += 1
        local = candidate(1)
        invalid = candidate(1, bbox=[0.0, 0.0, float("nan"), 4.0])
        result = tracker.fcvc_predict_frame(
            local_candidate=local,
            collaborative_candidate=invalid,
            debug_assertions=True,
        )
        report = tracker.fcvc_commit_frame_result(
            result, debug_assertions=True)
        self.assertEqual(report["target_bbox"], local["target_bbox"])
        self.assertEqual(tracker.state, local["target_bbox"])
        self.assertNotEqual(
            report["fcvc_safe_commit"]["report_fallback_reason"], "ok")

    def test_no_gt_runtime_signature_and_determinism(self):
        tracker_a = make_tracker()
        tracker_b = make_tracker()
        self.assertNotIn("gt", EnTeRTrack.fcvc_predict_frame.__code__.co_varnames)
        self.assertNotIn("gt", EnTeRTrack.fcvc_commit_state.__code__.co_varnames)
        digests_a = []
        digests_b = []
        for tracker, digests in ((tracker_a, digests_a), (tracker_b, digests_b)):
            for frame in range(4):
                tracker.frame_id += 1
                result = tracker.fcvc_predict_frame(
                    local_candidate=candidate(frame),
                    collaborative_candidate=candidate(
                        frame, bbox=[frame + 100.0, 0.0, 10.0, 10.0]),
                    debug_assertions=True,
                )
                tracker.fcvc_commit_frame_result(result, debug_assertions=True)
                digests.append(tracker.fcvc_persistent_state_digest())
        self.assertEqual(digests_a, digests_b)

    def test_multiview_stream_isolation_and_sender_local_provenance(self):
        streams = {
            (target, view): make_tracker()
            for target in ("md0001", "md0002")
            for view in ("A", "B", "C")
        }
        for frame in range(3):
            payloads = {}
            for stream_index, (key, tracker) in enumerate(streams.items()):
                tracker.frame_id += 1
                local = candidate(frame, bbox=[
                    float(frame + stream_index),
                    float(stream_index * 10 + ord(key[1])),
                    20.0, 20.0])
                result = tracker.fcvc_predict_frame(
                    local_candidate=local,
                    collaborative_candidate=candidate(
                        frame, bbox=[999.0, 999.0, 5.0, 5.0]),
                    debug_assertions=True,
                )
                payloads[key] = {
                    "source": result.local_runtime_payload["sender_bundle_source"],
                    "digest": result.local_runtime_payload["sender_source_digest"],
                }
                tracker.fcvc_commit_frame_result(result, debug_assertions=True)
            self.assertEqual(len({item["digest"] for item in payloads.values()}), 6)
            self.assertTrue(all(item["source"] == "local"
                                for item in payloads.values()))
        target_a_digest = streams[("md0001", "A")].fcvc_persistent_state_digest()
        target_b_digest = streams[("md0002", "A")].fcvc_persistent_state_digest()
        self.assertNotEqual(target_a_digest, target_b_digest)

    def test_default_off_e0_c1_commit_compatibility(self):
        e0 = make_tracker()
        c1 = make_tracker()
        local = candidate(1)
        c1_candidate = candidate(1, bbox=[30.0, 31.0, 20.0, 20.0], score=0.9)
        self.assertEqual(e0._commit_candidate(copy.deepcopy(local)),
                         local["output"])
        self.assertEqual(e0.state, local["target_bbox"])
        self.assertEqual(c1._commit_candidate(copy.deepcopy(c1_candidate)),
                         c1_candidate["output"])
        self.assertEqual(c1.state, c1_candidate["target_bbox"])


if __name__ == "__main__":
    unittest.main()
