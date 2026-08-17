import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from lib.config.entertrack.config import get_default_config, update_config_from_file
from lib.models.entertrack.c3r import (
    C3R,
    C3RMessage,
    CompactMessageEncoder,
)
from lib.test.tracker.entertrack import EnTeRTrack
from lib.test.utils.c3r_inference import C3RReceiverContext, target_session_hash
from lib.models.entertrack.temporal_gate import (
    TEMPORAL_GATE_PARAMETER_COUNT,
    TemporalGate,
    TemporalGateRuntime,
    temporal_utility_to_gate,
)
from lib.train.dataset.c3r_temporal_gate import (
    C3RTemporalGateDataset,
    assert_prediction_row,
    audit_split,
    build_fold1_inner_split,
    read_id_file,
)
from tracking.evaluate_c3r_temporal_gate_closed_loop import (
    assert_prediction_only,
    evaluate_joined_rows,
    run_prediction_only,
)
from tracking.evaluate_c3r_temporal_gate_offline import (
    GATE1_V2_THRESHOLDS,
    evaluate,
)
from tracking.generate_c3r_temporal_gate_rollouts import (
    PredictionTableWriter,
    generate_fold1_inner_split,
    sha256_file,
    write_label_table,
)
from tracking.train_c3r_temporal_gate import (
    build_optimizer,
    freeze_audit,
    smooth_l1_utility_loss,
)


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "lib/train/data_specs/threemdot"


def make_message(sender_id=1, frame_id=3, timestamp_ms=99,
                 sequence_hash=1234):
    prompt = torch.linspace(-1.0, 1.0, 256).reshape(4, 64)
    quantized, scales, reconstructed = CompactMessageEncoder.quantize(prompt)
    return C3RMessage(
        sender_id=sender_id,
        sequence_hash=sequence_hash,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        bbox=torch.tensor([0.5, 0.5, 0.2, 0.3]),
        bbox_delta=torch.tensor([0.01, -0.01, 0.0, 0.0]),
        quality=torch.tensor([0.8, 0.7, 0.2, 0.3]),
        scales=scales,
        quantized_prompt=quantized,
        prompt=reconstructed,
    )


def labeled_row(target="md3008", receiver=0, sender=1, frame=1,
                delta_diou=0.25, features=None, accepted=True):
    if features is None:
        features = [0.01 * (frame + index) for index in range(10)]
    return {
        "target_id": target,
        "receiver_id": receiver,
        "sender_id": sender,
        "frame_id": frame,
        "normalized_features": features,
        "model_input_fields": ["normalized_features"],
        "packet_accepted": accepted,
        "uses_gt_for_features": False,
        "delta_diou": delta_diou,
        "label_status": "valid",
    }


def prediction_row(frame=1):
    row = labeled_row(frame=frame)
    row.pop("delta_diou")
    row.pop("label_status")
    row.pop("uses_gt_for_features")
    row["uses_gt"] = False
    return row


class TemporalGateModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260718)
        self.model = TemporalGate()

    def test_output_shape_range_and_exact_parameter_count(self):
        gate, utility = self.model(torch.randn(4, 8, 10))
        self.assertEqual(tuple(gate.shape), (4, 1))
        self.assertEqual(tuple(utility.shape), (4, 1))
        self.assertTrue(torch.isfinite(gate).all())
        self.assertTrue(bool(((gate >= 0.0) & (gate <= 0.25)).all()))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            TEMPORAL_GATE_PARAMETER_COUNT,
        )

    def test_v2_fixed_utility_mapping_zero_monotonic_and_bounded(self):
        utility = torch.tensor([
            -100.0, -1.0, 0.0, 0.01, 0.5, 1.0, 100.0])
        gate = temporal_utility_to_gate(utility)
        self.assertTrue(torch.equal(gate[:3], torch.zeros(3)))
        self.assertTrue(bool(torch.all(gate[3:-1][1:] > gate[3:-1][:-1])))
        self.assertTrue(bool(((gate >= 0.0) & (gate <= 0.25)).all()))
        self.assertLessEqual(float(gate.max()), 0.25)

    def test_causality_future_cannot_change_present(self):
        history = torch.randn(8, 10)
        gate_t, _ = self.model(history[:7])
        changed = history.clone()
        changed[7] = 1000.0
        gate_t_changed, _ = self.model(changed[:7])
        self.assertTrue(torch.equal(gate_t, gate_t_changed))

    def test_prefix_and_w8_truncation(self):
        runtime = TemporalGateRuntime(copy.deepcopy(self.model))
        values = [torch.full((10,), float(index) / 20.0) for index in range(1, 10)]
        first = runtime.gate_for("md3008", 0, 1, 1, values[0])
        direct, _ = runtime.model(values[0].reshape(1, 10))
        self.assertTrue(torch.equal(first.cpu(), direct.reshape(()).cpu()))
        for frame, value in enumerate(values[1:], start=2):
            observed = runtime.gate_for("md3008", 0, 1, frame, value)
        expected, _ = runtime.model(torch.stack(values[-8:]))
        self.assertTrue(torch.equal(observed.cpu(), expected.reshape(()).cpu()))
        self.assertEqual(len(runtime.history("md3008", 0, 1)), 8)

    def test_target_receiver_sender_and_gap_reset_isolation(self):
        runtime = TemporalGateRuntime(self.model)
        vector = torch.arange(10, dtype=torch.float32) / 10.0
        runtime.gate_for("md3008", 0, 1, 1, vector)
        runtime.gate_for("md3008", 1, 1, 1, vector)
        runtime.gate_for("md3008", 0, 2, 1, vector)
        runtime.gate_for("md3013", 0, 1, 1, vector)
        self.assertEqual(len(runtime.keys), 4)
        runtime.gate_for("md3008", 0, 1, 3, vector)
        self.assertEqual(len(runtime.history("md3008", 0, 1)), 1)
        self.assertEqual(len(runtime.history("md3008", 1, 1)), 1)
        runtime.mark_gap("md3008", 1, 1)
        self.assertEqual(runtime.history("md3008", 1, 1), ())

    def test_input_is_detached_and_float32(self):
        runtime = TemporalGateRuntime(self.model)
        source = torch.randn(10, dtype=torch.float64, requires_grad=True)
        gate = runtime.gate_for("md3008", 0, 1, 1, source)
        stored = runtime.history("md3008", 0, 1)[0]
        self.assertEqual(stored.dtype, torch.float32)
        self.assertFalse(stored.requires_grad)
        self.assertIsNone(stored.grad_fn)
        self.assertFalse(gate.requires_grad)
        self.assertIsNone(source.grad)

    def test_deterministic_replay(self):
        state = copy.deepcopy(self.model.state_dict())
        outputs = []
        for _ in range(2):
            model = TemporalGate()
            model.load_state_dict(state)
            runtime = TemporalGateRuntime(model)
            outputs.append(torch.stack([
                runtime.gate_for("md3008", 0, 1, frame,
                                 torch.full((10,), frame / 10.0))
                for frame in range(1, 10)
            ]))
        self.assertTrue(torch.equal(outputs[0], outputs[1]))


class TemporalGateDatasetAndSplitTest(unittest.TestCase):
    def test_gt_schema_and_identity_input_rejection(self):
        row = prediction_row()
        assert_prediction_row(row)
        for field in ("gt_bbox", "iou", "label", "visibility"):
            bad = dict(row)
            bad[field] = 0
            with self.assertRaises(ValueError):
                assert_prediction_row(bad)
        bad = labeled_row()
        bad["model_input_fields"] = ["normalized_features", "target_id"]
        with self.assertRaises(ValueError):
            C3RTemporalGateDataset([bad])

    def test_contiguous_prefix_sender_and_gap_guards(self):
        rows = [
            labeled_row(frame=1), labeled_row(frame=2),
            labeled_row(frame=4),
            labeled_row(sender=2, frame=1),
            labeled_row(sender=2, frame=2, accepted=False),
            labeled_row(sender=2, frame=3),
        ]
        dataset = C3RTemporalGateDataset(rows)
        lengths = [item["history"].shape[0] for item in dataset]
        self.assertEqual(lengths, [1, 2, 1, 1, 1])
        self.assertTrue(all(tuple(item["history"].shape[1:]) == (10,)
                            for item in dataset))

    def test_continuous_delta_diou_keeps_zero_and_ignores_aux_delta_iou(self):
        negative = labeled_row(frame=1, delta_diou=-0.4)
        zero = labeled_row(frame=2, delta_diou=0.0)
        positive = labeled_row(frame=3, delta_diou=0.6)
        negative["delta_iou"] = 1.0
        zero["delta_iou"] = -1.0
        positive["delta_iou"] = 0.0
        dataset = C3RTemporalGateDataset([negative, zero, positive])
        self.assertTrue(torch.allclose(
            torch.stack([item["delta_diou"] for item in dataset]),
            torch.tensor([-0.4, 0.0, 0.6])))

    def test_frozen_fold1_inner_split_and_outer_leakage(self):
        outer_train = read_id_file(str(SPECS / "c3r_f1_train.txt"))
        holdout = read_id_file(str(SPECS / "c3r_f1_holdout.txt"))
        train, dev, ranked = build_fold1_inner_split(outer_train)
        self.assertEqual(
            ranked[:4], ["md3019", "md3005", "md3051", "md3044"])
        self.assertEqual(train, read_id_file(str(
            SPECS / "c3r_f1_temporal_v2_inner_train.txt")))
        self.assertEqual(dev, read_id_file(str(
            SPECS / "c3r_f1_temporal_v2_inner_dev.txt")))
        audit = audit_split(train, dev, holdout)
        self.assertTrue(audit["target_disjoint"])
        self.assertTrue(audit["view_disjoint"])
        with self.assertRaises(AssertionError):
            audit_split(train + [holdout[0]], dev, holdout)

    def test_prediction_digest_precedes_separate_label_table(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "prediction.jsonl"
            writer = PredictionTableWriter(str(prediction))
            writer.append(prediction_row())
            manifest = writer.close()
            self.assertEqual(manifest["sha256"], sha256_file(str(prediction)))
            label_path = Path(directory) / "labels.jsonl"
            label = labeled_row()
            label["uses_gt_for_features"] = False
            joined = write_label_table(
                str(prediction), manifest["sha256"], [label], str(label_path))
            self.assertEqual(joined["prediction_sha256"], manifest["sha256"])
            with self.assertRaises(RuntimeError):
                write_label_table(str(prediction), "0" * 64, [label],
                                  str(Path(directory) / "bad.jsonl"))

    def test_fold1_split_manifest_uses_only_id_files(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "inner_train.txt"
            dev_path = Path(directory) / "inner_dev.txt"
            manifest_path = Path(directory) / "manifest.json"
            report = generate_fold1_inner_split(
                train_path=SPECS / "c3r_f1_train.txt",
                holdout_path=SPECS / "c3r_f1_holdout.txt",
                train_output=train_path,
                dev_output=dev_path,
                manifest_output=manifest_path,
            )
            self.assertTrue(report["id_only_generation"])
            self.assertFalse(report["images_gt_predictions_metrics_read"])
            self.assertEqual(report["inner_dev_target_count"], 4)
            self.assertEqual(report["inner_train_sha256"],
                             sha256_file(str(train_path)))
            self.assertEqual(report["inner_dev_sha256"],
                             sha256_file(str(dev_path)))


class TemporalGateFreezeAndIdentityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77)
        self.tokens = torch.randn(1, 32, 8)
        self.response = torch.sigmoid(torch.randn(1, 1, 4, 8))
        self.context = dict(
            receiver_id=0, sequence_hash=1234, local_frame_id=3,
            local_timestamp_ms=99, frame_interval_ms=33,
            last_frame_by_sender={},
        )

    def test_optimizer_contains_only_gru_and_output(self):
        model = TemporalGate()
        optimizer = build_optimizer(model)
        audit = freeze_audit(model, optimizer)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["count"], 1361)
        self.assertTrue(all(row["name"].startswith(("gru.", "output."))
                            for row in audit["parameters"]))
        _, raw_utility = model(torch.randn(4, 8, 10))
        loss = smooth_l1_utility_loss(
            raw_utility, torch.tensor([-0.4, 0.0, 0.2, 0.8]))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None
                            for parameter in model.parameters()))

    def test_default_off_c1_is_bitwise_identical(self):
        model_a = C3R(token_dim=8, variant="c1")
        model_b = copy.deepcopy(model_a)
        message = make_message()
        first = model_a.collaborate(
            self.tokens, self.response, [message.detached_copy()], **self.context)
        context = dict(self.context)
        context["last_frame_by_sender"] = {}
        second = model_b.collaborate(
            self.tokens, self.response, [message.detached_copy()],
            gate_provider=None, **context)
        for field in ("search_tokens", "gates", "gate_logits", "reliability_inputs"):
            self.assertTrue(torch.equal(first[field], second[field]), field)
        self.assertEqual(first["accepted_sender_ids"], second["accepted_sender_ids"])

    def test_sidecar_changes_neither_packet_bytes_adapter_nor_fusion(self):
        c3r = C3R(token_dim=8, variant="c1")
        message = make_message()
        packet_before = c3r.codec.serialize(message)
        remote = message.prompt.unsqueeze(0)
        adapter_before = c3r.adapter(self.tokens, remote).detach().clone()
        local = self.tokens.detach().clone()
        peer = adapter_before.detach().clone()
        fused_before, _ = c3r.fusion(local, [peer], [torch.tensor(0.1)])
        model = TemporalGate()
        runtime = TemporalGateRuntime(model)
        runtime.gate_for("md3008", 0, 1, 1, torch.randn(10))
        packet_after = c3r.codec.serialize(message)
        adapter_after = c3r.adapter(self.tokens, remote).detach().clone()
        fused_after, _ = c3r.fusion(local, [peer], [torch.tensor(0.1)])
        self.assertEqual(packet_before, packet_after)
        self.assertTrue(torch.equal(adapter_before, adapter_after))
        self.assertTrue(torch.equal(fused_before, fused_after))

    def test_gate_override_only_replaces_gate_values(self):
        c3r = C3R(token_dim=8, variant="c1")
        result = c3r.collaborate(
            self.tokens, self.response, [make_message()],
            gate_provider=lambda sender, vector: torch.tensor(0.2),
            **self.context)
        self.assertTrue(torch.equal(result["gates"], torch.tensor([0.2])))
        self.assertEqual(result["accepted_sender_ids"], [1])

    def test_base_checkpoint_digest_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.pth"
            base.write_bytes(b"immutable-base-checkpoint")
            before = hashlib.sha256(base.read_bytes()).hexdigest()
            sidecar = Path(directory) / "temporal.pth"
            torch.save({"state_dict": TemporalGate().state_dict()}, sidecar)
            after = hashlib.sha256(base.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertNotEqual(base, sidecar)

    def test_default_off_config(self):
        config = get_default_config()
        update_config_from_file(
            ROOT / "experiments/entertrack/entertrack_c3r_temporal_gate_v2_f1.yaml",
            base_cfg=config)
        self.assertFalse(config.MODEL.TEMPORAL_GATE.ENABLED)
        self.assertFalse(config.TRAIN.TEMPORAL_GATE.ENABLED)
        self.assertFalse(config.TEST.TEMPORAL_GATE.ENABLED)
        self.assertEqual(config.MODEL.TEMPORAL_GATE.INPUT_DIM, 10)
        self.assertEqual(config.MODEL.TEMPORAL_GATE.HIDDEN_DIM, 16)
        self.assertEqual(config.MODEL.TEMPORAL_GATE.WINDOW, 8)
        self.assertEqual(config.MODEL.TEMPORAL_GATE.MAX_GATE, 0.25)
        self.assertEqual(config.MODEL.TEMPORAL_GATE.GATE_MAPPING, "relu_tanh")
        self.assertEqual(config.TRAIN.TEMPORAL_GATE.TARGET, "delta_diou")
        self.assertEqual(config.TRAIN.TEMPORAL_GATE.LOSS, "smooth_l1")
        self.assertEqual(config.TRAIN.TEMPORAL_GATE.SMOOTH_L1_BETA, 1.0)
        self.assertFalse(
            config.TEST.TEMPORAL_GATE.COUNTERFACTUAL_DIAGNOSTICS)


class TemporalGateCounterfactualDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _tracker(diagnostics):
        torch.manual_seed(112)
        tracker = EnTeRTrack.__new__(EnTeRTrack)
        tracker.c3r_enabled = True
        tracker.temporal_gate_counterfactual_diagnostics = bool(diagnostics)
        tracker._temporal_gate_backbone_forward_count = 7
        tracker.state = [10.0, 20.0, 30.0, 40.0]
        tracker.frame_id = 3
        tracker.c3r_last_frame_by_sender = {}
        c3r = C3R(token_dim=8, variant="c1")

        def forward_head(feature, _):
            return {"score_map": torch.sigmoid(feature[:, -1:, :4, None])}

        tracker.network = SimpleNamespace(
            c3r=c3r, feat_len_s=32, forward_head=forward_head)

        def candidate_from_head(local_candidate, head_output):
            candidate = dict(local_candidate)
            sender_ids = tuple(head_output["c3r"]["accepted_sender_ids"])
            offset = 0.1 * sum(int(value) for value in sender_ids)
            candidate["target_bbox"] = [
                float(value) + offset
                for value in local_candidate["target_bbox"]
            ]
            candidate["output"] = {
                "target_bbox": list(candidate["target_bbox"])}
            candidate["pred_boxes"] = torch.tensor(
                candidate["target_bbox"], dtype=torch.float32).reshape(1, 4)
            candidate["response"] = head_output["score_map"]
            return candidate

        tracker._c3r_candidate_from_head = candidate_from_head
        return tracker

    @staticmethod
    def _inputs():
        sequence_hash = target_session_hash("md3013")
        local_candidate = {
            "out_dict": {
                "backbone_feat": torch.randn(1, 40, 8),
                "score_map": torch.sigmoid(torch.randn(1, 1, 4, 8)),
            },
            "target_bbox": [10.0, 20.0, 30.0, 40.0],
            "pred_boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "response": torch.sigmoid(torch.randn(1, 1, 4, 8)),
            "output": {"target_bbox": [10.0, 20.0, 30.0, 40.0]},
        }
        packets = (
            make_message(sender_id=1, sequence_hash=sequence_hash),
            make_message(sender_id=2, sequence_hash=sequence_hash),
        )
        context = C3RReceiverContext(
            target_id="md3013", receiver_id=0,
            sequence_hash=sequence_hash, frame_id=3, timestamp_ms=99,
            frame_interval_ms=33, last_frame_by_sender={})
        return local_candidate, packets, context

    def test_sender_only_gate_and_packet_provenance_are_proven(self):
        tracker = self._tracker(diagnostics=True)
        local, packets, context = self._inputs()
        result = tracker.c3r_counterfactual_candidates(
            local, packets, context)
        behavior = dict(local)
        behavior["target_bbox"] = [11.0, 21.0, 31.0, 41.0]
        tracker._audit_behavior_candidate_submission(result, behavior)
        branches = result["diagnostics"]["branches"]
        for sender_id in (1, 2):
            trace = branches["sender{}_only".format(sender_id)]
            self.assertEqual(trace["executed_gates"], [
                {"sender_id": sender_id, "gate": 0.25}])
            self.assertEqual(
                trace["sender_provenance"][0]["sender_id"], sender_id)
            self.assertEqual(
                trace["sender_provenance"][0]["packet_bytes"], 320)
            self.assertEqual(
                len(trace["sender_provenance"][0]["payload_sha256"]), 64)
            saved = tracker._counterfactual_source_diagnostic_fields(
                result, sender_id)
            self.assertEqual(
                saved["counterfactual_sender_only_executed_gate"], 0.25)
            self.assertEqual(
                saved["counterfactual_sender_provenance_id"], sender_id)
            self.assertTrue(
                saved["counterfactual_only_frozen_c1_behavior_submitted"])

    def test_four_branches_share_pre_state_and_do_not_mutate_state(self):
        tracker = self._tracker(diagnostics=True)
        local, packets, context = self._inputs()
        before = tracker._counterfactual_state_digest()
        result = tracker.c3r_counterfactual_candidates(
            local, packets, context)
        diagnostics = result["diagnostics"]
        self.assertEqual(set(diagnostics["branches"]), {
            "local", "sender1_only", "sender2_only", "both_sender"})
        for trace in diagnostics["branches"].values():
            self.assertEqual(trace["pre_state_digest"], before)
            self.assertEqual(trace["post_state_digest"], before)
        self.assertEqual(tracker._counterfactual_state_digest(), before)

    def test_counterfactual_adds_no_backbone_forward(self):
        tracker = self._tracker(diagnostics=True)
        local, packets, context = self._inputs()
        result = tracker.c3r_counterfactual_candidates(
            local, packets, context)
        diagnostics = result["diagnostics"]
        self.assertTrue(diagnostics["no_additional_backbone_forward"])
        self.assertEqual(
            diagnostics["backbone_forward_count_before"],
            diagnostics["backbone_forward_count_after"])

    def test_only_frozen_c1_behavior_candidate_can_be_submitted(self):
        tracker = self._tracker(diagnostics=True)
        local, packets, context = self._inputs()
        counterfactual = tracker.c3r_counterfactual_candidates(
            local, packets, context)
        behavior = dict(local)
        behavior["target_bbox"] = [11.0, 21.0, 31.0, 41.0]
        diagnostics = tracker._audit_behavior_candidate_submission(
            counterfactual, behavior)
        self.assertTrue(diagnostics["only_frozen_c1_behavior_submitted"])
        self.assertEqual(
            diagnostics["behavior_candidate_digest"],
            diagnostics["submitted_candidate_digest"])
        with self.assertRaises(AssertionError):
            tracker._audit_behavior_candidate_submission(
                counterfactual, counterfactual["both"])

    def test_diagnostics_switch_does_not_change_candidates(self):
        local, packets, context = self._inputs()
        disabled = self._tracker(diagnostics=False)
        enabled = self._tracker(diagnostics=True)
        enabled.network.c3r.load_state_dict(disabled.network.c3r.state_dict())
        first = disabled.c3r_counterfactual_candidates(
            copy.deepcopy(local), packets, context)
        second = enabled.c3r_counterfactual_candidates(
            copy.deepcopy(local), packets, context)
        self.assertEqual(first["both_sender_ids"], second["both_sender_ids"])
        self.assertEqual(
            first["both"]["target_bbox"], second["both"]["target_bbox"])
        self.assertEqual(
            {key: value["target_bbox"]
             for key, value in first["sender_only"].items()},
            {key: value["target_bbox"]
             for key, value in second["sender_only"].items()})


class TemporalGateEvaluationFrameworkTest(unittest.TestCase):
    def test_prediction_only_runner_rejects_gt_and_is_deterministic(self):
        def runner(policy, target):
            return [{"policy": policy, "target_id": target, "frame_id": 1,
                     "target_bbox": [1, 2, 3, 4], "uses_gt": False}]

        first = run_prediction_only(["md3054"], runner, ["md3005"])
        second = run_prediction_only(["md3054"], runner, ["md3005"])
        self.assertEqual(first["T1"]["prediction_sha256"],
                         second["T1"]["prediction_sha256"])
        with self.assertRaises(RuntimeError):
            assert_prediction_only({"gt_bbox": [0, 0, 1, 1]})

    def test_offline_and_closed_loop_diagnostics_are_finite(self):
        rows = []
        for target_index, target in enumerate(("md3017", "md3035")):
            for frame in range(1, 5):
                rows.append(labeled_row(
                    target=target, receiver=target_index, sender=1,
                    frame=frame,
                    delta_diou=(-0.25 if frame % 2 == 0 else 0.25),
                    features=[(frame + index) / 20.0 for index in range(10)]))
        report = evaluate(TemporalGate(), C3RTemporalGateDataset(rows))
        self.assertTrue(report["finite_diagnostics"])
        self.assertEqual(report["gate1_v2_thresholds"], GATE1_V2_THRESHOLDS)
        self.assertIn("target_clustered_spearman", report)
        self.assertIn("target_cluster_bootstrap_95ci", report)
        for key, value in report.items():
            if isinstance(value, (float, int)):
                self.assertTrue(math.isfinite(float(value)), key)

        joined = []
        for policy_index, policy in enumerate(("E0", "C1", "T1")):
            for target in ("md3017", "md3035", "md3050", "md3054"):
                joined.append({
                    "policy": policy,
                    "target_id": target,
                    "success_auc": 0.5 + 0.01 * policy_index,
                    "precision": 0.6 + 0.01 * policy_index,
                    "normalized_precision": 0.7 + 0.01 * policy_index,
                    "communication_bytes": 640 if policy != "E0" else 0,
                    "accepted_packets": 2 if policy != "E0" else 0,
                    "packet_bytes": 320 if policy != "E0" else 0,
                    "gates": [0.1, 0.2] if policy == "T1" else [],
                })
        closed = evaluate_joined_rows(joined)
        self.assertTrue(closed["finite_diagnostics"])
        self.assertTrue(closed["packet_accounting_identical"])


if __name__ == "__main__":
    unittest.main()
