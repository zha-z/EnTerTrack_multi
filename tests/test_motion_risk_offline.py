import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tracking.analyze_motion_risk_offline import (
    FORBIDDEN_RISK_OUTPUT_FIELDS,
    PROFILE_DEFAULTS,
    ProvisionalRiskStateMachine,
    apply_risk_ema,
    causal_robust_risk,
    combine_risk_components,
    compute_prediction_only_risk,
    prediction_only_interface_is_gt_free,
    target_group_folds,
    validate_dataset,
    write_jsonl,
)


def diagnostic_record(frame_id, motion=0.1, entropy=0.5, score=0.8):
    return {
        "frame_id": frame_id,
        "max_score": score,
        "apce": 100.0,
        "response_entropy": entropy,
        "normalized_motion_residual": motion,
        "bbox_border_proximity": 0.4,
        "remote_quality": 0.8,
        "remote_weight_entropy": 0.5,
        "remote_max_weight": 0.7,
        "response_top1_top2_gap": 0.4,
        "response_peak_sharpness": 10.0,
    }


class MotionRiskOfflineTest(unittest.TestCase):
    def test_causal_normalization_does_not_read_future(self):
        common = [
            diagnostic_record(0, motion=0.1),
            diagnostic_record(1, motion=0.2),
            diagnostic_record(2, motion=0.3),
        ]
        first = compute_prediction_only_risk(
            common + [diagnostic_record(3, motion=100.0)], min_history=2
        )
        second = compute_prediction_only_risk(
            common + [diagnostic_record(3, motion=-100.0)], min_history=2
        )
        self.assertEqual(first[2], second[2])

    def test_zero_mad_is_safe(self):
        self.assertEqual(causal_robust_risk(1.0, [1.0, 1.0], "high"), 0.5)
        self.assertEqual(causal_robust_risk(2.0, [1.0, 1.0], "high"), 1.0)
        self.assertEqual(causal_robust_risk(2.0, [1.0, 1.0], "low"), 0.0)

    def test_high_and_low_risk_directions(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertGreater(
            causal_robust_risk(6.0, history, "high"),
            causal_robust_risk(2.0, history, "high"),
        )
        self.assertGreater(
            causal_robust_risk(1.0, history, "low"),
            causal_robust_risk(4.0, history, "low"),
        )

    def test_single_motion_or_border_signal_cannot_directly_trigger_lost(self):
        normalized = {
            "normalized_motion_residual_risk": 0.99,
            "response_entropy_risk": 0.1,
            "max_score_risk": 0.1,
            "apce_risk": 0.1,
            "remote_quality_risk": 0.1,
            "bbox_border_proximity_risk": 0.1,
            "top1_top2_gap_risk": 0.1,
            "peak_sharpness_risk": 0.1,
        }
        motion_only = combine_risk_components(normalized)
        self.assertEqual(motion_only["high_risk_signal_count"], 1)
        self.assertTrue(motion_only["consistency_reduced"])

        normalized["normalized_motion_residual_risk"] = 0.1
        normalized["bbox_border_proximity_risk"] = 0.99
        border_only = combine_risk_components(normalized)
        self.assertEqual(border_only["high_risk_signal_count"], 0)
        self.assertTrue(border_only["consistency_reduced"])

        machine = ProvisionalRiskStateMachine(PROFILE_DEFAULTS["balanced"])
        for _ in range(20):
            state, _ = machine.update(0.99, high_risk_count=1)
        self.assertNotEqual(state, "LOST")

    def test_two_high_risk_signals_raise_risk(self):
        base = {
            "normalized_motion_residual_risk": 0.9,
            "response_entropy_risk": 0.1,
            "max_score_risk": 0.1,
            "apce_risk": 0.1,
            "remote_quality_risk": 0.1,
            "bbox_border_proximity_risk": 0.1,
            "top1_top2_gap_risk": 0.1,
            "peak_sharpness_risk": 0.1,
        }
        one = combine_risk_components(base)
        base["response_entropy_risk"] = 0.9
        two = combine_risk_components(base)
        self.assertEqual(two["high_risk_signal_count"], 2)
        self.assertGreater(two["instantaneous_risk"], one["instantaneous_risk"])

    def test_ema(self):
        self.assertEqual(apply_risk_ema(None, 0.5, 0.8), 0.5)
        self.assertAlmostEqual(apply_risk_ema(0.5, 1.0, 0.8), 0.6)

    def test_hysteresis(self):
        profile = {
            "uncertain_enter_threshold": 0.5,
            "lost_enter_threshold": 0.7,
            "lost_exit_threshold": 0.3,
            "k_uncertain": 1,
            "k_lost": 1,
            "k_exit": 1,
            "minimum_high_risk_signal_count": 2,
        }
        machine = ProvisionalRiskStateMachine(profile)
        self.assertEqual(machine.update(0.8, 2)[0], "UNCERTAIN")
        self.assertEqual(machine.update(0.8, 2)[0], "LOST")
        self.assertEqual(machine.update(0.2, 0)[0], "RECOVER")
        self.assertEqual(machine.update(0.2, 0)[0], "NORMAL")

    def test_target_group_split_keeps_three_views_together(self):
        sequences = [
            "md3016-1", "md3016-2", "md3016-3",
            "md3027-1", "md3027-2", "md3027-3",
        ]
        folds = target_group_folds(sequences)
        held = {fold["held_out_target"]: fold for fold in folds}
        self.assertEqual(
            held["md3016"]["held_out_sequences"],
            ["md3016-1", "md3016-2", "md3016-3"],
        )
        self.assertTrue(
            set(held["md3016"]["held_out_sequences"]).isdisjoint(
                held["md3016"]["train_sequences"]
            )
        )

    def test_prediction_only_interface_accepts_no_gt(self):
        self.assertTrue(prediction_only_interface_is_gt_free())
        parameters = inspect.signature(compute_prediction_only_risk).parameters
        self.assertNotIn("gt", parameters)
        self.assertNotIn("visibility", parameters)

    def test_non_validation_dataset_is_rejected(self):
        validate_dataset("threemdot_val")
        with self.assertRaises(ValueError):
            validate_dataset("threemdot_test")

    def test_risk_jsonl_has_no_gt_fields(self):
        records = compute_prediction_only_risk(
            [diagnostic_record(index) for index in range(25)], min_history=20
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "risk.jsonl"
            write_jsonl(path, records)
            loaded = [json.loads(line) for line in path.read_text().splitlines()]
        for record in loaded:
            self.assertFalse(FORBIDDEN_RISK_OUTPUT_FIELDS.intersection(record))
            serialized = json.dumps(record).lower()
            for forbidden in ("gt_bbox", "iou", "visibility", "target_visible"):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
