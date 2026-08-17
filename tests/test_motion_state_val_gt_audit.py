import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tracking.evaluate_motion_state_with_val_gt import (
    bbox_iou_xywh,
    consecutive_true,
    construct_event_labels,
    detection_delays,
    ensure_output_separate,
    false_trigger_count,
    load_visibility,
    load_diagnostics,
    segmented_detection_delays,
    validate_dataset,
    validate_lengths,
)


class MotionStateValidationGtAuditTest(unittest.TestCase):
    def test_iou_failure_and_k_frame_debounce(self):
        iou = np.asarray([0.9, 0.05, 0.04, 0.03, 0.8])
        center_error = np.zeros_like(iou)
        labels = construct_event_labels(
            iou, center_error, failure_mode="iou", k_fail=3
        )
        np.testing.assert_array_equal(
            labels["stable_failure"], [False, False, False, True, False]
        )
        np.testing.assert_array_equal(
            consecutive_true([True, True, False, True, True, True], 3),
            [False, False, False, False, False, True],
        )

    def test_recovery_event_requires_failure_and_k_frames(self):
        labels = construct_event_labels(
            iou=np.asarray([0.0, 0.0, 0.0, 0.6, 0.7, 0.8]),
            center_error=np.zeros(6),
            failure_mode="iou",
            k_fail=3,
            k_recover=3,
        )
        np.testing.assert_array_equal(
            labels["stable_recovery"], [False, False, False, False, False, True]
        )

    def test_detection_delay_and_false_trigger_count(self):
        detection = np.asarray([True, False, False, False, True, False, True])
        timing = detection_delays([(3, 5)], detection, warning_window=0)
        self.assertEqual(timing["delays"], [1])
        self.assertEqual(timing["missed"], 0)
        target = np.asarray([False, False, False, True, True, True, False])
        self.assertEqual(false_trigger_count(detection, target), 2)

    def test_detection_delay_does_not_cross_sequence_boundary(self):
        timing = segmented_detection_delays(
            target_events=[(1, 1)],
            detection_mask=[False, False, True, False],
            sequence_ids=["a", "a", "b", "b"],
        )
        self.assertEqual(timing["missed"], 1)
        self.assertEqual(timing["first_after"], [None])

    def test_visibility_available_and_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "occlusion.txt", [0, 1, 0], fmt="%d")
            np.savetxt(root / "out_of_view.txt", [0, 0, 1], fmt="%d")
            available = load_visibility(root, 3)
            self.assertTrue(available["available"])
            np.testing.assert_array_equal(
                available["visible"], [True, False, False]
            )

        with tempfile.TemporaryDirectory() as directory:
            unavailable = load_visibility(Path(directory), 3)
            self.assertFalse(unavailable["available"])
            self.assertIsNone(unavailable["visible"])

    def test_frame_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame length mismatch"):
            validate_lengths(3, prediction=np.zeros((2, 4)))

    def test_output_cannot_write_into_prediction_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = Path(directory) / "motion_state_diagnostics"
            diagnostics.mkdir()
            with self.assertRaisesRegex(ValueError, "never write"):
                ensure_output_separate(diagnostics, diagnostics / "gt_audit")

    def test_loading_diagnostics_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.jsonl"
            original = (
                json.dumps({"frame_id": 0, "state": "NORMAL"}) + "\n"
            ).encode("utf-8")
            path.write_bytes(original)
            records = load_diagnostics(path)
            self.assertEqual(records[0]["state"], "NORMAL")
            self.assertEqual(path.read_bytes(), original)

    def test_threemdot_test_is_rejected(self):
        validate_dataset("threemdot_val")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            validate_dataset("threemdot_test")

    def test_bbox_iou_xywh(self):
        prediction = np.asarray([[0, 0, 10, 10], [10, 10, 2, 2]])
        ground_truth = np.asarray([[0, 0, 10, 10], [0, 0, 2, 2]])
        np.testing.assert_allclose(
            bbox_iou_xywh(prediction, ground_truth), [1.0, 0.0]
        )


if __name__ == "__main__":
    unittest.main()
