import tempfile
import unittest
from pathlib import Path

from tracking.analyze_target_level_uncertainty import (
    bootstrap_target_differences,
    percentile,
)
from tracking.audit_tracking_dataset_splits import eligibility_for
from tracking.build_tracking_event_manifest import (
    compute_annotation_events,
    load_tracker_events,
)
from tracking.dataset_development_utils import (
    assert_targets_not_split,
    read_sequence_list,
    split_overlap_rows,
)


def metric_rows():
    rows = []
    for target_id, delta in (("target_a", 1.0), ("target_b", -1.0)):
        for view_id in ("1", "2", "3"):
            rows.append(
                {
                    "target_id": target_id,
                    "view_id": view_id,
                    "method": "A0",
                    "auc": 10.0,
                    "precision": 20.0,
                    "norm_precision": 30.0,
                }
            )
            rows.append(
                {
                    "target_id": target_id,
                    "view_id": view_id,
                    "method": "candidate",
                    "auc": 10.0 + delta,
                    "precision": 20.0 + delta,
                    "norm_precision": 30.0 + delta,
                }
            )
    return rows


class DatasetSplitAuditTests(unittest.TestCase):
    def test_same_target_views_are_not_split(self):
        with self.assertRaisesRegex(ValueError, "Targets appear in multiple splits"):
            assert_targets_not_split(
                {
                    "train": ["md1-1", "md1-2"],
                    "val": ["md1-3"],
                }
            )

    def test_sequence_and_target_overlap_detection(self):
        rows = split_overlap_rows(
            {
                "train": ["md1-1", "md1-2", "md1-3"],
                "val": ["md1-1", "md2-1", "md2-2", "md2-3"],
                "test": ["md3-1", "md3-2", "md3-3"],
            }
        )
        failures = [row for row in rows if row["status"] == "FAIL"]
        self.assertTrue(any(row["overlap_type"] == "sequence" for row in failures))
        self.assertTrue(any(row["overlap_type"] == "target" for row in failures))
        train_test = [
            row
            for row in rows
            if {row["left_split"], row["right_split"]} == {"train", "test"}
        ]
        self.assertTrue(all(row["status"] == "PASS" for row in train_test))

    def test_test_is_never_eligible_for_development(self):
        eligible, reason = eligibility_for("test", used_in_training=False)
        self.assertFalse(eligible)
        self.assertEqual(reason, "formal_test_reserved")

    def test_missing_sequence_list_has_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Sequence list not found"):
            read_sequence_list(Path("/definitely/missing/split.txt"))


class TargetBootstrapTests(unittest.TestCase):
    def test_bootstrap_uses_target_groups_and_keeps_three_views_together(self):
        summary, replicates = bootstrap_target_differences(
            metric_rows(),
            "A0",
            "candidate",
            iterations=100,
            seed=3,
            expected_views=3,
        )
        auc_values = {
            round(row["delta"], 8)
            for row in replicates
            if row["metric"] == "auc"
        }
        # Sampling two whole targets can only yield -1, 0, or +1. View-level
        # resampling would introduce additional fractional values.
        self.assertTrue(auc_values.issubset({-1.0, 0.0, 1.0}))
        self.assertTrue(all(row["resampling_unit"] == "target_group" for row in replicates))
        self.assertEqual({row["target_count"] for row in summary}, {2})

    def test_ci_calculation(self):
        self.assertEqual(percentile([0.0, 1.0, 2.0, 3.0, 4.0], 0.5), 2.0)
        summary, _ = bootstrap_target_differences(
            metric_rows(), "A0", "candidate", iterations=200, seed=9
        )
        self.assertTrue(
            all(
                row["bootstrap_ci_low_95"] <= row["mean_delta"] <= row["bootstrap_ci_high_95"]
                for row in summary
            )
        )


class EventManifestTests(unittest.TestCase):
    def test_annotation_events_are_explicitly_separate(self):
        metadata = {
            "dataset": "synthetic",
            "split": "val",
            "target_id": "t1",
            "view_id": "1",
            "sequence_name": "t1-1",
        }
        boxes = [
            [0, 0, 10, 10],
            [1, 0, 10, 10],
            [2, 0, 10, 10],
            [30, 0, 10, 10],
            [31, 0, 10, 10],
        ]
        rows = compute_annotation_events(
            metadata,
            boxes,
            [False, True, True, False, False],
            [False] * 5,
            long_occlusion_frames=2,
        )
        self.assertTrue(rows)
        self.assertEqual({row["event_source"] for row in rows}, {"annotation-derived"})
        self.assertIn("reappearance", {row["event_type"] for row in rows})
        self.assertIn("long_occlusion", {row["event_type"] for row in rows})

    def test_tracker_events_require_tracker_specific_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(
                "event_source,event_type\nannotation-derived,failure\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "tracker-specific"):
                load_tracker_events(path)

    def test_annotation_length_mismatch_is_clear(self):
        metadata = {
            "dataset": "synthetic",
            "split": "val",
            "target_id": "t1",
            "view_id": "1",
            "sequence_name": "t1-1",
        }
        with self.assertRaisesRegex(ValueError, "Annotation length mismatch"):
            compute_annotation_events(
                metadata,
                [[0, 0, 10, 10]],
                [],
                [False],
            )


if __name__ == "__main__":
    unittest.main()
