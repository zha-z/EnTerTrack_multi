import unittest

import numpy as np
import pandas as pd

from tracking import analyze_c3r_reliability_v2_design as design


class ReliabilityV2DesignTest(unittest.TestCase):
    def test_catalog_is_frozen_and_prediction_only(self):
        catalog = design.candidate_catalog()
        self.assertEqual(len(catalog), 85)
        self.assertEqual(sum(row["availability"].startswith("AVAILABLE")
                             for row in catalog), 81)
        self.assertTrue(all(row["prediction_only"] for row in catalog))
        self.assertFalse(any(row["uses_cross_camera_absolute_center"]
                             for row in catalog))

    def test_outer_train_and_holdout_are_target_disjoint(self):
        for fold in range(5):
            train = set(design.read_target_set(
                design.SPECS / "c3r_f{}_train.txt".format(fold)))
            holdout = set(design.read_target_set(
                design.SPECS / "c3r_f{}_holdout.txt".format(fold)))
            self.assertFalse(train & holdout)
            self.assertEqual(train | holdout, set().union(*[
                set(design.read_target_set(
                    design.SPECS / "c3r_f{}_holdout.txt".format(index)))
                for index in range(5)
            ]))

    def test_run_length_resets(self):
        np.testing.assert_array_equal(
            design.run_length([False, True, True, False, True]),
            [0, 1, 2, 0, 1])

    def test_aggregate_features_do_not_depend_on_offline_delta(self):
        count = 20
        frame = np.arange(count, dtype=float)
        base = pd.DataFrame({
            "sequence_id": ["md9999-1"] * count,
            "receiver_view": [0] * count,
            "frame_id": np.arange(count),
            "local_bbox_x": frame,
            "local_bbox_y": 2 * frame,
            "local_bbox_w": [20.0] * count,
            "local_bbox_h": [10.0] * count,
            "c1_bbox_x": frame + 1.0,
            "c1_bbox_y": 2 * frame - 1.0,
            "c1_bbox_w": [20.0] * count,
            "c1_bbox_h": [10.0] * count,
            "local_confidence": np.linspace(0.9, 0.2, count),
            "c1_confidence": np.linspace(0.8, 0.3, count),
            "local_apce": np.linspace(100, 80, count),
            "c1_apce": np.linspace(95, 85, count),
            "local_quality_00": np.linspace(0.7, 0.5, count),
            "local_quality_01": np.linspace(0.8, 0.6, count),
            "local_quality_02": np.linspace(0.4, 0.6, count),
            "local_quality_03": np.linspace(0.5, 0.3, count),
            "iou_delta_offline": np.linspace(-1, 1, count),
        })
        changed = base.copy()
        changed["iou_delta_offline"] *= -7
        base.index = np.arange(100, 100 + count)
        changed.index = np.arange(200, 200 + count)
        first = design.compute_aggregate_features(base)
        second = design.compute_aggregate_features(changed)
        signal_names = [row["signal_name"] for row in design.candidate_catalog()
                        if row["group"] in ("A", "D")
                        and row["availability"].startswith("AVAILABLE")]
        for name in signal_names:
            np.testing.assert_equal(first[name].to_numpy(), second[name].to_numpy())
        self.assertGreater(
            first["state_center_divergence_trend_w8"].notna().sum(), 0)

    def test_primary_label_boundary(self):
        frame = pd.DataFrame({
            "sequence_id": ["x-1"] * 5,
            "receiver_view": [0] * 5,
            "frame_id": np.arange(5),
            "iou_delta_offline": [-0.02, -0.01, 0.0, 0.01, 0.02],
        })
        labels = design.add_offline_labels(frame).label_primary.tolist()
        self.assertEqual(labels, ["harmful", "tied", "tied", "tied", "helpful"])


if __name__ == "__main__":
    unittest.main()
