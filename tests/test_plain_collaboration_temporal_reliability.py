import math
import unittest

from tracking.analyze_plain_collaboration_temporal_reliability import (
    FEATURE_SETS,
    _bool,
    motion_series,
    policy_choice,
    temporal_summary,
)


class TemporalReliabilityAuditTests(unittest.TestCase):

    def test_temporal_summary_is_future_invariant(self):
        frames = list(range(12))
        original = [float(index) for index in frames]
        changed = list(original)
        changed[7:] = [1000.0] * 5
        self.assertEqual(
            temporal_summary(original, frames, 6, 8),
            temporal_summary(changed, frames, 6, 8),
        )

    def test_temporal_slope_uses_causal_prefix(self):
        result = temporal_summary([1.0, 3.0, 5.0, 7.0], [0, 1, 2, 3], 3, 4)
        self.assertAlmostEqual(result["mean"], 4.0)
        self.assertAlmostEqual(result["std"], math.sqrt(5.0))
        self.assertAlmostEqual(result["slope"], 2.0)

    def test_motion_series_is_constant_velocity_causal(self):
        boxes = [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 0.0, 10.0, 10.0],
            [2.0, 0.0, 10.0, 10.0],
        ]
        result = motion_series(boxes)
        self.assertAlmostEqual(result["normalized_motion_residual"][2], 0.0)
        self.assertAlmostEqual(result["acceleration_norm"][2], 0.0)
        self.assertGreater(result["velocity_norm"][1], 0.0)

    def test_feature_sets_are_preregistered_nested(self):
        self.assertEqual(tuple(FEATURE_SETS), ("T0", "T1", "T2", "T3", "T4"))
        for first, second in zip(("T0", "T1", "T2", "T3"),
                                 ("T1", "T2", "T3", "T4")):
            self.assertTrue(set(FEATURE_SETS[first]).issubset(
                set(FEATURE_SETS[second])))

    def test_policy_is_fixed_and_fail_closed(self):
        self.assertEqual(policy_choice(0.4, 0.3, 5), "local")
        self.assertEqual(policy_choice(0.7, 0.6, 5), "sender0_only")
        self.assertEqual(policy_choice(0.6, 0.7, 5), "sender1_only")
        self.assertEqual(policy_choice(0.7, 0.7, 5), "local")
        self.assertEqual(policy_choice(float("nan"), 0.9, 5), "local")
        self.assertEqual(policy_choice(0.9, 0.1, 0), "local")

    def test_csv_booleans_are_not_truthy_strings(self):
        self.assertTrue(_bool("True"))
        self.assertFalse(_bool("False"))


if __name__ == "__main__":
    unittest.main()
