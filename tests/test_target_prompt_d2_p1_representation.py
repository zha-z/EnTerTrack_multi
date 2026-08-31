import unittest

import numpy as np
import torch

from tracking.analyze_target_prompt_d2_p1_representation import (
    ks_statistic,
    paired_prompt_metrics,
    standardized_mean_difference,
    wasserstein_equal,
)
from tracking.run_target_prompt_d2_p1_representation import (
    previous_eligible,
    prompt_statistics,
    score_entropy,
    systematic_indices,
)


class TargetPromptD2P1Tests(unittest.TestCase):
    def test_midpoint_systematic_indices_are_frozen_and_unique(self):
        self.assertEqual(systematic_indices(10, 4), [1, 3, 6, 8])
        self.assertEqual(systematic_indices(3, 3), [0, 1, 2])
        self.assertEqual(systematic_indices(10, 0), [])
        with self.assertRaises(ValueError):
            systematic_indices(3, 4)

    def test_previous_eligible_is_causal_and_gap_limited(self):
        mask = np.zeros(405, dtype=bool)
        mask[3] = True
        mask[202] = True
        self.assertEqual(previous_eligible(mask, 203, max_gap=200), 202)
        self.assertIsNone(previous_eligible(mask, 405, max_gap=200))

    def test_score_entropy_uses_normalized_nonnegative_response(self):
        uniform = torch.ones(1, 1, 16, 16)
        impulse = torch.zeros(1, 1, 16, 16)
        impulse[0, 0, 3, 4] = 1.0
        self.assertAlmostEqual(float(score_entropy(uniform).item()), 1.0, 6)
        self.assertAlmostEqual(float(score_entropy(impulse).item()), 0.0, 6)

    def test_prompt_statistics_use_all_off_diagonal_pairs(self):
        prompt = torch.eye(8).unsqueeze(0)
        scores = torch.arange(8, 0, -1).float().unsqueeze(0)
        stats = prompt_statistics(prompt, scores)
        self.assertAlmostEqual(
            float(stats["prompt_pairwise_cos_mean"].item()), 0.0, 7)
        self.assertAlmostEqual(
            float(stats["prompt_top1_top8_gap"].item()), 7.0, 7)

    def test_pair_prompt_metrics_are_identity_for_equal_sets(self):
        prompt = np.eye(8, dtype=np.float32)
        metrics = paired_prompt_metrics(prompt, prompt.copy())
        self.assertAlmostEqual(metrics["c_to_s_nearest_cos_mean"], 1.0, 7)
        self.assertAlmostEqual(metrics["s_to_c_nearest_cos_mean"], 1.0, 7)
        self.assertAlmostEqual(metrics["symmetric_best_match_cos"], 1.0, 7)
        self.assertAlmostEqual(metrics["centroid_cos_control"], 1.0, 7)

    def test_fixed_distribution_distances(self):
        first = np.asarray([0.0, 1.0, 2.0])
        second = np.asarray([1.0, 2.0, 3.0])
        self.assertAlmostEqual(wasserstein_equal(first, second), 1.0)
        self.assertAlmostEqual(ks_statistic(first, second), 1.0 / 3.0)
        self.assertLess(standardized_mean_difference(first, second), 0.0)


if __name__ == "__main__":
    unittest.main()
