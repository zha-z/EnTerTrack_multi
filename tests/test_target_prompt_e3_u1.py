import unittest

import torch

from tracking.analyze_target_prompt_e3_u1_utility import (
    P0, P1, P2, P3, P4)
from tracking.run_target_prompt_e3_u1_counterfactual import (
    _bbox_disagreement,
    _masked_remote_candidates,
    _prompt_stats,
    _set_compatibility,
    _validate_no_forbidden_columns,
)


class TargetPromptE3U1Tests(unittest.TestCase):
    def test_single_sender_mask_keeps_only_requested_local_packet(self):
        remotes = tuple({
            "target_prompt_packet": {
                "valid": torch.tensor([True]),
                "source": "local",
                "prompt": torch.randn(1, 8, 12),
            }} for _ in range(2))
        masked = _masked_remote_candidates(remotes, 1)
        self.assertEqual(
            [bool(item["target_prompt_packet"]["valid"].item())
             for item in masked], [False, True])
        self.assertTrue(all(item["target_prompt_packet"]["source"] == "local"
                            for item in masked))
        self.assertTrue(all(item["target_prompt_packet"]["valid"].item()
                            for item in remotes))

    def test_prompt_statistics_use_all_off_diagonal_pairs(self):
        prompt = torch.eye(8).unsqueeze(0)
        metadata = {
            "prompt": prompt,
            "topk_scores": torch.arange(8, dtype=torch.float32).unsqueeze(0),
        }
        stats = _prompt_stats(metadata)
        self.assertEqual(stats["prompt_top1_top8_gap"], 7.0)
        self.assertEqual(stats["prompt_pairwise_cos_mean"], 0.0)
        self.assertEqual(stats["prompt_pairwise_cos_min"], 0.0)
        self.assertEqual(stats["prompt_pairwise_cos_max"], 0.0)

    def test_prompt_set_compatibility_is_symmetric_for_identical_sets(self):
        prompt = torch.eye(8).unsqueeze(0)
        values = _set_compatibility(prompt, prompt)
        self.assertAlmostEqual(values["sender_to_receiver_best_mean"], 1.0)
        self.assertAlmostEqual(values["receiver_to_sender_best_mean"], 1.0)
        self.assertAlmostEqual(values["symmetric_best_match"], 1.0)

    def test_bbox_disagreement_identity_is_zero(self):
        displacement, scale = _bbox_disagreement(
            [10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(displacement, 0.0)
        self.assertEqual(scale, 0.0)

    def test_prediction_schema_rejects_gt_and_feature_groups_are_nested(self):
        with self.assertRaises(RuntimeError):
            _validate_no_forbidden_columns(("target_id", "gt_bbox"))
        self.assertTrue(set(P0).issubset(P1))
        self.assertTrue(set(P1).issubset(P2))
        self.assertTrue(set(P2).issubset(P3))
        self.assertTrue(set(P3).issubset(P4))


if __name__ == "__main__":
    unittest.main()
