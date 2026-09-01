import unittest

import torch

from tracking.target_prompt_d2_p2_partial_degradation import (
    ORIENTATIONS,
    apply_partial_occlusion,
    orientation_for_sample,
    partial_pixel_box,
)


class TargetPromptD2P2Tests(unittest.TestCase):
    def setUp(self):
        self.box = torch.tensor([0.25, 0.25, 0.50, 0.50])

    def test_orientation_is_deterministic_and_finite(self):
        sample_id = "d2p1-train-md3005-1-000005-clean"
        first = orientation_for_sample(sample_id)
        self.assertIn(first, ORIENTATIONS)
        self.assertEqual(first, orientation_for_sample(sample_id))

    def test_partial_blocks_are_nested_for_each_orientation(self):
        for orientation in ORIENTATIONS:
            blocks = [partial_pixel_box(
                self.box, 100, 100, coverage, orientation)[0]
                for coverage in (0.25, 0.50, 0.75, 1.00)]
            areas = [(x1 - x0) * (y1 - y0)
                     for x0, y0, x1, y1 in blocks]
            self.assertEqual(areas, [650, 1250, 1900, 2500])
            for smaller, larger in zip(blocks, blocks[1:]):
                self.assertGreaterEqual(smaller[0], larger[0])
                self.assertGreaterEqual(smaller[1], larger[1])
                self.assertLessEqual(smaller[2], larger[2])
                self.assertLessEqual(smaller[3], larger[3])

    def test_p100_is_exact_full_d1_fill(self):
        search = torch.ones(3, 100, 100)
        degraded, audit = apply_partial_occlusion(
            search, self.box, "P100", "clean-sample")
        expected = search.clone()
        expected[:, 25:75, 25:75] = 0.0
        self.assertTrue(torch.equal(degraded, expected))
        self.assertEqual(audit["pixel_box"], [25, 25, 75, 75])
        self.assertEqual(audit["realized_coverage"], 1.0)

    def test_only_one_contiguous_candidate_block_changes(self):
        search = torch.ones(3, 100, 100)
        degraded, audit = apply_partial_occlusion(
            search, self.box, "P50", "clean-sample")
        x0, y0, x1, y1 = audit["pixel_box"]
        changed = (degraded != search).all(dim=0)
        expected = torch.zeros(100, 100, dtype=torch.bool)
        expected[y0:y1, x0:x1] = True
        self.assertTrue(torch.equal(changed, expected))
        self.assertEqual(audit["block_pixels"], 1250)

    def test_invalid_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_partial_occlusion(
                torch.ones(3, 100, 100), self.box, "P33", "sample")


if __name__ == "__main__":
    unittest.main()
