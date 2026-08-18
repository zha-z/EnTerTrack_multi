import unittest

import numpy as np

import common
from lib.test.analysis.fcvc_results import _curves


class OSTrackResultsTest(unittest.TestCase):
    def test_zero_overlap_is_not_success_at_zero_threshold(self):
        target = np.asarray([[0.0, 0.0, 10.0, 10.0]])
        prediction = np.asarray([[100.0, 100.0, 10.0, 10.0]])
        # OSTrack resets the first prediction to the initialization box, so use
        # a second frame to exercise a genuine zero-overlap tracking result.
        target = np.concatenate((target, target), axis=0)
        prediction = np.concatenate((prediction, prediction), axis=0)
        metrics = _curves(
            prediction, target, target_visible=[1, 1], dataset="threemdot"
        )
        self.assertAlmostEqual(metrics["auc"], 10.0 / 21.0, places=12)

    def test_visibility_and_first_frame_match_ostrack_semantics(self):
        target = np.asarray([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ])
        prediction = np.asarray([
            [100.0, 100.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ])
        metrics = _curves(
            prediction, target, target_visible=[1, 0], dataset="threemdot"
        )
        self.assertAlmostEqual(metrics["auc"], 10.0 / 21.0, places=12)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=12)
        self.assertAlmostEqual(metrics["normalized_precision"], 1.0, places=12)
        self.assertAlmostEqual(metrics["mean_iou"], 1.0, places=12)
        self.assertEqual(metrics["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
