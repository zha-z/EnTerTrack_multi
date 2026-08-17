import copy
import unittest
from pathlib import Path

import yaml

from lib.config.entertrack.config import cfg, update_config_from_file


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "experiments/entertrack/pcum_v2a_motion_state_shadow_val.yaml"
CONSERVATIVE = (
    ROOT
    / "experiments/entertrack/pcum_v2a_motion_state_shadow_val_conservative.yaml"
)
CHANGED_PATHS = {
    ("TEST", "MOTION_STATE", "SCORE_LOW"),
    ("TEST", "MOTION_STATE", "APCE_LOW"),
    ("TEST", "MOTION_STATE", "MOTION_RESIDUAL_HIGH"),
    ("TEST", "MOTION_STATE", "K_LOST"),
}


def leaf_values(value, path=()):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(leaf_values(child, path + (key,)))
        return result
    return {path: value}


class MotionStateConservativeConfigTest(unittest.TestCase):
    def test_conservative_config_loads_with_no_gt_shadow_settings(self):
        loaded = copy.deepcopy(cfg)
        update_config_from_file(str(CONSERVATIVE), base_cfg=loaded)
        motion = loaded.TEST.MOTION_STATE
        self.assertEqual(motion.SCORE_LOW, 0.3498124361)
        self.assertEqual(motion.APCE_LOW, 60.5456100464)
        self.assertEqual(motion.MOTION_RESIDUAL_HIGH, 0.1265809693)
        self.assertEqual(motion.K_LOST, 3)
        self.assertTrue(motion.ENABLED)
        self.assertTrue(motion.SHADOW_ONLY)
        self.assertTrue(motion.LOG_ENABLED)
        self.assertEqual(loaded.TEST.PCUM.REMOTE_STATE_SOURCE, "tracker")
        self.assertFalse(loaded.TEST.PCUM.USE_REMOTE_VISIBLE_MASK)

    def test_only_declared_threshold_fields_differ(self):
        with ORIGINAL.open(encoding="utf-8") as file_handle:
            original = yaml.safe_load(file_handle)
        with CONSERVATIVE.open(encoding="utf-8") as file_handle:
            conservative = yaml.safe_load(file_handle)
        original_leaves = leaf_values(original)
        conservative_leaves = leaf_values(conservative)
        self.assertEqual(set(original_leaves), set(conservative_leaves))
        differences = {
            path for path in original_leaves
            if original_leaves[path] != conservative_leaves[path]
        }
        self.assertEqual(differences, CHANGED_PATHS - {
            ("TEST", "MOTION_STATE", "K_LOST")
        })
        self.assertEqual(original["TEST"]["MOTION_STATE"]["K_LOST"], 3)
        self.assertEqual(conservative["TEST"]["MOTION_STATE"]["K_LOST"], 3)

    def test_config_contains_no_gt_or_test_specific_inputs(self):
        text = CONSERVATIVE.read_text(encoding="utf-8").lower()
        for forbidden in (
            "target_visible",
            "gt_visibility",
            "oracle_mask",
            "threemdot_test",
        ):
            if forbidden == "threemdot_test":
                self.assertIn("not valid for threemdot_test", text)
                continue
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
