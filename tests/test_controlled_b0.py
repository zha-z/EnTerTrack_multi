import copy
import inspect
import unittest

from lib.models.entertrack.entertrack import EnTeRTrack
from tracking.audit_controlled_baseline_config import (
    B0_BACKBONE_SPEC,
    B0_CONFIG,
    B0_ROLE,
    load_config,
    resolved_values,
    run_checkpoint_forward_smoke,
    validate_config,
)


class TestControlledB0Config(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.path = load_config(B0_CONFIG)
        cls.values = resolved_values(B0_CONFIG, cls.config)
        cls.smoke = run_checkpoint_forward_smoke(B0_CONFIG)

    def test_epoch_is_fixed_at_25(self):
        self.assertEqual(self.values["total_epochs"], 25)
        self.assertEqual(self.values["train_epochs"], 25)

    def test_pcum_is_disabled(self):
        self.assertFalse(self.values["pcum_enabled"])

    def test_mcr_is_disabled(self):
        self.assertFalse(self.values["mcr_enabled"])

    def test_pruning_is_disabled(self):
        self.assertFalse(self.values["pruning_enabled"])
        self.assertEqual(list(self.config.MODEL.BACKBONE.CE_LOC), [])

    def test_compensation_is_disabled(self):
        self.assertFalse(self.values["compensation_enabled"])

    def test_dynamic_threshold_is_disabled(self):
        self.assertFalse(self.values["dynamic_threshold_enabled"])

    def test_remote_input_is_disabled(self):
        self.assertFalse(self.values["remote_input_enabled"])
        self.assertEqual(self.values["remote_state_source"], "none")

    def test_backbone_matches_controlled_definition(self):
        self.assertEqual(self.values["embed_dim"], B0_BACKBONE_SPEC["embed_dim"])
        self.assertEqual(self.values["depth"], B0_BACKBONE_SPEC["depth"])
        self.assertEqual(self.values["heads"], B0_BACKBONE_SPEC["heads"])
        self.assertEqual(self.values["patch_size"], B0_BACKBONE_SPEC["patch_size"])

    def test_template_and_search_sizes(self):
        self.assertEqual(self.values["template_size"], 128)
        self.assertEqual(self.values["search_size"], 256)
        self.assertEqual(self.values["test_template_size"], 128)
        self.assertEqual(self.values["test_search_size"], 256)

    def test_valid_config_passes_audit(self):
        self.assertEqual(validate_config(self.values, B0_ROLE), [])

    def test_bad_config_fails_audit(self):
        bad = copy.deepcopy(self.values)
        bad["pcum_enabled"] = True
        bad["train_epochs"] = 24
        errors = validate_config(bad, B0_ROLE)
        self.assertTrue(any("PCUM" in error for error in errors))
        self.assertTrue(any("25" in error for error in errors))

    def test_checkpoint_core_load_is_strict(self):
        self.assertTrue(self.smoke["strict_core_load"])
        self.assertEqual(self.smoke["missing_keys"], [])
        self.assertEqual(self.smoke["unexpected_keys"], [])
        self.assertEqual(self.smoke["shape_mismatches"], [])
        self.assertTrue(self.smoke["excluded_source_keys"])
        self.assertTrue(all(
            ".atp." in key for key in self.smoke["excluded_source_keys"]
        ))

    def test_forward_outputs_have_expected_shapes(self):
        self.assertEqual(self.smoke["pred_boxes_shape"], (1, 1, 4))
        self.assertEqual(self.smoke["score_map_shape"], (1, 1, 16, 16))

    def test_forward_outputs_are_finite(self):
        self.assertTrue(self.smoke["finite"])

    def test_no_gt_inference_input_added(self):
        self.assertTrue(self.values["no_gt_inference"])
        names = set(inspect.signature(EnTeRTrack.forward).parameters)
        forbidden = {"gt", "ground_truth", "target_visible", "gt_visibility"}
        self.assertTrue(names.isdisjoint(forbidden))
        self.assertFalse(self.values["use_remote_visible_mask"])


if __name__ == "__main__":
    unittest.main()
