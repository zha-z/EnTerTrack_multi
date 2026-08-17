import contextlib
import io
import os
import unittest

import yaml

from tracking.audit_mcr_resolved_config import (
    ROOT,
    audit_config,
    main,
    resolve_config,
    resolved_values,
)


MCR_MODES = {
    "pcum_v2a_mcr_v0_shadow_val": "SHADOW",
    "pcum_v2a_mcr_v0_active_smoke": "ACTIVE",
    "pcum_v2a_mcr_v0_current_anchor_only": "ACTIVE",
    "pcum_v2a_mcr_v0_current_motion_anchors": "ACTIVE",
    "pcum_v2a_mcr_v0_no_remote_verify": "ACTIVE",
    "pcum_v2a_mcr_v0_no_multiframe_confirm": "ACTIVE",
    "pcum_v2a_mcr_v0_full_local": "ACTIVE",
    "pcum_v2a_mcr_v0_full_local_safegeom": "ACTIVE",
}


class TestMCRConfigAudit(unittest.TestCase):
    def test_all_mcr_configs_resolve_to_declared_mode(self):
        for name, mode in MCR_MODES.items():
            with self.subTest(config=name):
                values = audit_config(name, mode.lower())
                self.assertTrue(values["MCR.ENABLED"])
                self.assertFalse(values["MCR.GLOBAL_ENABLED"])
                self.assertEqual(values["mode"], mode)

    def test_mode_fields_are_explicit_in_every_yaml(self):
        for name, mode in MCR_MODES.items():
            path = os.path.join(ROOT, "experiments", "entertrack", name + ".yaml")
            with open(path) as handle:
                raw = yaml.safe_load(handle)
            mcr = raw["TEST"]["MCR"]
            with self.subTest(config=name):
                self.assertIs(mcr["ENABLED"], True)
                self.assertIs(mcr["GLOBAL_ENABLED"], False)
                self.assertIs(mcr["SHADOW_ONLY"], mode == "SHADOW")

    def test_active_expectation_rejects_shadow_config(self):
        with contextlib.redirect_stderr(io.StringIO()):
            exit_code = main([
                "--config", "pcum_v2a_mcr_v0_shadow_val",
                "--expect-mode", "active",
            ])
        self.assertNotEqual(exit_code, 0)

    def test_shadow_expectation_rejects_active_config(self):
        with contextlib.redirect_stderr(io.StringIO()):
            exit_code = main([
                "--config", "pcum_v2a_mcr_v0_full_local",
                "--expect-mode", "shadow",
            ])
        self.assertNotEqual(exit_code, 0)

    def test_full_local_checkpoint_and_input_match_a0(self):
        full = resolved_values("pcum_v2a_mcr_v0_full_local")
        _, full_cfg = resolve_config("pcum_v2a_mcr_v0_full_local")
        full_architecture = (
            full_cfg.DATA.SEARCH.SIZE,
            full_cfg.DATA.TEMPLATE.SIZE,
            full_cfg.TEST.SEARCH_SIZE,
            full_cfg.TEST.TEMPLATE_SIZE,
            full_cfg.MODEL.BACKBONE.STRIDE,
        )
        _, a0_cfg = resolve_config("pcum_v2_a0_softmax_t010_ep0015")
        a0_architecture = (
            a0_cfg.DATA.SEARCH.SIZE,
            a0_cfg.DATA.TEMPLATE.SIZE,
            a0_cfg.TEST.SEARCH_SIZE,
            a0_cfg.TEST.TEMPLATE_SIZE,
            a0_cfg.MODEL.BACKBONE.STRIDE,
        )
        self.assertEqual(full_architecture, a0_architecture)
        self.assertEqual(full_architecture, (256, 128, 256, 128, 16))
        self.assertIn("pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40", full["checkpoint"])
        self.assertTrue(full["checkpoint"].endswith("EnTeRTrack_ep0015.pth.tar"))

    def test_full_local_guard_is_disabled_and_safegeom_guard_is_active(self):
        full = resolved_values("pcum_v2a_mcr_v0_full_local")
        safe = resolved_values("pcum_v2a_mcr_v0_full_local_safegeom")
        self.assertFalse(full["guard_enabled"])
        self.assertTrue(safe["guard_enabled"])
        self.assertEqual(safe["min_scale"], 2.0)
        self.assertEqual(safe["min_geometry"], 0.4)
        self.assertEqual(safe["mode"], "ACTIVE")

    def test_safegeom_yaml_only_adds_the_guard_method_difference(self):
        paths = [
            os.path.join(ROOT, "experiments", "entertrack", name + ".yaml")
            for name in (
                "pcum_v2a_mcr_v0_full_local",
                "pcum_v2a_mcr_v0_full_local_safegeom",
            )
        ]
        configs = []
        for path in paths:
            with open(path) as handle:
                configs.append(yaml.safe_load(handle))
        guard = configs[1]["TEST"]["MCR"].pop(
            "CURRENT_LARGE_SCALE_GEOMETRY_GUARD")
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(guard, {
            "ENABLED": True,
            "MIN_SCALE": 2.0,
            "MIN_GEOMETRY": 0.4,
        })

    def test_scoring_switching_and_schedule_parameters_are_unchanged(self):
        expected = {
            "LOCAL_INTERVAL": 10,
            "LOCAL_SCALES": [1.5, 2.0, 3.0],
            "CONFIRM_FRAMES": 2,
            "VERIFY_WINDOW": 3,
            "SWITCH_MARGIN": 0.05,
            "VISUAL_WEIGHT": 0.55,
            "REMOTE_WEIGHT": 0.20,
            "MOTION_WEIGHT": 0.15,
            "GEOMETRY_WEIGHT": 0.10,
            "MIN_VISUAL_SCORE": 0.30,
            "MIN_CANDIDATE_SCORE": 0.50,
        }
        for name in MCR_MODES:
            _, resolved = resolve_config(name)
            with self.subTest(config=name):
                for key, value in expected.items():
                    self.assertEqual(getattr(resolved.TEST.MCR, key), value, key)


if __name__ == "__main__":
    unittest.main()
