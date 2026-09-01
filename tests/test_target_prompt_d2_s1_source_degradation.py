import copy
import os
import sys
import unittest

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.train.target_prompt_asymmetric_degradation import (  # noqa: E402
    apply_e3_d1_asymmetric_degradation,
)
from lib.train.target_prompt_d2_s1_source_degradation import (  # noqa: E402
    COVERAGE,
    ORIENTATION_NAMESPACE,
    apply_e3_d2_s1_source_degradation,
    d2_p1_clean_sample_id,
    orientation_for_sample,
)
from tracking.target_prompt_d2_p2_partial_degradation import (  # noqa: E402
    apply_partial_occlusion as apply_frozen_d2_p2,
    orientation_for_sample as frozen_orientation,
)


CONFIG_DIR = os.path.join(ROOT, "experiments", "entertrack")


def load_config(name):
    resolved = copy.deepcopy(cfg)
    update_config_from_file(
        os.path.join(CONFIG_DIR, name + ".yaml"), base_cfg=resolved)
    return resolved


def make_data(batch_size=2):
    images = torch.arange(
        1, 3 * batch_size * 3 * 16 * 16 + 1, dtype=torch.float32
    ).reshape(3, batch_size, 3, 16, 16)
    annotations = torch.tensor(
        [[[[0.25, 0.25, 0.50, 0.50]] for _ in range(batch_size)]
         for _ in range(3)], dtype=torch.float32)
    return {
        "search_images": images,
        "search_anno": annotations,
        "template_images": torch.randn(3, batch_size, 3, 8, 8),
        "template_anno": annotations.clone(),
        "target_id": ["md{:04d}".format(3005 + index)
                      for index in range(batch_size)],
        "view_ids": [[view] * batch_size for view in ("A", "B", "C")],
        "template_frame_ids": [[4 + index for index in range(batch_size)]],
        "search_frame_ids": [[5 + index for index in range(batch_size)]],
    }


class D2S1SourceDegradationTests(unittest.TestCase):
    def setUp(self):
        self.e3 = load_config("target_prompt_collaboration_e3")
        self.d1 = load_config("target_prompt_collaboration_e3_d1")
        self.d2 = load_config("target_prompt_collaboration_e3_d2_s1")

    def test_frozen_p50_pixel_identity_with_d2_p2(self):
        data = make_data()
        source = data["search_images"].clone()
        effective, audit = apply_e3_d2_s1_source_degradation(
            data, self.d2, training=True,
            generator=torch.Generator().manual_seed(41))
        changed = (effective["search_images"] != source).flatten(2).any(2)
        self.assertEqual(int(changed.sum().item()), 1)
        view_index, batch_index = [
            int(value.item()) for value in changed.nonzero()[0]]
        sample_id = d2_p1_clean_sample_id(data, view_index, batch_index)
        frozen, frozen_audit = apply_frozen_d2_p2(
            source[view_index, batch_index],
            data["search_anno"][view_index, batch_index],
            "P50", sample_id)
        self.assertTrue(torch.equal(
            effective["search_images"][view_index, batch_index], frozen))
        self.assertEqual(audit["sample_ids"], [sample_id])
        self.assertEqual(
            sum(audit["orientation_{}".format(name)]
                for name in ("left", "right", "top", "bottom")), 1)
        self.assertEqual(
            audit["realized_bbox_coverage_mean"],
            frozen_audit["realized_coverage"])

    def test_sha256_orientation_and_identity_format_are_exact(self):
        samples = (
            "d2p1-train-md3005-1-000005-clean",
            "d2p1-train-md3010-2-000101-clean",
            "d2p1-train-md3020-3-000999-clean",
        )
        self.assertEqual(ORIENTATION_NAMESPACE, "D2-P2-orientation-v1")
        for sample_id in samples:
            with self.subTest(sample_id=sample_id):
                self.assertEqual(
                    orientation_for_sample(sample_id),
                    frozen_orientation(sample_id))
                self.assertEqual(
                    orientation_for_sample(sample_id),
                    orientation_for_sample(sample_id))
        data = make_data()
        self.assertEqual(
            d2_p1_clean_sample_id(data, 0, 0),
            "d2p1-train-md3005-1-000005-clean")
        self.assertEqual(
            d2_p1_clean_sample_id(data, 2, 1),
            "d2p1-train-md3006-3-000006-clean")

    def test_e3_off_and_validation_are_exact_identity(self):
        data = make_data()
        effective, audit = apply_e3_d2_s1_source_degradation(
            data, self.e3, training=True)
        self.assertIs(effective, data)
        self.assertIs(effective["search_images"], data["search_images"])
        self.assertFalse(audit["enabled"])
        effective, audit = apply_e3_d2_s1_source_degradation(
            data, self.d2, training=False)
        self.assertIs(effective, data)
        self.assertIs(effective["search_images"], data["search_images"])
        self.assertTrue(audit["enabled"])
        self.assertFalse(audit["applied"])

    def test_historical_d1_p100_route_is_exact_regression(self):
        data = make_data()
        direct, direct_audit = apply_e3_d1_asymmetric_degradation(
            data, self.d1, training=True,
            generator=torch.Generator().manual_seed(77))
        routed_d1, routed_audit = apply_e3_d1_asymmetric_degradation(
            data, self.d1, training=True,
            generator=torch.Generator().manual_seed(77))
        routed, d2_audit = apply_e3_d2_s1_source_degradation(
            routed_d1, self.d1, training=True)
        self.assertTrue(torch.equal(
            direct["search_images"], routed["search_images"]))
        self.assertEqual(direct_audit, routed_audit)
        self.assertIs(routed, routed_d1)
        self.assertFalse(d2_audit["enabled"])

    def test_exactly_one_weak_view_and_metadata_unchanged(self):
        data = make_data(batch_size=4)
        source = data["search_images"].clone()
        template = data["template_images"]
        annotation = data["search_anno"]
        effective, audit = apply_e3_d2_s1_source_degradation(
            data, self.d2, training=True,
            generator=torch.Generator().manual_seed(7))
        changed = (effective["search_images"] != source).flatten(2).any(2)
        self.assertEqual(audit["selected_triplets"], 2)
        self.assertEqual(audit["selected_ratio"], 0.5)
        self.assertEqual(int(changed.sum().item()), 2)
        self.assertEqual(int(changed.any(0).sum().item()), 2)
        self.assertTrue(bool(torch.all(changed.sum(0) <= 1).item()))
        self.assertIs(effective["template_images"], template)
        self.assertIs(effective["search_anno"], annotation)
        self.assertTrue(audit["template_unchanged"])
        self.assertTrue(audit["annotation_unchanged"])
        self.assertAlmostEqual(
            audit["realized_bbox_coverage_mean"], COVERAGE, places=12)

    def test_identity_metadata_is_required_and_fail_closed(self):
        data = make_data()
        del data["search_frame_ids"]
        with self.assertRaisesRegex(ValueError, "search_frame_ids"):
            apply_e3_d2_s1_source_degradation(
                data, self.d2, training=True,
                generator=torch.Generator().manual_seed(1))
        data = make_data()
        data["view_ids"] = [["X", "X"] for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "canonical A/B/C"):
            apply_e3_d2_s1_source_degradation(
                data, self.d2, training=True,
                generator=torch.Generator().manual_seed(41))

    def test_frozen_source_constants_and_mutual_exclusion(self):
        mutations = (
            ("CANDIDATE", "P25"),
            ("COVERAGE", 0.75),
            ("TRIPLET_RATIO", 0.75),
            ("WEAK_VIEWS_PER_TRIPLET", 2),
            ("FILL_VALUE_NORMALIZED", -1.0),
            ("BLOCK_MECHANISM", "random"),
            ("ORIENTATION_NAMESPACE", "different"),
        )
        for field, value in mutations:
            altered = copy.deepcopy(self.d2)
            setattr(altered.TRAIN.TARGET_PROMPT_COLLABORATION.
                    SOURCE_DEGRADATION, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "freezes {}".format(field)):
                apply_e3_d2_s1_source_degradation(
                    make_data(), altered, training=True,
                    generator=torch.Generator().manual_seed(1))
        both = copy.deepcopy(self.d2)
        both.TRAIN.TARGET_PROMPT_COLLABORATION.\
            ASYMMETRIC_DEGRADATION.ENABLED = True
        with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
            apply_e3_d2_s1_source_degradation(
                make_data(), both, training=True,
                generator=torch.Generator().manual_seed(1))

    def test_d2_config_changes_only_p100_to_p50_source(self):
        self.assertTrue(self.d1.TRAIN.TARGET_PROMPT_COLLABORATION.
                        ASYMMETRIC_DEGRADATION.ENABLED)
        self.assertFalse(self.d1.TRAIN.TARGET_PROMPT_COLLABORATION.
                         SOURCE_DEGRADATION.ENABLED)
        self.assertFalse(self.d2.TRAIN.TARGET_PROMPT_COLLABORATION.
                         ASYMMETRIC_DEGRADATION.ENABLED)
        self.assertTrue(self.d2.TRAIN.TARGET_PROMPT_COLLABORATION.
                        SOURCE_DEGRADATION.ENABLED)
        paths = (
            "B0_CHECKPOINT", "MODEL", "TRAIN.LR", "TRAIN.WEIGHT_DECAY",
            "TRAIN.EPOCH", "TRAIN.TOTAL_EPOCH", "TRAIN.LR_DROP_EPOCH",
            "TRAIN.BATCH_SIZE", "TRAIN.OPTIMIZER", "TRAIN.GIOU_WEIGHT",
            "TRAIN.L1_WEIGHT", "TRAIN.FOCAL_WEIGHT", "TRAIN.MULTIVIEW",
            "TRAIN.TARGET_PROMPT_COLLABORATION.ENABLED",
            "TRAIN.TARGET_PROMPT_COLLABORATION.FREEZE_LOCAL",
            "TRAIN.TARGET_PROMPT_COLLABORATION.DETACH_REMOTE",
            "TRAIN.TARGET_PROMPT_COLLABORATION.LR", "DATA",
            "TEST.TARGET_PROMPT_COLLABORATION",
        )
        for path in paths:
            with self.subTest(path=path):
                left, right = self.d1, self.d2
                for key in path.split("."):
                    left, right = getattr(left, key), getattr(right, key)
                self.assertEqual(left, right)
        self.assertEqual(
            self.d2.MODEL.TARGET_PROMPT_COLLABORATION.PROMPT_K, 8)
        self.assertTrue(
            self.d2.TEST.TARGET_PROMPT_COLLABORATION.SAFE_COMMIT)


if __name__ == "__main__":
    unittest.main()
