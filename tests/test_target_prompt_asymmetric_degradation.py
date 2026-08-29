import copy
import os
import sys
import tempfile
import unittest

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.models.entertrack import build_entertrack  # noqa: E402
from lib.train.actors.entertrack_threemdot import (  # noqa: E402
    EnTeRTrackActorThreeMDOT,
)
from lib.train.base_functions import get_optimizer_scheduler  # noqa: E402
from lib.train.target_prompt_asymmetric_degradation import (  # noqa: E402
    _normalized_box_to_pixels,
    apply_e3_d1_asymmetric_degradation,
)
from lib.utils.box_ops import giou_loss  # noqa: E402
from lib.utils.focal_loss import FocalLoss  # noqa: E402


CONFIG_DIR = os.path.join(ROOT, "experiments", "entertrack")


def load_config(name):
    resolved = copy.deepcopy(cfg)
    update_config_from_file(
        os.path.join(CONFIG_DIR, name + ".yaml"), base_cfg=resolved)
    return resolved


def make_data(batch_size=2):
    images = torch.arange(
        3 * batch_size * 3 * 16 * 16, dtype=torch.float32
    ).reshape(3, batch_size, 3, 16, 16)
    annotations = torch.tensor(
        [[[[0.25, 0.25, 0.50, 0.50]] for _ in range(batch_size)]
         for _ in range(3)], dtype=torch.float32)
    return {
        "search_images": images,
        "search_anno": annotations,
        "template_images": torch.randn(3, batch_size, 3, 8, 8),
        "template_anno": annotations.clone(),
        "target_id": ["target_{}".format(index)
                      for index in range(batch_size)],
        "view_ids": [[view] * batch_size for view in ("A", "B", "C")],
    }


class AsymmetricDegradationTests(unittest.TestCase):
    def setUp(self):
        self.e3 = load_config("target_prompt_collaboration_e3")
        self.d1 = load_config("target_prompt_collaboration_e3_d1")

    def test_default_off_is_exact_identity(self):
        data = make_data()
        effective, audit = apply_e3_d1_asymmetric_degradation(
            data, self.e3, training=True)
        self.assertIs(effective, data)
        self.assertIs(effective["search_images"], data["search_images"])
        self.assertFalse(audit["enabled"])
        self.assertFalse(audit["applied"])

    def test_validation_is_exact_identity_when_enabled(self):
        data = make_data()
        effective, audit = apply_e3_d1_asymmetric_degradation(
            data, self.d1, training=False)
        self.assertIs(effective, data)
        self.assertIs(effective["search_images"], data["search_images"])
        self.assertTrue(audit["enabled"])
        self.assertFalse(audit["applied"])
        self.assertEqual(audit["selected_triplets"], 0)

    def test_b2_changes_exactly_one_view_of_one_triplet(self):
        data = make_data(batch_size=2)
        original_images = data["search_images"].clone()
        original_template = data["template_images"]
        original_annotations = data["search_anno"]
        generator = torch.Generator().manual_seed(41)
        effective, audit = apply_e3_d1_asymmetric_degradation(
            data, self.d1, training=True, generator=generator)

        self.assertIsNot(effective, data)
        self.assertEqual(audit["selected_triplets"], 1)
        self.assertEqual(audit["selected_ratio"], 0.5)
        self.assertEqual(sum(audit[key] for key in (
            "weak_view_A", "weak_view_B", "weak_view_C")), 1)
        self.assertEqual(audit["exactly_one_view_violations"], 0)
        changed = (effective["search_images"] != original_images).flatten(2).any(2)
        self.assertEqual(int(changed.sum().item()), 1)
        self.assertEqual(int(changed.any(0).sum().item()), 1)
        self.assertIs(effective["template_images"], original_template)
        self.assertIs(effective["search_anno"], original_annotations)

        changed_index = changed.nonzero(as_tuple=False)[0]
        view_index, batch_index = [int(value.item()) for value in changed_index]
        output = effective["search_images"][view_index, batch_index]
        source = original_images[view_index, batch_index]
        torch.testing.assert_close(output[:, 4:12, 4:12], torch.zeros_like(
            output[:, 4:12, 4:12]), rtol=0, atol=0)
        outside = torch.ones_like(output, dtype=torch.bool)
        outside[:, 4:12, 4:12] = False
        torch.testing.assert_close(output[outside], source[outside], rtol=0,
                                   atol=0)

    def test_b4_changes_exactly_half_the_triplets(self):
        data = make_data(batch_size=4)
        source = data["search_images"].clone()
        effective, audit = apply_e3_d1_asymmetric_degradation(
            data, self.d1, training=True,
            generator=torch.Generator().manual_seed(7))
        changed = (effective["search_images"] != source).flatten(2).any(2)
        self.assertEqual(audit["selected_triplets"], 2)
        self.assertEqual(int(changed.sum().item()), 2)
        self.assertEqual(int(changed.any(0).sum().item()), 2)
        self.assertTrue(torch.all(changed.sum(0) <= 1))

    def test_normalized_bbox_conversion_and_clipping(self):
        pixel_box, clipped = _normalized_box_to_pixels(
            torch.tensor([0.25, 0.25, 0.50, 0.50]), 16, 16)
        self.assertEqual(pixel_box, (4, 4, 12, 12))
        self.assertFalse(clipped)
        pixel_box, clipped = _normalized_box_to_pixels(
            torch.tensor([-0.10, 0.75, 0.40, 0.40]), 20, 10)
        self.assertEqual(pixel_box, (0, 15, 4, 20))
        self.assertTrue(clipped)

    def test_odd_batch_and_invalid_bbox_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "even positive local batch"):
            apply_e3_d1_asymmetric_degradation(
                make_data(batch_size=3), self.d1, training=True)
        data = make_data()
        data["search_anno"][:] = float("nan")
        with self.assertRaisesRegex(ValueError, "four finite values"):
            apply_e3_d1_asymmetric_degradation(
                data, self.d1, training=True,
                generator=torch.Generator().manual_seed(1))

    def test_frozen_constants_fail_closed(self):
        mutations = (
            ("TRIPLET_RATIO", 0.75, "TRIPLET_RATIO"),
            ("WEAK_VIEWS_PER_TRIPLET", 2, "exactly one weak view"),
            ("OCCLUSION_BOX_SCALE", 1.2, "OCCLUSION_BOX_SCALE"),
            ("FILL_VALUE_NORMALIZED", -1.0, "fill value"),
        )
        for field, value, message in mutations:
            altered = copy.deepcopy(self.d1)
            setattr(altered.TRAIN.TARGET_PROMPT_COLLABORATION.
                    ASYMMETRIC_DEGRADATION, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, message):
                apply_e3_d1_asymmetric_degradation(
                    make_data(), altered, training=True)

    def test_d1_resolved_config_preserves_e3_contract(self):
        self.assertFalse(
            self.e3.TRAIN.TARGET_PROMPT_COLLABORATION.
            ASYMMETRIC_DEGRADATION.ENABLED)
        self.assertTrue(
            self.d1.TRAIN.TARGET_PROMPT_COLLABORATION.
            ASYMMETRIC_DEGRADATION.ENABLED)
        paths = (
            "B0_CHECKPOINT",
            "MODEL.TARGET_PROMPT_COLLABORATION",
            "MODEL.BACKBONE",
            "MODEL.HEAD",
            "TRAIN.LR",
            "TRAIN.WEIGHT_DECAY",
            "TRAIN.EPOCH",
            "TRAIN.TOTAL_EPOCH",
            "TRAIN.LR_DROP_EPOCH",
            "TRAIN.BATCH_SIZE",
            "TRAIN.OPTIMIZER",
            "TRAIN.GIOU_WEIGHT",
            "TRAIN.L1_WEIGHT",
            "TRAIN.FOCAL_WEIGHT",
            "TRAIN.MULTIVIEW",
            "TRAIN.TARGET_PROMPT_COLLABORATION.FREEZE_LOCAL",
            "TRAIN.TARGET_PROMPT_COLLABORATION.DETACH_REMOTE",
            "TRAIN.TARGET_PROMPT_COLLABORATION.LR",
            "DATA",
            "TEST.TARGET_PROMPT_COLLABORATION",
        )
        for path in paths:
            with self.subTest(path=path):
                left = self.e3
                right = self.d1
                for key in path.split("."):
                    left = getattr(left, key)
                    right = getattr(right, key)
                self.assertEqual(left, right)
        self.assertEqual(
            self.d1.MODEL.TARGET_PROMPT_COLLABORATION.PROMPT_K, 8)
        self.assertTrue(
            self.d1.TEST.TARGET_PROMPT_COLLABORATION.SAFE_COMMIT)


class AsymmetricDegradationRealModelSmoke(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_b0_initialized_actor_train_val_checkpoint_smoke(self):
        resolved = load_config("target_prompt_collaboration_e3_d1")
        checkpoint_path = resolved.B0_CHECKPOINT
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(ROOT, checkpoint_path)
        self.assertTrue(os.path.isfile(checkpoint_path))

        device = torch.device("cuda:0")
        torch.manual_seed(20260829)
        torch.cuda.manual_seed_all(20260829)
        model = build_entertrack(resolved, training=True).to(device)
        initialization = model.initialization_audit
        self.assertTrue(initialization["strict_full_load"])
        self.assertEqual(os.path.realpath(initialization["checkpoint_path"]),
                         os.path.realpath(checkpoint_path))
        self.assertGreater(initialization["fresh_adapter_key_count"], 0)
        optimizer, _ = get_optimizer_scheduler(model, resolved)
        trainable = {
            name: parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(sum(value.numel() for value in trainable.values()),
                         148993)
        self.assertTrue(all(name.startswith("target_prompt_collaboration.")
                            for name in trainable))

        settings = type("Settings", (), {"batchsize": 2})()
        actor = EnTeRTrackActorThreeMDOT(
            net=model,
            objective={
                "giou": giou_loss,
                "l1": torch.nn.functional.l1_loss,
                "focal": FocalLoss(),
            },
            loss_weight={
                "giou": resolved.TRAIN.GIOU_WEIGHT,
                "l1": resolved.TRAIN.L1_WEIGHT,
                "focal": resolved.TRAIN.FOCAL_WEIGHT,
            },
            settings=settings,
            cfg=resolved,
        )
        data = {
            "template_images": torch.randn(
                3, 2, 3, 128, 128, device=device),
            "search_images": torch.randn(
                3, 2, 3, 256, 256, device=device),
            "template_anno": torch.tensor(
                [[[0.30, 0.30, 0.20, 0.20]] * 2] * 3, device=device),
            "search_anno": torch.tensor(
                [[[0.35, 0.35, 0.25, 0.25]] * 2] * 3, device=device),
            "target_id": ["synthetic_0", "synthetic_1"],
            "view_ids": [[view] * 2 for view in ("A", "B", "C")],
            "search_frame_ids": [[100, 101]],
            "epoch": 1,
        }

        frozen_before = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if not name.startswith("target_prompt_collaboration.")
        }
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()
        loss, train_status = actor(data)
        self.assertTrue(bool(torch.isfinite(loss).item()))
        loss.backward()
        gradients = [parameter.grad for parameter in trainable.values()]
        self.assertTrue(any(value is not None for value in gradients))
        self.assertTrue(all(value is None or bool(torch.isfinite(value).all())
                            for value in gradients))
        optimizer.step()
        torch.cuda.synchronize(device)
        self.assertEqual(train_status["E3D1/applied"], 1.0)
        self.assertEqual(train_status["E3D1/selected_triplets"], 1.0)
        self.assertEqual(train_status["E3D1/selected_ratio"], 0.5)
        self.assertEqual(train_status["E3/prompt_k"], 8.0)
        self.assertEqual(train_status["Plain/search_tokens"], 256.0)
        self.assertEqual(train_status["Plain/center_map_side"], 16.0)
        self.assertEqual(train_status["Plain/pcum_present"], 0.0)
        self.assertEqual(train_status["Plain/pruning_present"], 0.0)
        for name, value in frozen_before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name].cpu()),
                            name)

        saved_adapter = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name.startswith("target_prompt_collaboration.")
        }
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = os.path.join(directory, "e3_d1_smoke.pth.tar")
            torch.save({
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": 1,
            }, path)
            self.assertGreater(os.path.getsize(path), 0)
            first_parameter = next(iter(trainable.values()))
            with torch.no_grad():
                first_parameter.add_(1.0)
            payload = torch.load(path, map_location=device)
            model.load_state_dict(payload["net"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            self.assertEqual(payload["epoch"], 1)
        for name, value in saved_adapter.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name].cpu()),
                            name)

        model.eval()
        with torch.no_grad():
            validation_loss, validation_status = actor(data)
        self.assertTrue(bool(torch.isfinite(validation_loss).item()))
        self.assertEqual(validation_status["E3D1/enabled"], 1.0)
        self.assertEqual(validation_status["E3D1/applied"], 0.0)
        self.assertEqual(validation_status["E3D1/selected_triplets"], 0.0)
        print("E3_D1_REAL_SMOKE={}".format({
            "loss": float(loss.detach().cpu().item()),
            "validation_loss": float(validation_loss.cpu().item()),
            "trainable_parameters": 148993,
            "search_tokens": 256,
            "prompt_k": 8,
            "selected_triplets": 1,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint_save_load": "PASS",
            "validation_bypass": "PASS",
        }))


if __name__ == "__main__":
    unittest.main()
