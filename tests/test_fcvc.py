import copy
import unittest

import torch
import torch.nn as nn

from lib.models.entertrack.fcvc import (
    FCVCConfig,
    FCVCModel,
    SafeCommitRuntime,
    build_sender_bundle,
    capture_taps,
    reconstruction_loss,
    state_digest,
    validate_sender_pair,
)


class TinyPatch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(3, dim)

    def forward(self, image):
        out_tokens = (image.shape[-2] // 16) * (image.shape[-1] // 16)
        pooled = image.mean(dim=(-1, -2)).unsqueeze(1)
        return self.proj(pooled).repeat(1, out_tokens, 1)


class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Linear(dim, dim)

    def forward(self, x, global_index_template, global_index_search, mask=None,
                ce_template_mask=None, training=False, keep_ratio_search=1.0,
                temperature=2.0, frozen_token=None):
        out = x + 0.1 * self.ff(self.norm(x))
        return out, global_index_search, None, None, None, mask, frozen_token


class TinyBackbone(nn.Module):
    def __init__(self, dim=192, template_tokens=64, search_tokens=256):
        super().__init__()
        self.patch_embed = TinyPatch(dim)
        self.pos_embed_z = nn.Parameter(torch.zeros(1, template_tokens, dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, search_tokens, dim))
        self.blocks = nn.Sequential(*[TinyBlock(dim) for _ in range(6)])
        self.norm = nn.LayerNorm(dim)
        self.cat_mode = "direct"
        self.add_cls_token = False
        self.add_sep_seg = False
        self.pos_drop = nn.Identity()

    def forward(self, z, x):
        from lib.models.entertrack.utils import combine_tokens

        search = self.patch_embed(x) + self.pos_embed_x
        template = self.patch_embed(z) + self.pos_embed_z
        tokens = combine_tokens(template, search, mode="direct")
        for block in self.blocks:
            tokens, _, _, _, _, _, _ = block(
                tokens, torch.arange(64).view(1, -1).repeat(x.shape[0], 1),
                torch.arange(256).view(1, -1).repeat(x.shape[0], 1))
        return self.norm(tokens)


class TinyHead(nn.Module):
    def forward(self, tokens):
        search = tokens[:, -256:]
        return {
            "bbox": search.mean(dim=(1, 2)),
            "score": search.mean(dim=-1).amax(dim=-1),
            "feature": search,
        }


def local_record(batch=2, dtype=torch.float32):
    torch.manual_seed(7)
    mid = torch.randn(batch, 256, 192, dtype=dtype)
    high = torch.randn(batch, 256, 192, dtype=dtype)
    response = torch.sigmoid(torch.randn(batch, 1, 16, 16, dtype=dtype))
    conf_unc = torch.cat((response, response * 0.0 + 0.5), dim=1)
    proto = high.mean(dim=1)
    return {
        "template_mid": torch.randn(batch, 64, 192, dtype=dtype),
        "template_high": torch.randn(batch, 64, 192, dtype=dtype),
        "mid_search": mid,
        "high_search": high,
        "response_map": response,
        "confidence_uncertainty": conf_unc,
        "target_prototype": proto,
        "local_output": {
            "bbox": torch.randn(batch, 4, dtype=dtype),
            "score": torch.rand(batch, dtype=dtype),
            "feature": high,
        },
    }


def sender(view_id, timestamp=3, batch=2, dtype=torch.float32, zero=False):
    value = torch.zeros if zero else torch.randn
    mid = value(batch, 256, 192, dtype=dtype)
    high = value(batch, 256, 192, dtype=dtype)
    response = torch.sigmoid(value(batch, 1, 16, 16, dtype=dtype))
    return build_sender_bundle(
        mid, high, response,
        view_id=torch.full((batch,), view_id, dtype=torch.int16),
        timestamp=torch.full((batch,), timestamp, dtype=torch.int64),
    )


class FCVCContractTest(unittest.TestCase):
    def test_dense_tap_replay_identity(self):
        torch.manual_seed(1)
        backbone = TinyBackbone()
        template = torch.randn(2, 3, 128, 128)
        search = torch.randn(2, 3, 256, 256)
        full = backbone(template, search)
        taps = capture_taps(backbone, template, search)
        self.assertTrue(torch.equal(taps.final_tokens, full))
        self.assertTrue(torch.equal(taps.replay_tokens, full))
        self.assertEqual(tuple(taps.mid_tokens[:, :64].shape), (2, 64, 192))
        self.assertEqual(tuple(taps.mid_tokens[:, -256:].shape), (2, 256, 192))

    def test_sender_bundle_schema_and_no_gt(self):
        bundle = sender(1, batch=1)
        schema = bundle.schema()
        self.assertEqual(schema["mid_features"]["shape"], (1, 256, 192))
        self.assertEqual(schema["high_features"]["dtype"], "float16")
        self.assertTrue(all(item["detached"] for item in schema.values()))
        self.assertNotIn("gt", " ".join(schema.keys()).lower())
        ok, reason = validate_sender_pair((sender(1, batch=1), sender(2, batch=1)), 1)
        self.assertTrue(ok, reason)

    def test_default_off_and_no_remote_identity(self):
        local = local_record(batch=1)
        disabled = FCVCModel(FCVCConfig(enabled=False))
        self.assertIs(disabled(local)["reported_output"], local["local_output"])
        enabled = FCVCModel(FCVCConfig(enabled=True))
        no_remote = enabled(local, sender_bundles=())
        self.assertIs(no_remote["reported_output"], local["local_output"])

    def test_zero_residual_template_and_local_immutability(self):
        torch.manual_seed(2)
        local = local_record(batch=1)
        before_mid = local["mid_search"].clone()
        before_high = local["high_search"].clone()
        model = FCVCModel(FCVCConfig(enabled=True))
        out = model(local, (sender(1, batch=1), sender(2, batch=1)),
                    force_zero_residual=True)
        self.assertIs(out["reported_output"], local["local_output"])
        self.assertTrue(torch.equal(out["mid_writer"]["template_tokens"],
                                    local["template_mid"]))
        self.assertTrue(torch.equal(out["high_writer"]["template_tokens"],
                                    local["template_high"]))
        self.assertTrue(torch.equal(local["mid_search"], before_mid))
        self.assertTrue(torch.equal(local["high_search"], before_high))
        self.assertNotEqual(out["mid_writer"]["search_tokens"].data_ptr(),
                            local["mid_search"].data_ptr())

    def test_coordinate_isolation_and_null_behavior(self):
        torch.manual_seed(3)
        model = FCVCModel(FCVCConfig(enabled=True))
        local = local_record(batch=1)
        first = sender(1, batch=1, zero=True)
        second = sender(2, batch=1, zero=True)
        out = model(local, (first, second), force_null=True)
        coords = out["mid_block"]["sample_coordinates"]
        self.assertTrue(bool(torch.isfinite(coords).all()))
        self.assertGreaterEqual(float(coords.min()), 0.0)
        self.assertLessEqual(float(coords.max()), 1.0)
        self.assertTrue(torch.equal(out["reported_output"]["feature"],
                                    local["local_output"]["feature"]))
        self.assertTrue((out["mid_block"]["attention_weights"][:, :, -1] == 1).all())

    def test_sender_permutation_consistency_by_view_id(self):
        torch.manual_seed(4)
        model = FCVCModel(FCVCConfig(enabled=True))
        model.eval()
        local = local_record(batch=1)
        a = sender(1, batch=1)
        b = sender(2, batch=1)
        with torch.no_grad():
            out_ab = model(local, (a, b))["reported_output"]["fcvc_search_tokens"]
            out_ba = model(local, (b, a))["reported_output"]["fcvc_search_tokens"]
        self.assertTrue(torch.allclose(out_ab, out_ba, atol=1e-5, rtol=1e-5))

    def test_shape_dtype_device_and_determinism(self):
        for dtype in (torch.float32, torch.float16):
            for batch in (1, 2):
                torch.manual_seed(5)
                model = FCVCModel(FCVCConfig(enabled=True))
                local = local_record(batch=batch, dtype=dtype)
                bundles = (sender(1, batch=batch, dtype=dtype),
                           sender(2, batch=batch, dtype=dtype))
                out = model(local, bundles)
                self.assertEqual(tuple(out["queries"].shape), (batch, 8, 128))
                self.assertTrue(torch.isfinite(out["reported_output"]["fcvc_search_tokens"]).all())
        torch.manual_seed(6)
        model1 = FCVCModel(FCVCConfig(enabled=True))
        local1 = local_record(batch=1)
        bundles1 = (sender(1, batch=1), sender(2, batch=1))
        out1 = model1(local1, bundles1)
        torch.manual_seed(6)
        model2 = FCVCModel(FCVCConfig(enabled=True))
        local2 = local_record(batch=1)
        bundles2 = (sender(1, batch=1), sender(2, batch=1))
        out2 = model2(local2, bundles2)
        self.assertTrue(torch.equal(out1["reported_output"]["fcvc_search_tokens"],
                                    out2["reported_output"]["fcvc_search_tokens"]))

    def test_freeze_gradient_teacher_detach_and_checkpoint_isolation(self):
        torch.manual_seed(8)
        model = FCVCModel(FCVCConfig(enabled=True))
        backbone = TinyBackbone()
        head = TinyHead()
        for module in (backbone, head):
            for p in module.parameters():
                p.requires_grad_(False)
        local = local_record(batch=1)
        out = model(local, (sender(1, batch=1), sender(2, batch=1)))
        loss = out["queries"].square().mean()
        loss.backward()
        whitelist = model.trainable_parameter_names(include_teacher=False)
        self.assertTrue(any(name.startswith("matcher.") for name in whitelist))
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))
                            for p in model.teacher.parameters()))
        gt = torch.ones(1, 1, 16, 16)
        teacher_slots = model.teacher(
            [local["mid_search"], local["mid_search"], local["mid_search"]],
            [local["high_search"], local["high_search"], local["high_search"]],
            gt)
        model.zero_grad(set_to_none=True)
        out = model(local, (sender(1, batch=1), sender(2, batch=1)))
        recon = reconstruction_loss(out["queries"], teacher_slots)
        recon.backward()
        self.assertTrue(all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))
                            for p in model.teacher.parameters()))
        model.zero_grad(set_to_none=True)
        teacher_track = model.teacher.tracking_residual(
            local["high_search"].detach(), teacher_slots).square().mean()
        teacher_track.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.teacher.parameters()))
        fcvc_state = model.fcvc_state_dict(include_teacher=False)
        self.assertTrue(fcvc_state)
        self.assertFalse(any(k.startswith("backbone") or k.startswith("box_head")
                             for k in fcvc_state))

    def test_safe_commit_multiframe_reported_output_not_state(self):
        e0 = SafeCommitRuntime({"bbox": [0, 0, 10, 10], "crop": [0, 0, 20, 20],
                                "sender_source": "local"})
        fcvc = SafeCommitRuntime(copy.deepcopy(e0.state))
        for frame in range(5):
            local = {
                "bbox": [frame, frame, 10, 10],
                "crop": [frame, frame, 20, 20],
                "sender_source": "local-{}".format(frame),
            }
            reported = {
                "bbox": [9999, -9999, 1, 1],
                "crop": ["must", "not", "commit"],
                "sender_source": "collab",
            }
            e0.commit(local, local)
            fcvc.commit(local, reported)
            self.assertEqual(state_digest(e0.state), state_digest(fcvc.state))
            self.assertEqual(e0.state["crop"], fcvc.state["crop"])


if __name__ == "__main__":
    unittest.main()
