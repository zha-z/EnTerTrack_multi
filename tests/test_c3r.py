import copy
import inspect
import io
import os
import struct
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from lib.config.entertrack.config import get_default_config, update_config_from_file
from lib.models.entertrack.c3r import (
    C3R,
    C3R_PACKET_BYTES,
    C3R_PROMPT_SHAPE,
    C3RMessage,
    C3RPacketCodec,
    CompactMessageEncoder,
    CommunicationPerturbation,
    MessageAccounting,
    PacketValidationError,
    gate_ranking_loss,
)
from lib.models.entertrack.entertrack import EnTeRTrack, build_entertrack
from lib.train.base_functions import get_optimizer_scheduler
from lib.train.c3r_freeze import assert_c3r_optimizer_membership
from lib.train.actors.entertrack_threemdot import EnTeRTrackActorThreeMDOT
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = (
    "/data/zjy/multi/output/entertrack_single_lasot_ft_cons/checkpoints/"
    "train/entertrack/entertrack_threemdot_lasot_ft_cons/EnTeRTrack_ep0004.pth.tar"
)


def load_config(name):
    config = get_default_config()
    update_config_from_file(
        os.path.join(ROOT, "experiments/entertrack/" + name + ".yaml"),
        base_cfg=config)
    return config


def flatten_config(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            child = key if not prefix else prefix + "." + key
            result.update(flatten_config(item, child))
        return result
    return {prefix: value}


class C3RConfigAuditTest(unittest.TestCase):
    def test_resolved_config_diffs_and_legacy_disable(self):
        canonical = get_default_config()
        update_config_from_file(
            os.path.join(ROOT, "experiments/entertrack/entertrack_threemdot_lasot_ft_cons.yaml"),
            base_cfg=canonical)
        e0 = load_config("entertrack_c3r_e0")
        canonical_flat = flatten_config(canonical)
        e0_flat = flatten_config(e0)
        e0_diff = {key for key in canonical_flat if canonical_flat[key] != e0_flat[key]}
        self.assertEqual(e0_diff, {"BASE_CONFIG", "MODEL_ROLE"})

        configs = {
            name: load_config("entertrack_c3r_" + name)
            for name in ("c0", "c1", "a1", "a2", "smoke")
        }
        for name, config in configs.items():
            self.assertFalse(config.MODEL.PCUM.ENABLED, name)
            self.assertFalse(config.MODEL.USE_SEARCH_PROMPT, name)
            self.assertFalse(config.TEST.PCUM.USE_REMOTE, name)
            self.assertFalse(config.TEST.PCUM.USE_REMOTE_VISIBLE_MASK, name)
            self.assertFalse(config.TEST.MOTION_STATE.ENABLED, name)
            self.assertFalse(config.TEST.MCR.ENABLED, name)
            self.assertFalse(config.TEST.COOP.ENABLED, name)
            self.assertEqual(config.MODEL.C3R.NUM_PROMPTS, 4)
            self.assertEqual(config.MODEL.C3R.MESSAGE_DIM, 64)
            self.assertEqual(config.MODEL.C3R.MAX_GATE, 0.25)
            self.assertEqual(config.MODEL.C3R.PEER_NORM_CAP, 0.25)
            self.assertEqual(config.MODEL.C3R.AGGREGATE_NORM_CAP, 0.35)

        c0_flat = flatten_config(configs["c0"])
        c1_flat = flatten_config(configs["c1"])
        c0_c1_diff = {key for key in c0_flat if c0_flat[key] != c1_flat[key]}
        self.assertEqual(c0_c1_diff, {
            "MODEL_ROLE", "MODEL.C3R.VARIANT",
            "TRAIN.C3R.PERTURBATIONS_ENABLED", "TEST.SAVE_DIR",
        })
        for ablation in ("a1", "a2"):
            ablation_flat = flatten_config(configs[ablation])
            difference = {
                key for key in c1_flat if c1_flat[key] != ablation_flat[key]
            }
            self.assertEqual(difference, {
                "MODEL_ROLE", "MODEL.C3R.VARIANT", "TEST.SAVE_DIR",
            })

    def test_f0_training_split_matches_frozen_manifest(self):
        path = os.path.join(
            ROOT, "lib/train/data_specs/threemdot/c3r_f0_train.txt")
        with open(path) as handle:
            sequences = [line.strip() for line in handle if line.strip()]
        self.assertEqual(len(sequences), 54)
        targets = {sequence.rsplit("-", 1)[0] for sequence in sequences}
        self.assertEqual(len(targets), 18)
        held_out = {"md3005", "md3019", "md3032", "md3044", "md3059"}
        self.assertTrue(targets.isdisjoint(held_out))
        for target in targets:
            self.assertEqual(
                {sequence for sequence in sequences if sequence.startswith(target + "-")},
                {target + "-1", target + "-2", target + "-3"},
            )
        config = load_config("entertrack_c3r_c0_f0")
        self.assertEqual(
            config.DATA.TRAIN.SPLIT_FILE,
            "lib/train/data_specs/threemdot/c3r_f0_train.txt")


class DummyBackbone(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, z, x, **kwargs):
        batch = x.shape[0]
        base = x.mean(dim=(1, 2, 3), keepdim=True).reshape(batch, 1, 1)
        tokens = base.expand(batch, 320, self.dim) * self.scale
        return tokens, {"dummy_aux": tokens.new_zeros(())}


class DummyCenterHead(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.feat_sz = 16
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, feature, gt_score_map=None):
        score = torch.sigmoid((feature * self.weight[None, :, None, None]).mean(dim=1, keepdim=True))
        size = torch.sigmoid(feature[:, :2])
        offset = feature[:, :2] * 0.0
        flat = score.flatten(1)
        index = flat.argmax(dim=1)
        y = index // self.feat_sz
        x = index % self.feat_sz
        bbox = torch.stack((
            x.float() / self.feat_sz,
            y.float() / self.feat_sz,
            size[:, 0].flatten(1).mean(dim=1),
            size[:, 1].flatten(1).mean(dim=1),
        ), dim=1)
        return score, bbox, size, offset


def make_message(sender_id=1, sequence_hash=1234, frame_id=10,
                 timestamp_ms=330, prompt_offset=0.0):
    prompt = torch.linspace(-1.0, 1.0, 256).reshape(4, 64) + float(prompt_offset)
    quantized, scales, reconstructed = CompactMessageEncoder.quantize(prompt)
    return C3RMessage(
        sender_id=sender_id,
        sequence_hash=sequence_hash,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        bbox=torch.tensor([0.5, 0.5, 0.2, 0.3]),
        bbox_delta=torch.tensor([0.01, -0.01, 0.0, 0.0]),
        quality=torch.tensor([0.8, 0.7, 0.2, 0.3]),
        scales=scales,
        quantized_prompt=quantized,
        prompt=reconstructed,
    )

class C3RProtocolTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.codec = C3RPacketCodec()
        self.message = make_message()

    def test_serialization_roundtrip_offsets_endian_crc(self):
        packet = self.codec.serialize(self.message)
        self.assertEqual(len(packet), 320)
        self.assertEqual(packet[:4], b"C3R1")
        self.assertEqual(packet[4], 1)
        self.assertEqual(packet[5], 1)
        self.assertEqual(struct.unpack_from("<H", packet, 6)[0], 1)
        self.assertEqual(struct.unpack_from("<I", packet, 8)[0], 1234)
        self.assertEqual(struct.unpack_from("<I", packet, 12)[0], 10)
        self.assertEqual(struct.unpack_from("<Q", packet, 16)[0], 330)
        self.assertEqual(struct.unpack_from("<H", packet, 24)[0], 288)
        self.assertEqual(struct.unpack_from("<H", packet, 26)[0], 0)
        parsed = self.codec.parse(packet)
        self.assertEqual(parsed.sender_id, self.message.sender_id)
        self.assertEqual(tuple(parsed.quantized_prompt.shape), C3R_PROMPT_SHAPE)
        self.assertTrue(torch.equal(parsed.quantized_prompt, self.message.quantized_prompt))
        self.assertTrue(torch.allclose(parsed.prompt, self.message.prompt, atol=1e-3, rtol=1e-3))
        corrupted = bytearray(packet)
        corrupted[100] ^= 0x01
        with self.assertRaisesRegex(PacketValidationError, "crc32"):
            self.codec.parse(corrupted)

    def test_protocol_rejection_and_stale(self):
        packet = self.codec.serialize(self.message)
        accepted = self.codec.validate_for_receiver(
            packet, 0, 1234, 10, 330, 33, {}, 4)
        self.assertEqual(accepted.sender_id, 1)
        with self.assertRaisesRegex(PacketValidationError, "self_sender"):
            self.codec.validate_for_receiver(packet, 1, 1234, 10, 330, 33, {}, 4)
        with self.assertRaisesRegex(PacketValidationError, "sequence_hash"):
            self.codec.validate_for_receiver(packet, 0, 999, 10, 330, 33, {}, 4)
        stale = make_message(timestamp_ms=165, frame_id=5)
        with self.assertRaisesRegex(PacketValidationError, "stale"):
            self.codec.validate_for_receiver(stale, 0, 1234, 10, 330, 33, {}, 4)
        with self.assertRaisesRegex(PacketValidationError, "replay"):
            self.codec.validate_for_receiver(packet, 0, 1234, 10, 330, 33, {1: 10}, 4)

    def test_message_accounting(self):
        accounting = MessageAccounting()
        accounting.record_frame(sent=1, received=2, accepted=2,
                                received_by_peer={1: 1, 2: 1})
        report = accounting.report(peers=2, broadcast=True)
        self.assertEqual(report["header_bytes"], 32)
        self.assertEqual(report["predicted_state_and_quality_bytes"], 24)
        self.assertEqual(report["quantization_scales_bytes"], 8)
        self.assertEqual(report["prompt_int8_bytes"], 256)
        self.assertEqual(report["total_bytes"], 320)
        self.assertEqual(report["serialized_bytes_sent"], 320)
        self.assertEqual(report["serialized_bytes_received"], 640)
        self.assertEqual(report["p90_received_bytes_per_frame"], 640)
        self.assertEqual(report["received_bytes_by_peer"], {1: 320, 2: 320})
        zero = MessageAccounting().report()
        self.assertEqual(zero["serialized_bytes_sent"], 0)
        self.assertEqual(zero["serialized_bytes_received"], 0)


class C3RModuleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.tokens = torch.randn(1, 256, 192)
        self.response = torch.sigmoid(torch.randn(1, 1, 16, 16))
        self.context = dict(
            receiver_id=0,
            sequence_hash=1234,
            local_frame_id=10,
            local_timestamp_ms=330,
            frame_interval_ms=33,
            last_frame_by_sender={},
        )

    def collaborate(self, module, messages):
        return module.collaborate(
            self.tokens, self.response, messages, **self.context)

    def test_parameter_counts_and_shapes(self):
        c1 = C3R(variant="c1")
        a2 = C3R(variant="a2")
        self.assertEqual(sum(parameter.numel() for parameter in c1.parameters()), 46017)
        self.assertEqual(sum(parameter.numel() for parameter in a2.parameters()), 58305)
        messages = c1.encoder(
            self.tokens, self.response,
            torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]), None,
            [1], [1234], [10], [330])
        self.assertEqual(tuple(messages[0].prompt.shape), (4, 64))
        self.assertEqual(messages[0].quantized_prompt.dtype, torch.int8)
        self.assertEqual(messages[0].scales.dtype, torch.float32)
        self.assertEqual(len(c1.codec.serialize(messages[0])), 320)

    def test_no_remote_and_all_rejected_identity(self):
        module = C3R(variant="c1")
        empty = self.collaborate(module, [])
        self.assertIs(empty["search_tokens"], self.tokens)
        self.assertFalse(empty["used_remote"])
        stale = make_message(frame_id=5, timestamp_ms=165)
        rejected = self.collaborate(module, [stale])
        self.assertIs(rejected["search_tokens"], self.tokens)
        self.assertFalse(rejected["used_remote"])

    def test_three_view_c0_and_c1_forward(self):
        for variant in ("c0", "c1"):
            module = C3R(variant=variant)
            for receiver in range(3):
                messages = [
                    make_message(sender_id=sender, prompt_offset=sender * 0.1)
                    for sender in range(3) if sender != receiver
                ]
                result = module.collaborate(
                    self.tokens, self.response, messages,
                    receiver_id=receiver, sequence_hash=1234,
                    local_frame_id=10, local_timestamp_ms=330,
                    frame_interval_ms=33, last_frame_by_sender={})
                self.assertTrue(result["used_remote"])
                self.assertEqual(result["accepted_count"], 2)
                self.assertEqual(result["search_tokens"].shape, self.tokens.shape)
                self.assertTrue(torch.isfinite(result["search_tokens"]).all())
                if variant == "c1":
                    self.assertLessEqual(float(result["aggregate_ratio"]), 0.350001)

    def test_order_invariance_and_conflict(self):
        module = C3R(variant="c1")
        one = make_message(sender_id=1, prompt_offset=0.1)
        two = make_message(sender_id=2, prompt_offset=-0.1)
        forward = self.collaborate(module, [one, two])
        self.context["last_frame_by_sender"] = {}
        reverse = self.collaborate(module, [two, one])
        self.assertTrue(torch.equal(forward["search_tokens"], reverse["search_tokens"]))
        self.assertEqual(forward["accepted_sender_ids"], [1, 2])

    def test_perturbation_interfaces_remote_only(self):
        local_before = self.tokens.clone()
        messages = [make_message(1), make_message(2, prompt_offset=0.2)]
        perturb = CommunicationPerturbation(enabled=False, seed=3)
        self.assertEqual(perturb.apply(messages, 33, force_fault="dropout"), [])
        zero = perturb.apply(messages, 33, force_fault="zero")
        self.assertTrue(torch.count_nonzero(zero[0].prompt) == 0)
        delay = perturb.apply(messages, 33, force_fault="delay")
        self.assertEqual(delay[0].frame_id, 9)
        stale = perturb.apply(messages, 33, force_fault="stale")
        self.assertEqual(stale[0].fault, "stale")
        wrong = perturb.apply(messages, 33, force_fault="wrong_remote")
        self.assertEqual(wrong[0].construction_label, 0)
        one_bad = perturb.apply(messages, 33, force_fault="one_bad")
        self.assertEqual([item.construction_label for item in one_bad], [1, 0])
        corrupt = perturb.apply(messages, 33, force_fault="corrupt")
        self.assertEqual(corrupt[0].fault, "corrupt")
        self.assertTrue(torch.equal(local_before, self.tokens))

    def test_gate_range_and_no_gt_api(self):
        module = C3R(variant="c1")
        result = self.collaborate(module, [make_message(1), make_message(2)])
        self.assertTrue(bool((result["gates"] > 0).all().item()))
        self.assertTrue(bool((result["gates"] < 0.25).all().item()))
        forbidden = ("gt", "annotation", "visible", "occlusion", "oracle", "iou")
        signatures = " ".join((
            str(inspect.signature(C3R.collaborate)).lower(),
            str(inspect.signature(C3R._reliability_input)).lower(),
        ))
        for name in forbidden:
            self.assertNotIn(name, signatures)

    def test_one_step_loss_gradient_update_resume(self):
        module = C3R(variant="c1")
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        sender_tokens = torch.randn(2, 256, 192)
        sender_response = torch.sigmoid(torch.randn(2, 1, 16, 16))
        messages = module.encoder(
            sender_tokens, sender_response,
            torch.tensor([[[0.5, 0.5, 0.2, 0.2]], [[0.4, 0.6, 0.3, 0.2]]]),
            None, [1, 2], [1234, 1234], [10, 10], [330, 330])
        messages[1].construction_label = 0
        before = {name: value.detach().clone() for name, value in module.named_parameters()}
        result = self.collaborate(module, messages)
        track_loss = result["search_tokens"].square().mean()
        rank_loss = gate_ranking_loss(result["gate_logits"], result["gate_labels"])
        loss = track_loss + 0.10 * rank_loss + 0.05 * result["residual_budget_loss"]
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad()
        loss.backward()
        finite_gradients = [
            bool(torch.isfinite(parameter.grad).all().item())
            for parameter in module.parameters() if parameter.grad is not None
        ]
        self.assertTrue(finite_gradients and all(finite_gradients))
        optimizer.step()
        self.assertTrue(any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in module.named_parameters()
        ))

        buffer = io.BytesIO()
        torch.save({"model": module.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        resumed = C3R(variant="c1")
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        checkpoint = torch.load(buffer, map_location="cpu")
        resumed.load_state_dict(checkpoint["model"], strict=True)
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        for name, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[name]))


class C3RModelIntegrationTest(unittest.TestCase):
    def test_dummy_e0_forward_and_no_remote_model_identity(self):
        torch.manual_seed(13)
        backbone = DummyBackbone()
        head = DummyCenterHead()
        e0 = EnTeRTrack(copy.deepcopy(backbone), copy.deepcopy(head), head_type="CENTER")
        c1 = EnTeRTrack(
            copy.deepcopy(backbone), copy.deepcopy(head), head_type="CENTER",
            c3r=C3R(variant="c1"), c3r_freeze_local=True)
        template = torch.randn(1, 3, 8, 8)
        search = torch.randn(1, 3, 16, 16)
        e0.eval()
        c1.eval()
        local = e0(template, search, training=False)
        no_remote = c1(template, search, training=False, c3r_packets=[])
        for key in ("pred_boxes", "score_map", "backbone_feat"):
            self.assertTrue(torch.equal(local[key], no_remote[key]))
        self.assertNotIn("c3r", no_remote)
        collaborative = c1(
            template, search, training=False,
            c3r_packets=[make_message(sender_id=1)],
            c3r_context={
                "receiver_id": 0,
                "sequence_hash": 1234,
                "frame_id": 10,
                "timestamp_ms": 330,
                "frame_interval_ms": 33,
                "last_frame_by_sender": {},
            },
        )
        self.assertIn("c3r", collaborative)
        self.assertEqual(collaborative["c3r"]["accepted_count"], 1)
        self.assertEqual(collaborative["pred_boxes"].shape, local["pred_boxes"].shape)

    def test_actor_one_step_train_finite_and_local_frozen(self):
        cfg = get_default_config()
        update_config_from_file(
            os.path.join(ROOT, "experiments/entertrack/entertrack_c3r_c1.yaml"),
            base_cfg=cfg)
        cfg.TRAIN.C3R.PERTURBATIONS_ENABLED = False
        model = EnTeRTrack(
            DummyBackbone(), DummyCenterHead(), head_type="CENTER",
            c3r=C3R(variant="c1"), c3r_freeze_local=True)
        optimizer, _ = get_optimizer_scheduler(model, cfg)
        settings = type("Settings", (), {"batchsize": 1})()
        actor = EnTeRTrackActorThreeMDOT(
            net=model,
            objective={
                "giou": giou_loss,
                "l1": torch.nn.functional.l1_loss,
                "focal": FocalLoss(),
            },
            loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
            settings=settings,
            cfg=cfg,
        )
        model.train()
        local_before = {
            name: value.detach().clone() for name, value in model.state_dict().items()
            if not name.startswith("c3r.")
        }
        data = {
            "template_images": torch.randn(3, 1, 3, 16, 16),
            "search_images": torch.randn(3, 1, 3, 32, 32),
            "template_anno": torch.tensor([[[0.3, 0.3, 0.2, 0.2]]] * 3),
            "search_anno": torch.tensor([[[0.35, 0.35, 0.2, 0.2]]] * 3),
            "epoch": 0,
        }
        optimizer.zero_grad()
        loss, status = actor(data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad for name, parameter in model.named_parameters()
            if name.startswith("c3r.") and parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(all(
            gradient is None or bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
        ))
        target_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("c3r.") and parameter.requires_grad
        }
        optimizer.step()
        self.assertTrue(any(
            not torch.equal(target_before[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if name in target_before
        ))
        for name, value in local_before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        self.assertTrue(torch.isfinite(torch.tensor(status["Loss/total"])))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_gpu_actor_one_step_memory_smoke(self):
        cfg = load_config("entertrack_c3r_smoke")
        model = build_entertrack(cfg, training=True).cuda(0)
        optimizer, _ = get_optimizer_scheduler(model, cfg)
        settings = type("Settings", (), {"batchsize": 1})()
        actor = EnTeRTrackActorThreeMDOT(
            net=model,
            objective={
                "giou": giou_loss,
                "l1": torch.nn.functional.l1_loss,
                "focal": FocalLoss(),
            },
            loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
            settings=settings,
            cfg=cfg,
        )
        local_before = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if not name.startswith("c3r.")
        }
        device = torch.device("cuda:0")
        data = {
            "template_images": torch.randn(3, 1, 3, 128, 128, device=device),
            "search_images": torch.randn(3, 1, 3, 256, 256, device=device),
            "template_anno": torch.tensor(
                [[[0.3, 0.3, 0.2, 0.2]]] * 3, device=device),
            "search_anno": torch.tensor(
                [[[0.35, 0.35, 0.2, 0.2]]] * 3, device=device),
            "epoch": 0,
        }
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()
        loss, status = actor(data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        self.assertGreater(peak_bytes, 0)
        for name, value in local_before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name].detach().cpu()), name)
        self.assertTrue(torch.isfinite(torch.tensor(status["Loss/total"])))
        print("C3R_REAL_GPU_SMOKE_PEAK_BYTES={}".format(peak_bytes))

    @unittest.skipUnless(os.path.isfile(CHECKPOINT), "frozen checkpoint unavailable")
    def test_checkpoint_strict_load_optimizer_freeze_and_update_scope(self):
        e0_cfg = get_default_config()
        update_config_from_file(
            os.path.join(ROOT, "experiments/entertrack/entertrack_c3r_e0.yaml"),
            base_cfg=e0_cfg)
        e0 = build_entertrack(e0_cfg, training=False)
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
        state = checkpoint.get("net", checkpoint.get("model", checkpoint))
        e0.load_state_dict(state, strict=True)

        c1_cfg = get_default_config()
        update_config_from_file(
            os.path.join(ROOT, "experiments/entertrack/entertrack_c3r_c1.yaml"),
            base_cfg=c1_cfg)
        c1 = build_entertrack(c1_cfg, training=True)
        self.assertTrue(c1.initialization_audit["strict_full_load"])
        self.assertEqual(c1.initialization_audit["loaded_local_key_count"], 182)
        local_before = {
            name: value.detach().clone() for name, value in c1.state_dict().items()
            if not name.startswith("c3r.")
        }
        optimizer, _ = get_optimizer_scheduler(c1, c1_cfg)
        optimizer_names = assert_c3r_optimizer_membership(c1, optimizer)
        self.assertTrue(optimizer_names)
        self.assertTrue(all(name.startswith("c3r.") for name in optimizer_names))
        self.assertEqual(sum(
            parameter.numel() for parameter in c1.parameters() if parameter.requires_grad
        ), 46017)

        result = c1.c3r.collaborate(
            torch.randn(1, 256, 192),
            torch.sigmoid(torch.randn(1, 1, 16, 16)),
            [make_message(1), make_message(2, prompt_offset=0.1)],
            0, 1234, 10, 330, 33, {})
        loss = result["search_tokens"].square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for name, value in local_before.items():
            self.assertTrue(torch.equal(value, c1.state_dict()[name]), name)

    @unittest.skipUnless(os.path.isfile(CHECKPOINT), "frozen checkpoint unavailable")
    def test_real_checkpoint_no_remote_identity(self):
        e0_cfg = get_default_config()
        update_config_from_file(
            os.path.join(ROOT, "experiments/entertrack/entertrack_c3r_e0.yaml"),
            base_cfg=e0_cfg)
        c1_cfg = get_default_config()
        update_config_from_file(
            os.path.join(ROOT, "experiments/entertrack/entertrack_c3r_c1.yaml"),
            base_cfg=c1_cfg)
        e0 = build_entertrack(e0_cfg, training=False)
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
        state = checkpoint.get("net", checkpoint.get("model", checkpoint))
        e0.load_state_dict(state, strict=True)
        c1 = build_entertrack(c1_cfg, training=True)
        e0.eval()
        c1.eval()
        torch.manual_seed(17)
        template = torch.randn(1, 3, 128, 128)
        search = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            local = e0(template, search, training=False)
            fallback = c1(template, search, training=False, c3r_packets=[])
        for key in ("pred_boxes", "score_map", "backbone_feat"):
            self.assertTrue(torch.equal(local[key], fallback[key]), key)
            self.assertEqual(float((local[key] - fallback[key]).abs().max()), 0.0)

    def test_ddp_world_size_one_smoke(self):
        if not dist.is_available():
            self.skipTest("torch.distributed unavailable")
        if dist.is_initialized():
            self.skipTest("process group already initialized")
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            init_path = handle.name
        try:
            os.environ["GLOO_SOCKET_IFNAME"] = "lo"
            dist.init_process_group(
                backend="gloo", init_method="file://" + init_path,
                rank=0, world_size=1)
            encoder = C3R(variant="c1").encoder
            ddp = DistributedDataParallel(encoder)
            messages = ddp(
                torch.randn(1, 256, 192),
                torch.sigmoid(torch.randn(1, 1, 16, 16)),
                torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
                None, [1], [1234], [10], [330])
            output = messages[0].prompt
            self.assertEqual(tuple(output.shape), (4, 64))
            gathered = [torch.zeros_like(output)]
            dist.all_gather(gathered, output.detach())
            self.assertTrue(torch.equal(gathered[0], output.detach()))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            if os.path.exists(init_path):
                os.unlink(init_path)


if __name__ == "__main__":
    unittest.main()
