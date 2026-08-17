#!/usr/bin/env python3
"""Audit whether PCUM-v2A can be enabled on the frozen controlled B0 model.

This tool is intentionally read-only.  It distinguishes the parameter-free
confidence aggregator from the trainable PCUM encoder/aligner/fusion module and
never falls back to a non-strict checkpoint load.
"""

import argparse
import copy
import hashlib
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.models.entertrack.entertrack import build_entertrack  # noqa: E402
from lib.models.entertrack.pcum import build_pcum  # noqa: E402
from lib.train.actors.entertrack_threemdot import EnTeRTrackActorThreeMDOT  # noqa: E402
from lib.train.optimizer_groups import build_optimizer_param_groups  # noqa: E402
from lib.train.pcum_freeze import apply_pcum_ranking_freeze  # noqa: E402
from lib.utils.focal_loss import FocalLoss  # noqa: E402
from lib.test.utils.pcum_remote_state import build_remote_state  # noqa: E402
from lib.utils.box_ops import giou_loss  # noqa: E402


DEFAULT_B0 = "ostrack_deit_tiny_b0_ep25"
DEFAULT_A0 = "pcum_v2_a0_weighted_softmax_t010_ep15_t2_raw_test"
DEFAULT_TRAINING_CANDIDATE = "ostrack_deit_tiny_b0_pcum_frozen_ep15"
FROZEN_B0_SHA256 = (
    "88706aa3087d245c22c152d3feb5417e20bd12f06942283cc0c513c53d2c6128"
)
EXPECTED_PCUM_TENSORS = 34
EXPECTED_PCUM_PARAMETERS = 682753
DEFAULT_INIT_SEED = 42

# These are the only candidate-vs-B0 namespaces that a future trained
# B0+PCUM parameter file may change.  Core model, training recipe and normal
# tracker fields remain outside this allowlist.
ALLOWED_CANDIDATE_DIFF_PREFIXES = (
    "MODEL_ROLE",
    "B0_CHECKPOINT",
    "TRAIN_PCUM_ONLY",
    "FIXED_FINAL_EPOCH",
    "MODEL.PCUM.",
    "MODEL.PRETRAIN_FILE",
    "TEST.EPOCH",
    "TEST.PCUM.",
    "TEST.SAVE_DIR",
    "TEST.CHECKPOINT_NAME",
    "TRAIN.",
)


def load_config(name):
    resolved = copy.deepcopy(cfg)
    path = os.path.join(ROOT, "experiments", "entertrack", name + ".yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    update_config_from_file(path, base_cfg=resolved)
    return resolved


def checkpoint_path(resolved):
    save_dir = str(resolved.TEST.SAVE_DIR)
    name = str(resolved.TEST.CHECKPOINT_NAME)
    epoch = int(resolved.TEST.EPOCH)
    return os.path.join(
        ROOT,
        save_dir,
        "checkpoints",
        "train",
        "entertrack",
        name,
        "EnTeRTrack_ep{:04d}.pth.tar".format(epoch),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    if state and all(name.startswith("module.") for name in state):
        state = {name[len("module."):]: value for name, value in state.items()}
    return checkpoint, state


def state_dict_sha256(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def pcum_named_parameters(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("pcum.")
    ]


def non_pcum_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("pcum.")
    }


def assert_tensor_dict_equal(left, right):
    if set(left) != set(right):
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        output = {}
        for key in sorted(value):
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            output.update(flatten(value[key], child))
        return output
    return {prefix: value}


def config_diff(base, candidate):
    left = flatten(base)
    right = flatten(candidate)
    rows = []
    for key in sorted(set(left).union(right)):
        before = left.get(key, "<missing>")
        after = right.get(key, "<missing>")
        if before != after:
            allowed = any(
                key == prefix or key.startswith(prefix)
                for prefix in ALLOWED_CANDIDATE_DIFF_PREFIXES
            )
            rows.append((key, before, after, allowed))
    return rows


def parameter_audit(b0_cfg, a0_cfg):
    b0_model = build_entertrack(b0_cfg, training=False)
    pcum = build_pcum(a0_cfg, token_dim=192)
    grouped = defaultdict(int)
    for name, parameter in pcum.named_parameters():
        grouped[name.split(".", 1)[0]] += parameter.numel()

    path = checkpoint_path(b0_cfg)
    checkpoint, state = checkpoint_state(path)
    # The frozen B0 model itself must load completely and strictly.
    b0_model.load_state_dict(state, strict=True)

    missing_pcum = sorted("pcum." + name for name, _ in pcum.named_parameters())
    present_pcum = sorted(name for name in state if name.startswith("pcum."))
    a0_path = checkpoint_path(a0_cfg)
    a0_checkpoint, a0_state = checkpoint_state(a0_path)
    a0_pcum_keys = sorted(name for name in a0_state if name.startswith("pcum."))
    return {
        "b0_model": b0_model,
        "pcum": pcum,
        "checkpoint": checkpoint,
        "checkpoint_state": state,
        "checkpoint_path": path,
        "checkpoint_sha256": sha256_file(path),
        "b0_parameter_count": sum(p.numel() for p in b0_model.parameters()),
        "pcum_parameter_count": sum(p.numel() for p in pcum.parameters()),
        "pcum_trainable_count": sum(
            p.numel() for p in pcum.parameters() if p.requires_grad
        ),
        "pcum_parameter_tensors": len(list(pcum.named_parameters())),
        "pcum_buffer_count": sum(b.numel() for b in pcum.buffers()),
        "pcum_buffer_tensors": len(list(pcum.named_buffers())),
        "pcum_grouped_parameters": dict(grouped),
        "b0_checkpoint_pcum_keys": present_pcum,
        "hypothetical_missing_pcum_keys": missing_pcum,
        "a0_checkpoint_path": a0_path,
        "a0_checkpoint_epoch": a0_checkpoint.get("epoch", None),
        "a0_checkpoint_pcum_keys": a0_pcum_keys,
        "a0_checkpoint_pcum_values": sum(
            a0_state[name].numel() for name in a0_pcum_keys
        ),
    }


def make_training_model(candidate_cfg, seed=DEFAULT_INIT_SEED):
    torch.manual_seed(int(seed))
    model = build_entertrack(candidate_cfg, training=True)
    return model


def training_ready_audit(candidate_cfg, expected_tensors=EXPECTED_PCUM_TENSORS,
                         expected_parameters=EXPECTED_PCUM_PARAMETERS,
                         seed=DEFAULT_INIT_SEED):
    model = make_training_model(candidate_cfg, seed=seed)
    report = getattr(model, "initialization_audit", {})
    freeze = apply_pcum_ranking_freeze(model, candidate_cfg, verbose=False)
    groups = build_optimizer_param_groups(model, candidate_cfg, verbose=False)

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    pcum_trainable = [
        (name, parameter)
        for name, parameter in trainable
        if name.startswith("pcum.")
    ]
    group_params = [
        parameter
        for group in groups
        for parameter in group["params"]
    ]
    group_param_ids = [id(parameter) for parameter in group_params]
    pcum_items = [
        (name, parameter.detach())
        for name, parameter in pcum_named_parameters(model)
    ]
    pcum_count = sum(parameter.numel() for _, parameter in pcum_trainable)
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(str(candidate_cfg.MODEL_ROLE) == "posthoc_b0_pcum_frozen",
            "MODEL_ROLE must be posthoc_b0_pcum_frozen")
    require(bool(candidate_cfg.TRAIN_PCUM_ONLY), "TRAIN_PCUM_ONLY must be true")
    require(int(candidate_cfg.FIXED_FINAL_EPOCH) == int(candidate_cfg.TRAIN.EPOCH),
            "FIXED_FINAL_EPOCH must match TRAIN.EPOCH")
    require(str(candidate_cfg.B0_CHECKPOINT) == str(candidate_cfg.MODEL.PRETRAIN_FILE),
            "B0_CHECKPOINT must match MODEL.PRETRAIN_FILE")
    require(sha256_file(str(candidate_cfg.B0_CHECKPOINT)) == FROZEN_B0_SHA256,
            "B0 checkpoint SHA256 mismatch")
    require(bool(candidate_cfg.MODEL.PCUM.ENABLED), "PCUM must be enabled")
    require(str(candidate_cfg.MODEL.PCUM.REMOTE_AGGREGATION) == "confidence_softmax",
            "REMOTE_AGGREGATION must be confidence_softmax")
    require(abs(float(candidate_cfg.MODEL.PCUM.REMOTE_WEIGHT_TEMPERATURE) - 0.10) < 1e-12,
            "REMOTE_WEIGHT_TEMPERATURE must be 0.10")
    require(str(candidate_cfg.TEST.PCUM.REMOTE_STATE_SOURCE) == "tracker",
            "REMOTE_STATE_SOURCE must be tracker")
    require(not bool(candidate_cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK),
            "Inference visible mask must be false")
    require(not bool(candidate_cfg.MODEL.USE_SEARCH_PROMPT),
            "ARP/ATP search prompt must be disabled")
    require(str(candidate_cfg.MODEL.BACKBONE.TYPE) == "vit_tiny_patch16_224_half",
            "Backbone must be B0 half/tiny, not ARP")
    require(not bool(candidate_cfg.MODEL.BACKBONE.PRUNING_ENABLED),
            "Pruning must be disabled")
    require(not bool(candidate_cfg.MODEL.BACKBONE.TOKEN_COMPENSATION_ENABLED),
            "Token compensation must be disabled")
    require(not bool(candidate_cfg.TEST.MCR.ENABLED), "MCR must be disabled")
    require(bool(candidate_cfg.TRAIN.PCUM_RANKING.FREEZE_BACKBONE),
            "FREEZE_BACKBONE must be enabled")
    require(bool(candidate_cfg.TRAIN.PCUM_RANKING.FREEZE_HEAD),
            "FREEZE_HEAD must be enabled")
    require(len(pcum_trainable) == expected_tensors,
            "trainable PCUM tensor count mismatch")
    require(pcum_count == expected_parameters,
            "trainable PCUM parameter count mismatch")
    require(all(name.startswith("pcum.") for name, _ in trainable),
            "non-PCUM parameter is trainable")
    require(len(group_param_ids) == len(set(group_param_ids)),
            "optimizer parameter list contains duplicates")
    require(set(group_param_ids) == {id(parameter) for _, parameter in pcum_trainable},
            "optimizer parameter set does not exactly match PCUM trainables")
    require({group["group_name"] for group in groups} == {"pcum"},
            "optimizer must contain only the pcum group")
    require(int(report.get("inherited_a0_pcum_parameters", -1)) == 0,
            "inherited_a0_pcum_parameters must be 0")
    require(bool(report.get("strict_full_load", False)),
            "posthoc B0+PCUM full load must be strict")

    return {
        "model": model,
        "failures": failures,
        "initialization_audit": report,
        "freeze_summary": freeze,
        "optimizer_groups": groups,
        "trainable_names": [name for name, _ in trainable],
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "pcum_init_sha256": state_dict_sha256(pcum_items),
        "pcum_shapes": [
            (name, tuple(parameter.shape), parameter.numel())
            for name, parameter in pcum_trainable
        ],
    }


def synthetic_training_batch(batch_size=1):
    torch.manual_seed(20260715)
    search_anno = torch.tensor(
        [
            [[0.50, 0.50, 0.20, 0.24]],
            [[0.48, 0.52, 0.18, 0.22]],
            [[0.53, 0.49, 0.19, 0.23]],
        ],
        dtype=torch.float32,
    )
    template_anno = search_anno.clone()
    if batch_size != 1:
        search_anno = search_anno.repeat(1, batch_size, 1)
        template_anno = template_anno.repeat(1, batch_size, 1)
    return {
        "template_images": torch.randn(3, batch_size, 3, 128, 128),
        "search_images": torch.randn(3, batch_size, 3, 256, 256),
        "template_anno": template_anno,
        "search_anno": search_anno,
        "template_view_valid": torch.ones(3, batch_size, dtype=torch.bool),
        "search_view_valid": torch.ones(3, batch_size, dtype=torch.bool),
        "epoch": 1,
    }


def one_step_training_smoke(candidate_cfg, seed=DEFAULT_INIT_SEED):
    audit = training_ready_audit(candidate_cfg, seed=seed)
    if audit["failures"]:
        return {
            "ready": False,
            "failures": list(audit["failures"]),
        }
    model = audit["model"]
    model.train(True)
    apply_pcum_ranking_freeze(model, candidate_cfg, verbose=False)
    before_core = non_pcum_state(model)
    before_pcum = {
        name: parameter.detach().cpu().clone()
        for name, parameter in pcum_named_parameters(model)
    }
    optimizer = torch.optim.AdamW(
        audit["optimizer_groups"],
        lr=float(candidate_cfg.TRAIN.LR),
        weight_decay=float(candidate_cfg.TRAIN.WEIGHT_DECAY),
    )
    actor = EnTeRTrackActorThreeMDOT(
        net=model,
        objective={
            "giou": giou_loss,
            "l1": F.l1_loss,
            "focal": FocalLoss(),
        },
        loss_weight={
            "giou": float(candidate_cfg.TRAIN.GIOU_WEIGHT),
            "l1": float(candidate_cfg.TRAIN.L1_WEIGHT),
            "focal": float(candidate_cfg.TRAIN.FOCAL_WEIGHT),
        },
        settings=edict({"batchsize": 1}),
        cfg=candidate_cfg,
    )
    actor.train(True)
    data = synthetic_training_batch(batch_size=1)
    optimizer.zero_grad()
    local_loss, cache = actor.paired_local_stage(data)
    collaborative_loss, stats = actor.paired_collaborative_stage(data, cache)
    total_loss = local_loss + collaborative_loss
    total_loss.backward()
    grad_finite = True
    grad_nonzero = False
    for name, parameter in model.named_parameters():
        if not name.startswith("pcum."):
            continue
        if parameter.grad is not None:
            grad_finite = grad_finite and bool(torch.isfinite(parameter.grad).all().item())
            grad_nonzero = grad_nonzero or bool(parameter.grad.detach().abs().sum().item() > 0.0)
    optimizer.step()
    after_core = non_pcum_state(model)
    pcum_changed = any(
        not torch.equal(before_pcum[name], parameter.detach().cpu())
        for name, parameter in pcum_named_parameters(model)
    )
    stats_finite = all(
        torch.isfinite(torch.tensor(float(value))).item()
        for value in stats.values()
        if isinstance(value, (float, int))
    )
    return {
        "ready": True,
        "loss_finite": bool(torch.isfinite(total_loss.detach()).item()),
        "stats_finite": bool(stats_finite),
        "grad_finite": bool(grad_finite),
        "grad_nonzero": bool(grad_nonzero),
        "pcum_changed": bool(pcum_changed),
        "core_byte_identical": assert_tensor_dict_equal(before_core, after_core),
        "local_loss": float(local_loss.detach().item()),
        "collaborative_loss": float(collaborative_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "remote_source": str(candidate_cfg.TEST.PCUM.REMOTE_STATE_SOURCE),
        "inference_visible_mask": bool(candidate_cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK),
        "gt_inference_fields": [],
    }


def synthetic_multiview_smoke(audit, a0_cfg):
    """Compatibility-only smoke with untrained PCUM; never a result run."""
    model = audit["b0_model"].eval()
    pcum = audit["pcum"].eval()
    torch.manual_seed(20260715)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    templates = [torch.randn(1, 3, 128, 128) for _ in range(3)]
    searches = [torch.randn(1, 3, 256, 256) for _ in range(3)]
    local_outputs = []
    local_prompts = []
    with torch.inference_mode():
        for template, search in zip(templates, searches):
            output = model(template, search)
            feature = output["backbone_feat"]
            if isinstance(feature, (list, tuple)):
                feature = feature[-1]
            local_outputs.append(output)
            prompt_output = pcum({
                "search": feature[:, -model.feat_len_s:],
                "template": feature[:, :-model.feat_len_s],
            })
            local_prompts.append(prompt_output["local_prompt"])

        collaborative_outputs = []
        weight_sums = []
        for view in range(3):
            remotes = [local_prompts[index] for index in range(3) if index != view]
            remote_state = build_remote_state(
                scores=[0.8, 0.6],
                motion_reliabilities=[0.9, 0.7],
                source="tracker",
                device=torch.device("cpu"),
                gt_visibility=None,
                apces=[0.75, 0.65],
                bbox_scores=[0.85, 0.55],
                valid=[True, True],
                uav_indices=[index for index in range(3) if index != view],
            )
            feature = local_outputs[view]["backbone_feat"]
            if isinstance(feature, (list, tuple)):
                feature = feature[-1]
            result = pcum(
                {
                    "search": feature[:, -model.feat_len_s:],
                    "template": feature[:, :-model.feat_len_s],
                },
                remote_prompts=remotes,
                remote_states=remote_state,
            )
            fused = torch.cat([
                feature[:, :-model.feat_len_s],
                result["search_tokens"],
            ], dim=1)
            head = model.forward_head(fused)
            collaborative_outputs.append(head)
            weight_sums.append(result["remote_weights"].sum(dim=1))

        disabled_pcum = copy.deepcopy(pcum)
        disabled_pcum.enabled = False
        probe = torch.randn(1, 256, 192)
        identity = disabled_pcum({"search": probe})["search_tokens"]

    core_unchanged = all(
        torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    forbidden = {
        "gt_bbox", "gt_visibility", "target_visible", "visible",
        "occlusion", "oov", "oracle_mask",
    }
    return {
        "local_bbox_shapes": [tuple(out["pred_boxes"].shape) for out in local_outputs],
        "local_score_shapes": [tuple(out["score_map"].shape) for out in local_outputs],
        "collaborative_bbox_shapes": [
            tuple(out["pred_boxes"].shape) for out in collaborative_outputs
        ],
        "collaborative_score_shapes": [
            tuple(out["score_map"].shape) for out in collaborative_outputs
        ],
        "finite": all(
            torch.isfinite(out["pred_boxes"]).all().item()
            and torch.isfinite(out["score_map"]).all().item()
            for out in local_outputs + collaborative_outputs
        ),
        "weight_sums": [float(value.item()) for value in weight_sums],
        "temperature": float(a0_cfg.MODEL.PCUM.REMOTE_WEIGHT_TEMPERATURE),
        "remote_source": str(a0_cfg.TEST.PCUM.REMOTE_STATE_SOURCE),
        "remote_visible_mask": bool(a0_cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK),
        "tracker_state_has_forbidden_gt_fields": bool(
            forbidden.intersection(remote_state)
        ),
        "disabled_identity": torch.equal(identity, probe),
        "b0_core_weights_unchanged": core_unchanged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_B0)
    parser.add_argument("--pcum-reference", default=DEFAULT_A0)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--expect-path", choices=("A", "B", "C"), default="B")
    parser.add_argument("--expect-training-ready", action="store_true")
    parser.add_argument("--expected-trainable-tensors", type=int,
                        default=EXPECTED_PCUM_TENSORS)
    parser.add_argument("--expected-trainable-parameters", type=int,
                        default=EXPECTED_PCUM_PARAMETERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_INIT_SEED)
    parser.add_argument("--one-step-smoke", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()

    b0_cfg = load_config(args.base)
    a0_cfg = load_config(args.pcum_reference)
    audit = parameter_audit(b0_cfg, a0_cfg)
    path = "B" if audit["pcum_trainable_count"] > 0 else "A"

    print("base_config={}".format(args.base))
    print("pcum_reference={}".format(args.pcum_reference))
    print("selected_path={}".format(path))
    print("checkpoint={}".format(audit["checkpoint_path"]))
    print("checkpoint_sha256={}".format(audit["checkpoint_sha256"]))
    print("checkpoint_sha256_matches_frozen={}".format(
        audit["checkpoint_sha256"] == FROZEN_B0_SHA256))
    print("b0_strict_load=true")
    print("b0_parameter_count={}".format(audit["b0_parameter_count"]))
    print("pcum_parameter_count={}".format(audit["pcum_parameter_count"]))
    print("pcum_trainable_count={}".format(audit["pcum_trainable_count"]))
    print("pcum_parameter_tensors={}".format(audit["pcum_parameter_tensors"]))
    print("pcum_persistent_buffers={}".format(audit["pcum_buffer_count"]))
    print("b0_checkpoint_pcum_keys={}".format(
        len(audit["b0_checkpoint_pcum_keys"])))
    print("strict_b0_plus_pcum_missing_keys={}".format(
        len(audit["hypothetical_missing_pcum_keys"])))
    print("a0_checkpoint={}".format(audit["a0_checkpoint_path"]))
    print("a0_checkpoint_epoch={}".format(audit["a0_checkpoint_epoch"]))
    print("a0_checkpoint_pcum_keys={}".format(
        len(audit["a0_checkpoint_pcum_keys"])))
    print("a0_checkpoint_pcum_values={}".format(
        audit["a0_checkpoint_pcum_values"]))
    print("confidence_aggregation={}".format(
        a0_cfg.MODEL.PCUM.REMOTE_AGGREGATION))
    print("temperature={}".format(
        a0_cfg.MODEL.PCUM.REMOTE_WEIGHT_TEMPERATURE))

    failed = path != args.expect_path
    candidate_cfg = None
    candidate_name = args.candidate
    if args.expect_training_ready and candidate_name is None:
        candidate_name = DEFAULT_TRAINING_CANDIDATE
    if candidate_name:
        candidate_cfg = load_config(candidate_name)
        print("candidate_config={}".format(candidate_name))
        rows = config_diff(b0_cfg, candidate_cfg)
        for key, before, after, allowed in rows:
            print("diff={} | {!r} -> {!r} | allowed={}".format(
                key, before, after, allowed))
        disallowed = [row for row in rows if not row[3]]
        print("config_diff_status={}".format("PASS" if not disallowed else "FAIL"))
        failed = failed or bool(disallowed)
    else:
        print("candidate_config=NOT_CREATED_PATH_B")
        print("config_diff_status=BLOCKED_UNTIL_PCUM_TRAINING_PROTOCOL_IS_AUTHORIZED")

    if args.expect_training_ready:
        if candidate_cfg is None:
            raise SystemExit("training-ready audit requires a candidate config")
        ready = training_ready_audit(
            candidate_cfg,
            expected_tensors=args.expected_trainable_tensors,
            expected_parameters=args.expected_trainable_parameters,
            seed=args.seed,
        )
        print("training_ready_status={}".format(
            "PASS" if not ready["failures"] else "FAIL"))
        for failure in ready["failures"]:
            print("training_ready_failure={}".format(failure))
        print("trainable_tensor_count={}".format(
            ready["trainable_tensor_count"]))
        print("trainable_parameter_count={}".format(
            ready["trainable_parameter_count"]))
        print("pcum_init_sha256={}".format(ready["pcum_init_sha256"]))
        print("inherited_a0_pcum_parameters={}".format(
            ready["initialization_audit"].get(
                "inherited_a0_pcum_parameters", "UNKNOWN")))
        print("optimizer_groups={}".format([
            (group["group_name"], len(group["params"]), group["lr"])
            for group in ready["optimizer_groups"]
        ]))
        print("trainable_names={}".format(ready["trainable_names"]))
        failed = failed or bool(ready["failures"])

    if args.one_step_smoke:
        if candidate_cfg is None:
            raise SystemExit("one-step smoke requires a candidate config")
        smoke = one_step_training_smoke(candidate_cfg, seed=args.seed)
        for key in sorted(smoke):
            print("train_smoke_{}={}".format(key, smoke[key]))
        failed = failed or not (
            smoke.get("ready", False)
            and smoke.get("loss_finite", False)
            and smoke.get("stats_finite", False)
            and smoke.get("grad_finite", False)
            and smoke.get("grad_nonzero", False)
            and smoke.get("pcum_changed", False)
            and smoke.get("core_byte_identical", False)
            and smoke.get("remote_source") == "tracker"
            and not smoke.get("inference_visible_mask", True)
            and not smoke.get("gt_inference_fields")
        )

    if args.synthetic_smoke:
        smoke = synthetic_multiview_smoke(audit, a0_cfg)
        for key in sorted(smoke):
            print("smoke_{}={}".format(key, smoke[key]))
        failed = failed or not (
            smoke["finite"]
            and all(abs(value - 1.0) < 1e-6 for value in smoke["weight_sums"])
            and smoke["temperature"] == 0.1
            and smoke["remote_source"] == "tracker"
            and not smoke["remote_visible_mask"]
            and not smoke["tracker_state_has_forbidden_gt_fields"]
            and smoke["disabled_identity"]
            and smoke["b0_core_weights_unchanged"]
        )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
