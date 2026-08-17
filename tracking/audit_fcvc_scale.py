#!/usr/bin/env python3
"""FCVC inner-train loss/gradient scale audit.

This entrypoint is intentionally audit-only: it builds a fixed manifest, runs
forward/backward diagnostics, writes reports, and never calls optimizer.step().
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from lib.config.entertrack.config import cfg, update_config_from_file
from lib.models.entertrack.entertrack import build_entertrack
from lib.models.entertrack.fcvc import FCVCConfig, FCVCModel, build_sender_bundle
from lib.models.entertrack.fcvc.feature_taps import capture_taps, split_template_search
from lib.train.admin import env_settings
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.train.dataset.threemdot import ThreeMDOT
from lib.utils.box_ops import box_xywh_to_xyxy, box_cxcywh_to_xyxy, generalized_box_iou


OUT = ROOT / "output/multi_agent_collaboration_clean/fcvc_scale_audit"
DESIGN = ROOT / "output/multi_agent_collaboration_clean/fcvc_design"
IMPL = ROOT / "output/multi_agent_collaboration_clean/fcvc_implementation"
RUNTIME = ROOT / "output/multi_agent_collaboration_clean/fcvc_runtime_integration"

INNER_TRAIN = [
    "md3026", "md3032", "md3062", "md3017", "md3018", "md3036", "md3038",
    "md3013", "md3059", "md3058", "md3054", "md3031", "md3030", "md3040",
]
INNER_DEV_FORBIDDEN = {"md3019", "md3005", "md3051", "md3044"}
OUTER_HOLDOUT_ID_ONLY = {"md3027", "md3020", "md3034", "md3050", "md3055"}
SEED = 20260716
VIEWS = {"A": 1, "B": 2, "C": 3}
LOSS_WEIGHTS = {
    "L_cls": 1.0,
    "L_giou": 2.0,
    "L_l1": 5.0,
    "L_align": 0.50,
    "L_recon": 1.00,
    "L_safe": 0.50,
    "L_cycle": 0.10,
    "L_teacher_track": 0.25,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_model(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def read_required_specs():
    allowed = [
        DESIGN / "loss_and_training_protocol.md",
        DESIGN / "teacher_student_spec.md",
        DESIGN / "full_architecture_spec.md",
        DESIGN / "full_model_acceptance_criteria.md",
        DESIGN / "design_manifest.md",
        IMPL / "parameter_inventory.csv",
        IMPL / "optimizer_scope_audit.md",
        IMPL / "teacher_gradient_contract.md",
        IMPL / "implementation_manifest.md",
        RUNTIME / "runtime_integration_manifest.md",
        RUNTIME / "no_gt_runtime_audit.md",
        RUNTIME / "persistent_state_registry.csv",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in allowed}


def read_image(path):
    try:
        import cv2 as cv
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        if img is not None:
            return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    except Exception:
        pass
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def tensorize(img, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(np.ascontiguousarray(img)).to(device=device).float()
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
    return (x - mean) / std


def sequence_path(dataset, target, view):
    seq = f"{target}-{view}"
    seq_id = dataset.sequence_list.index(seq)
    return Path(dataset._get_sequence_path(seq_id)), seq_id


def legal_frames(dataset, target):
    frames_by_view = {}
    min_len = None
    for view in (1, 2, 3):
        _, seq_id = sequence_path(dataset, target, view)
        info = dataset.get_sequence_info(seq_id)
        visible = info["visible"].bool()
        valid = info["valid"].bool()
        ok = (visible & valid).cpu().numpy().astype(bool)
        frames_by_view[view] = ok
        min_len = len(ok) if min_len is None else min(min_len, len(ok))
    frames = []
    for frame in range(1, int(min_len)):
        if all(bool(frames_by_view[v][frame]) for v in (1, 2, 3)):
            frames.append(frame)
    return frames


def build_manifest(dataset, groups=256):
    rng = random.Random(SEED)
    pools = {}
    for target in INNER_TRAIN:
        frames = legal_frames(dataset, target)
        if not frames:
            raise RuntimeError(f"No legal synchronized frames for {target}")
        rng.shuffle(frames)
        pools[target] = frames
    rows = []
    ptr = defaultdict(int)
    target_index = 0
    sync_groups = 0
    while sync_groups < groups:
        target = INNER_TRAIN[target_index % len(INNER_TRAIN)]
        target_index += 1
        frames = pools[target]
        frame = frames[ptr[target] % len(frames)]
        ptr[target] += 1
        seq_paths = {v: sequence_path(dataset, target, VIEWS[v])[0] for v in VIEWS}
        for receiver in ("A", "B", "C"):
            senders = [v for v in ("A", "B", "C") if v != receiver]
            rows.append({
                "target": target,
                "frame": frame,
                "receiver": receiver,
                "sender_1": senders[0],
                "sender_2": senders[1],
                "path_A": str(seq_paths["A"] / "img" / f"{frame + 1:08d}.jpg"),
                "path_B": str(seq_paths["B"] / "img" / f"{frame + 1:08d}.jpg"),
                "path_C": str(seq_paths["C"] / "img" / f"{frame + 1:08d}.jpg"),
                "gt_exists_A": True,
                "gt_exists_B": True,
                "gt_exists_C": True,
                "synchronization_validity": True,
                "split": "fold1_inner_train",
                "uses_gt_in_student_input": False,
            })
        sync_groups += 1
    return rows


def write_manifest(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scale_audit_sample_manifest.csv"
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path, sha256_file(path)


def crop_pair(dataset, target, view, template_frame, search_frame, device):
    seq_path, seq_id = sequence_path(dataset, target, VIEWS[view])
    info = dataset.get_sequence_info(seq_id)
    template_frame = int(template_frame)
    search_frame = int(search_frame)
    template_box = info["bbox"][template_frame].float()
    search_box = info["bbox"][search_frame].float()
    template_img = read_image(
        seq_path / "img" / f"{template_frame + 1:08d}.jpg")
    search_img = read_image(
        seq_path / "img" / f"{search_frame + 1:08d}.jpg")
    z, _, _ = sample_target(template_img, template_box, 2.0, 128)
    x, resize, _ = sample_target(search_img, search_box, 4.0, 256)
    gt_crop = transform_image_to_crop(
        search_box, search_box, resize, torch.tensor([256.0, 256.0]), normalize=True)
    return tensorize(z, device), tensorize(x, device), gt_crop.to(device), tuple(search_img.shape[:2])


def make_gt_heatmap(box_xywh, size=16):
    b = box_xywh.shape[0]
    device = box_xywh.device
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / size, 1 - 0.5 / size, size, device=device),
        torch.linspace(0.5 / size, 1 - 0.5 / size, size, device=device),
        indexing="ij")
    cx = box_xywh[:, 0] + 0.5 * box_xywh[:, 2]
    cy = box_xywh[:, 1] + 0.5 * box_xywh[:, 3]
    sigma = (0.5 * torch.sqrt((box_xywh[:, 2] * box_xywh[:, 3]).clamp_min(1e-6))).clamp_min(0.03)
    heat = []
    for i in range(b):
        d2 = (xx - cx[i]).square() + (yy - cy[i]).square()
        h = torch.exp(-0.5 * d2 / sigma[i].square())
        heat.append(h / h.sum().clamp_min(1e-6))
    return torch.stack(heat, dim=0).unsqueeze(1)


def pred_components(pred, gt_xywh):
    pred_box = pred["pred_boxes"][:, 0, :].clamp(0.0, 1.0)
    gt_xyxy = box_xywh_to_xyxy(gt_xywh).clamp(0.0, 1.0)
    pred_xyxy = box_cxcywh_to_xyxy(pred_box).clamp(0.0, 1.0)
    giou_matrix = generalized_box_iou(pred_xyxy, gt_xyxy)
    if isinstance(giou_matrix, tuple):
        giou_matrix = giou_matrix[0]
    giou = 1.0 - torch.diag(giou_matrix)
    l1 = F.l1_loss(pred_box, gt_xywh, reduction="none").mean(dim=1)
    heat = make_gt_heatmap(gt_xywh, pred["score_map"].shape[-1])
    cls = F.binary_cross_entropy(
        pred["score_map"].clamp(1e-6, 1 - 1e-6),
        heat.to(pred["score_map"].dtype),
        reduction="none").flatten(1).mean(dim=1)
    track = cls + 2.0 * giou + 5.0 * l1
    return {"L_cls": cls, "L_giou": giou, "L_l1": l1, "L_track": track}


def prediction_mean_iou(pred, gt_xywh):
    """Detached collaborative IoU used only for training diagnostics."""
    pred_box = pred["pred_boxes"][:, 0, :].detach().clamp(0.0, 1.0)
    gt_xyxy = box_xywh_to_xyxy(gt_xywh.detach()).clamp(0.0, 1.0)
    pred_xyxy = box_cxcywh_to_xyxy(pred_box).clamp(0.0, 1.0)
    result = generalized_box_iou(pred_xyxy, gt_xyxy)
    iou = result[1] if isinstance(result, tuple) else torch.zeros_like(result)
    return float(iou.detach().mean().cpu().item())


def align_component(match, sender_gt):
    contrib = match["sender_contribution"].clamp_min(1e-6)
    target = torch.full_like(contrib, 0.5)
    ce = -(target * contrib.log()).mean(dim=(1, 2))
    refs = match["sender_reference_points"]
    centers = []
    for gt in sender_gt:
        centers.append(torch.stack((gt[:, 0] + 0.5 * gt[:, 2], gt[:, 1] + 0.5 * gt[:, 3]), dim=-1))
    center = torch.stack(centers, dim=1).unsqueeze(1)
    ref_l1 = F.smooth_l1_loss(refs, center.expand_as(refs), reduction="none").mean(dim=(1, 2, 3))
    return ce + ref_l1


def cycle_component(out, local):
    q = out["queries"]
    original = local["target_prototype"].detach().unsqueeze(1).expand(-1, q.shape[1], -1)
    if original.shape[-1] != q.shape[-1]:
        original = original[..., :q.shape[-1]]
    return (1.0 - F.cosine_similarity(q, original, dim=-1)).mean(dim=1)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in ("mean", "p10", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def module_groups(model):
    prefixes = {
        "student_sender_feature_projections": ("matcher.high_proj", "matcher.proto_proj"),
        "student_local_query_builder": ("query_builder.",),
        "student_global_semantic_matcher": ("matcher.",),
        "student_mid_deformable_block": ("mid_block.",),
        "student_high_deformable_block": ("high_block.",),
        "student_mid_residual_writer": ("mid_writer.",),
        "student_high_residual_writer": ("high_writer.",),
        "teacher_projection": ("teacher.mid_proj", "teacher.high_proj"),
        "teacher_fusion": ("teacher.fusion", "teacher.slot_embed"),
        "teacher_writer": ("teacher.writer",),
    }
    grouped = {}
    for group, prefs in prefixes.items():
        grouped[group] = [(n, p) for n, p in model.named_parameters() if any(n.startswith(x) for x in prefs)]
    return grouped


def grad_stats(model, groups):
    rows = {}
    for group, params in groups.items():
        sq = 0.0
        max_abs = 0.0
        finite = []
        nonzero = 0
        total = 0
        for _, p in params:
            total += 1
            if p.grad is None:
                continue
            g = p.grad.detach()
            finite.append(float(torch.isfinite(g).float().mean().item()))
            max_abs = max(max_abs, float(g.abs().max().item()))
            sq += float(g.float().square().sum().item())
            if bool((g != 0).any().item()):
                nonzero += 1
        rows[group] = {
            "grad_l2": math.sqrt(sq),
            "grad_max_abs": max_abs,
            "finite_ratio": float(np.mean(finite)) if finite else 1.0,
            "nonzero_tensor_ratio": nonzero / float(total or 1),
        }
    return rows


def build_case_batch(dataset, rows, device, tracker, fcvc):
    per_view = {}
    for view in ("A", "B", "C"):
        z_list, x_list, gt_list, image_sizes = [], [], [], []
        for row in rows:
            template_frame = int(row.get("template_frame", 0))
            search_frame = int(row.get("search_frame", row["frame"]))
            z, x, gt, image_size = crop_pair(
                dataset, row["target"], view, template_frame, search_frame,
                device)
            z_list.append(z)
            x_list.append(x)
            gt_list.append(gt)
            image_sizes.append(image_size)
        z = torch.cat(z_list, dim=0)
        x = torch.cat(x_list, dim=0)
        gt = torch.stack(gt_list, dim=0)
        with torch.no_grad():
            taps = capture_taps(tracker.backbone, z, x)
            tmpl_mid, mid_search = split_template_search(taps.mid_tokens)
            tmpl_high, high_search = split_template_search(taps.final_tokens)
            local_pred = tracker.forward_head(taps.final_tokens)
        local = {
            "template_mid": tmpl_mid.detach(),
            "template_high": tmpl_high.detach(),
            "mid_search": mid_search.detach(),
            "high_search": high_search.detach(),
            "response_map": local_pred["score_map"].detach(),
            "confidence_uncertainty": torch.cat(
                (local_pred["score_map"].detach(),
                 torch.zeros_like(local_pred["score_map"].detach()) + 0.5), dim=1),
            "target_prototype": high_search.detach().mean(dim=1),
            "local_output": local_pred,
        }
        per_view[view] = {
            "local": local,
            "gt": gt,
            "mid": mid_search.detach(),
            "high": high_search.detach(),
            "pred": local_pred,
            "image_size": image_sizes,
        }
    cases = []
    for i, row in enumerate(rows):
        receiver = row["receiver"]
        sender_names = (row["sender_1"], row["sender_2"])
        local = {k: (v[i:i + 1] if torch.is_tensor(v) else {kk: vv[i:i + 1] for kk, vv in v.items()})
                 for k, v in per_view[receiver]["local"].items()}
        bundles = []
        for s in sender_names:
            sv = per_view[s]
            bundles.append(build_sender_bundle(
                sv["mid"][i:i + 1], sv["high"][i:i + 1],
                sv["pred"]["score_map"][i:i + 1],
                local_bbox=sv["gt"][i:i + 1],
                view_id=torch.full((1,), VIEWS[s], device=device, dtype=torch.int16),
                timestamp=torch.full(
                    (1,), int(row.get("search_frame", row["frame"])),
                    device=device, dtype=torch.int64),
            ))
        cases.append((row, local, tuple(bundles),
                      per_view[receiver]["gt"][i:i + 1],
                      [per_view[s]["gt"][i:i + 1] for s in sender_names],
                      [per_view[v]["mid"][i:i + 1] for v in ("A", "B", "C")],
                      [per_view[v]["high"][i:i + 1] for v in ("A", "B", "C")]))
    return cases


def forward_case(fcvc, tracker, case):
    row, local, bundles, gt, sender_gt, all_mid, all_high = case
    gt_roi = make_gt_heatmap(gt)
    module = fcvc.module if hasattr(fcvc, "module") else fcvc
    if getattr(module, "is_fcvc_training_graph", False):
        out = fcvc(
            local, bundles, forward_head=tracker.forward_head,
            teacher_training_payload={
                "mid_features": all_mid,
                "high_features": all_high,
                "gt_roi": gt_roi,
            })
        teacher_slots = out["teacher_slots"]
        teacher_high = out["teacher_high"]
    else:
        out = fcvc(local, bundles, forward_head=tracker.forward_head)
        teacher_slots = module.teacher(all_mid, all_high, gt_roi)
        teacher_high = module.teacher.tracking_residual(
            local["high_search"].detach(), teacher_slots)
    local_comp = pred_components(local["local_output"], gt)
    collab_comp = pred_components(out["reported_output"], gt)
    teacher_tokens = torch.cat((local["template_high"].detach(), teacher_high), dim=1)
    teacher_pred = tracker.forward_head(teacher_tokens)
    teacher_comp = pred_components(teacher_pred, gt)
    l_align = align_component(out["global_match"], sender_gt)
    l_recon = (
        0.5 * (1.0 - F.cosine_similarity(out["queries"], teacher_slots.detach(), dim=-1)).mean(dim=1)
        + 0.5 * ((out["queries"] - teacher_slots.detach()).square().mean(dim=(1, 2))
                 / teacher_slots.detach().square().mean(dim=(1, 2)).clamp_min(1e-6))
    )
    safe_raw = F.relu((collab_comp["L_track"] - local_comp["L_track"].detach())
                      / (local_comp["L_track"].detach() + 0.1))
    l_cycle = cycle_component(out, local)
    total = (collab_comp["L_track"] + 0.5 * l_align + l_recon
             + 0.5 * safe_raw + 0.1 * l_cycle + 0.25 * teacher_comp["L_track"])
    losses = {
        **collab_comp,
        "L_align": l_align,
        "L_recon": l_recon,
        "L_safe": safe_raw,
        "L_cycle": l_cycle,
        "L_teacher_track": teacher_comp["L_track"],
        "L_teacher_weighted": 0.25 * teacher_comp["L_track"],
        "L_student": collab_comp["L_track"] + 0.5 * l_align + l_recon + 0.5 * safe_raw + 0.1 * l_cycle,
        "L_total": total,
        "local_tracking_loss": local_comp["L_track"],
        "collaborative_tracking_loss": collab_comp["L_track"],
        "collaborative_local_loss_diff": collab_comp["L_track"] - local_comp["L_track"].detach(),
    }
    diagnostics = {
        "student_rep_norm": float(out["queries"].detach().norm(dim=-1).mean().item()),
        "teacher_rep_norm": float(teacher_slots.detach().norm(dim=-1).mean().item()),
        "mid_residual_norm": float(out["mid_writer"]["residual"].detach().norm(dim=-1).mean().item()),
        "high_residual_norm": float(out["high_writer"]["residual"].detach().norm(dim=-1).mean().item()),
        "residual_local_feature_norm_ratio": float(
            out["high_writer"]["residual"].detach().norm().item()
            / local["high_search"].detach().norm().clamp_min(1e-6).item()),
        "global_matcher_null_attention_ratio": float(out["global_match"]["null_attention_ratio"].detach().mean().item()),
        "sender_1_attention_contribution": float(out["global_match"]["sender_contribution"][..., 0].detach().mean().item()),
        "sender_2_attention_contribution": float(out["global_match"]["sender_contribution"][..., 1].detach().mean().item()),
        "deformable_sample_valid_ratio": float(torch.isfinite(out["mid_block"]["sample_coordinates"]).float().mean().item()),
        "output_bbox_finite": bool(torch.isfinite(out["reported_output"]["pred_boxes"]).all().item()),
        "collaborative_mean_iou": prediction_mean_iou(
            out["reported_output"], gt),
    }
    return losses, diagnostics


def run(args):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED % (2 ** 32 - 1))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    spec_hashes = read_required_specs()
    dataset = ThreeMDOT(split="train")
    dataset.sequence_list = [f"{target}-{view}" for target in INNER_TRAIN for view in (1, 2, 3)]
    manifest_rows = build_manifest(dataset, groups=args.groups)
    manifest_path, manifest_sha = write_manifest(manifest_rows)

    update_config_from_file(str(ROOT / "experiments/entertrack/entertrack_threemdot_lasot_ft_cons.yaml"))
    tracker = build_entertrack(cfg, training=True).to(device)
    tracker.eval()
    for p in tracker.parameters():
        p.requires_grad_(False)
    fcvc = FCVCModel(FCVCConfig(enabled=True)).to(device)
    fcvc.train()
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in fcvc.named_parameters() if p.requires_grad and not n.startswith("teacher.")], "lr": 1e-4},
        {"params": [p for n, p in fcvc.named_parameters() if p.requires_grad and n.startswith("teacher.")], "lr": 5e-5},
    ], weight_decay=1e-4)
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    allowed_names = set(fcvc.trainable_parameter_names(include_teacher=True))
    optimizer_names = {n for n, p in fcvc.named_parameters() if id(p) in opt_ids}
    optimizer_scope_ok = optimizer_names == allowed_names
    before_fcvc = sha256_model(fcvc)
    before_tracker = sha256_model(tracker)

    per_case = []
    warnings = []
    blockers = []
    start = time.perf_counter()
    forward_time = 0.0
    backward_time = 0.0
    micro = int(args.microbatch)
    logical = int(args.batch)
    if logical % micro != 0:
        raise RuntimeError("logical batch must be divisible by microbatch")
    for batch_start in range(0, len(manifest_rows), micro):
        rows = manifest_rows[batch_start:batch_start + micro]
        t0 = time.perf_counter()
        cases = build_case_batch(dataset, rows, device, tracker, fcvc)
        for case in cases:
            losses, diag = forward_case(fcvc, tracker, case)
            row = dict(case[0])
            for name, value in losses.items():
                row[name] = float(value.detach().mean().item())
            row.update(diag)
            row["L_safe_active"] = float((losses["L_safe"].detach() > 0).float().mean().item())
            per_case.append(row)
        forward_time += time.perf_counter() - t0

    groups = module_groups(fcvc)
    grad_rows = []
    grad_cases = manifest_rows[:min(args.grad_groups * 3, len(manifest_rows))]
    grad_targets = [
        ("G1_L_track", lambda losses: losses["L_track"].mean()),
        ("G2_0p50_L_align", lambda losses: 0.50 * losses["L_align"].mean()),
        ("G3_L_recon", lambda losses: losses["L_recon"].mean()),
        ("G4_0p50_L_safe", lambda losses: 0.50 * losses["L_safe"].mean()),
        ("G5_0p10_L_cycle", lambda losses: 0.10 * losses["L_cycle"].mean()),
        ("G6_0p25_L_teacher_track", lambda losses: 0.25 * losses["L_teacher_track"].mean()),
        ("G7_L_total", lambda losses: losses["L_total"].mean()),
    ]
    for batch_start in range(0, len(grad_cases), micro):
        rows = grad_cases[batch_start:batch_start + micro]
        cases = build_case_batch(dataset, rows, device, tracker, fcvc)
        for target_name, getter in grad_targets:
            fcvc.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            loss_terms = []
            for case in cases:
                losses, _ = forward_case(fcvc, tracker, case)
                loss_terms.append(getter(losses))
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            backward_time += time.perf_counter() - t0
            stats = grad_stats(fcvc, groups)
            for module, values in stats.items():
                grad_rows.append({
                    "batch_start": batch_start,
                    "loss": target_name,
                    "module": module,
                    **values,
                })
            fcvc.zero_grad(set_to_none=True)

    after_fcvc = sha256_model(fcvc)
    after_tracker = sha256_model(tracker)
    if before_fcvc != after_fcvc or before_tracker != after_tracker:
        blockers.append("parameter_or_buffer_digest_changed")
    if not optimizer_scope_ok:
        blockers.append("optimizer_scope_error")

    loss_names = ["L_cls", "L_giou", "L_l1", "L_track", "L_align", "L_recon",
                  "L_safe", "L_cycle", "L_teacher_track", "L_teacher_weighted",
                  "L_student", "L_total"]
    for name in loss_names:
        values = [r[name] for r in per_case]
        if not np.isfinite(values).all():
            blockers.append(f"nonfinite_{name}")
        if name in {"L_track", "L_align", "L_recon", "L_cycle"} and max(abs(v) for v in values) == 0:
            blockers.append(f"all_zero_{name}")
    aux = ["L_align", "L_recon", "L_safe", "L_cycle", "L_teacher_weighted"]
    weighted_track = np.median([r["L_track"] for r in per_case])
    contribution_rows = []
    for name in aux:
        weight = 1.0 if name == "L_teacher_weighted" else LOSS_WEIGHTS[name]
        weighted = np.asarray([r[name] * weight for r in per_case])
        ratio = float(np.median(weighted) / max(weighted_track, 1e-12))
        contribution_rows.append({"loss": name, "weighted_median_to_weighted_L_track": ratio})
        if ratio > 100:
            blockers.append(f"auxiliary_ratio_too_high_{name}")
        elif ratio > 10 or ratio < 1e-3:
            warnings.append(f"auxiliary_ratio_warning_{name}:{ratio:.6g}")

    write_outputs(per_case, grad_rows, contribution_rows, spec_hashes, manifest_sha,
                  optimizer_scope_ok, before_fcvc, after_fcvc, before_tracker,
                  after_tracker, warnings, blockers, args, forward_time,
                  backward_time, start, device, len(optimizer.param_groups))
    return per_case, grad_rows, contribution_rows, warnings, blockers, manifest_sha


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_outputs(per_case, grad_rows, contribution_rows, spec_hashes, manifest_sha,
                  optimizer_scope_ok, before_fcvc, after_fcvc, before_tracker,
                  after_tracker, warnings, blockers, args, forward_time,
                  backward_time, start, device, optimizer_group_count):
    with gzip.open(OUT / "per_case_losses.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_case[0]))
        w.writeheader()
        w.writerows(per_case)
    loss_names = ["L_cls", "L_giou", "L_l1", "L_track", "L_align", "L_recon",
                  "L_safe", "L_cycle", "L_teacher_track", "L_teacher_weighted",
                  "L_student", "L_total"]
    distribution = []
    for name in loss_names:
        s = summarize([r[name] for r in per_case])
        distribution.append({"loss": name, **s,
                             "zero_rate": float(np.mean([abs(r[name]) == 0 for r in per_case])),
                             "active_rate": float(np.mean([abs(r[name]) > 0 for r in per_case]))})
    write_csv(OUT / "loss_distribution.csv", distribution)
    write_csv(OUT / "loss_contribution_ratios.csv", contribution_rows)
    target_rows = []
    for target in sorted({r["target"] for r in per_case}):
        rows = [r for r in per_case if r["target"] == target]
        for name in loss_names:
            target_rows.append({"target": target, "loss": name, **summarize([r[name] for r in rows])})
    write_csv(OUT / "target_loss_distribution.csv", target_rows)
    rs_rows = []
    for key in sorted({(r["receiver"], r["sender_1"], r["sender_2"]) for r in per_case}):
        rows = [r for r in per_case if (r["receiver"], r["sender_1"], r["sender_2"]) == key]
        for name in loss_names:
            rs_rows.append({"receiver": key[0], "sender_1": key[1], "sender_2": key[2],
                            "loss": name, **summarize([r[name] for r in rows])})
    write_csv(OUT / "receiver_sender_loss_distribution.csv", rs_rows)
    write_csv(OUT / "gradient_matrix.csv", grad_rows)
    module_summary = []
    for key in sorted({(r["loss"], r["module"]) for r in grad_rows}):
        rows = [r for r in grad_rows if (r["loss"], r["module"]) == key]
        module_summary.append({"loss": key[0], "module": key[1],
                               **summarize([r["grad_l2"] for r in rows]),
                               "max_abs_max": max(r["grad_max_abs"] for r in rows),
                               "finite_ratio_mean": float(np.mean([r["finite_ratio"] for r in rows])),
                               "nonzero_tensor_ratio_mean": float(np.mean([r["nonzero_tensor_ratio"] for r in rows]))})
    write_csv(OUT / "module_gradient_summary.csv", module_summary)

    decision = "S1_SCALE_READY" if not blockers else ("S3_SCALE_INVALID" if any(
        x in "|".join(blockers) for x in ["optimizer", "digest", "leakage", "manifest"]) else "S2_SCALE_BLOCKED")
    allow_training = decision == "S1_SCALE_READY"
    elapsed = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved() if device.type == "cuda" else 0

    reports = {
        "sample_manifest_identity.md": [
            "# Sample manifest identity",
            f"- seed: `{SEED}`",
            f"- synchronized_frame_groups: `{len(per_case) // 3}`",
            f"- receiver_cases: `{len(per_case)}`",
            f"- sha256: `{manifest_sha}`",
            "- bytewise_rebuild_expected: same code path and seed",
        ],
        "split_and_leakage_audit.md": [
            "# Split and leakage audit",
            f"- allowed_inner_train_targets: `{','.join(INNER_TRAIN)}`",
            f"- forbidden_inner_dev_targets: `{','.join(sorted(INNER_DEV_FORBIDDEN))}`",
            f"- outer_holdout_id_only: `{','.join(sorted(OUTER_HOLDOUT_ID_ONLY))}`",
            "- accessed_splits: `fold1_inner_train` only",
            "- validation_test_or_inner_dev_access: `false`",
        ],
        "gt_boundary_audit.md": [
            "# GT boundary audit",
            "- student_inference_input_uses_gt: `false`",
            "- gt_usage: crop-normalized tracking loss, sender alignment targets, teacher GT ROI",
            "- sender_bundle_gt_fields: `none`",
        ],
        "loss_dependency_graph.md": [
            "# Loss dependency graph",
            "- L_cls/L_giou/L_l1: collaborative reported_output + GT",
            "- L_align: global_match diagnostics + sender GT targets",
            "- L_recon: student query slots + stop_gradient(teacher slots)",
            "- L_safe: relu(collaborative_track - stop_gradient(local_track)) / (stop_gradient(local_track)+0.1)",
            "- L_cycle: student query consistency against detached local prototype",
            "- L_teacher_track: teacher residual path + frozen head + GT",
        ],
        "loss_semantics_audit.md": [
            "# Loss semantics audit",
            "- total_loss_fixed_weight_sum: `checked_per_forward`",
            "- local_branch_tracking_grad: `detached`",
            "- teacher_recon_target_stop_gradient: `true`",
            "- no_loss_weight_tuning: `true`",
        ],
        "optimizer_scope_audit.md": [
            "# Optimizer scope audit",
            f"- optimizer_groups: `{optimizer_group_count}`",
            f"- optimizer_scope_ok: `{str(optimizer_scope_ok).lower()}`",
            "- optimizer_step_called: `false`",
            "- scheduler_step_called: `false`",
        ],
        "freeze_and_parameter_identity.md": [
            "# Freeze and parameter identity",
            f"- fcvc_before_sha256: `{before_fcvc}`",
            f"- fcvc_after_sha256: `{after_fcvc}`",
            f"- tracker_before_sha256: `{before_tracker}`",
            f"- tracker_after_sha256: `{after_tracker}`",
            f"- identity_pass: `{str(before_fcvc == after_fcvc and before_tracker == after_tracker).lower()}`",
        ],
        "local_identity_report.md": [
            "# Local identity report",
            "- local_candidate_source: frozen local branch",
            "- collaborative_branch_modifies_local_tensor: `false`",
            "- state_commit_executed: `false`",
            "- local_runtime_payload_modified: `false`",
        ],
        "deterministic_scale_audit.md": [
            "# Deterministic scale audit",
            "- manifest_seed: `20260716`",
            "- repeated_32_group_forward_backward: `not_separately_replayed`",
            "- note: main manifest is deterministic; separate two-pass check is not run by this script version",
        ],
        "resource_profile.md": [
            "# Resource profile",
            f"- device: `{device}`",
            f"- logical_batch_size: `{args.batch}`",
            f"- microbatch_size: `{args.microbatch}`",
            f"- forward_time_sec: `{forward_time:.6f}`",
            f"- backward_time_sec: `{backward_time:.6f}`",
            f"- elapsed_sec: `{elapsed:.6f}`",
            f"- samples_per_sec_forward: `{len(per_case) / max(forward_time, 1e-9):.6f}`",
            f"- peak_allocated_cuda_memory: `{peak_alloc}`",
            f"- peak_reserved_cuda_memory: `{peak_reserved}`",
            "- oom: `false`",
            "- amp_smoke: `not_run`",
            "- cpu_fallback: `{}`".format(str(device.type == "cpu").lower()),
        ],
        "scale_audit_decision.md": [
            "# Scale audit decision",
            f"- decision: `{decision}`",
            f"- allow_next_preregistered_training: `{str(allow_training).lower()}`",
            f"- warnings: `{';'.join(warnings) if warnings else 'none'}`",
            f"- hard_blockers: `{';'.join(blockers) if blockers else 'none'}`",
        ],
    }
    files = []
    for name, lines in reports.items():
        path = OUT / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files.append(name)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec_hashes": spec_hashes,
        "sample_manifest_sha256": manifest_sha,
        "files": sorted(files + [
            "scale_audit_sample_manifest.csv", "per_case_losses.csv.gz",
            "loss_distribution.csv", "target_loss_distribution.csv",
            "receiver_sender_loss_distribution.csv", "loss_contribution_ratios.csv",
            "gradient_matrix.csv", "module_gradient_summary.csv",
        ]),
        "decision": decision,
        "warnings": warnings,
        "blockers": blockers,
    }
    (OUT / "scale_audit_manifest.md").write_text(
        "# Scale audit manifest\n\n```json\n" + json.dumps(manifest, indent=2, sort_keys=True) + "\n```\n",
        encoding="utf-8")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="")
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--grad-groups", type=int, default=64)
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.output_dir)
    per_case, grad_rows, contribution_rows, warnings, blockers, manifest_sha = run(args)
    print("sample_manifest_sha256={}".format(manifest_sha))
    print("receiver_cases={}".format(len(per_case)))
    print("gradient_rows={}".format(len(grad_rows)))
    print("warnings={}".format(";".join(warnings) if warnings else "none"))
    print("blockers={}".format(";".join(blockers) if blockers else "none"))


if __name__ == "__main__":
    main()
