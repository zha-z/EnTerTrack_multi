#!/usr/bin/env python3
"""Freeze D2-P1 inventory and extract B0 local representations."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from lib.config.entertrack.config import cfg, update_config_from_file  # noqa: E402
from lib.models.entertrack.entertrack import build_entertrack  # noqa: E402
from lib.train.admin import env_settings  # noqa: E402
from lib.train.data.image_loader import opencv_loader  # noqa: E402
from lib.train.data.processing_utils import (  # noqa: E402
    sample_target,
    transform_image_to_crop,
)
from lib.train.target_prompt_asymmetric_degradation import (  # noqa: E402
    _normalized_box_to_pixels,
)


VIEWS = ("A", "B", "C")
VIEW_SUFFIX = {"A": "1", "B": "2", "C": "3"}
GROUP_ORDER = {"clean": 0, "synthetic": 1, "natural": 2}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
CHECKPOINT_REL = Path(
    "output/diagnostics/b0_abc_plain/run_20260818_seed42_4gpu_r001/"
    "checkpoints/train/entertrack/b0_abc_plain_4gpu/"
    "EnTeRTrack_ep0025.pth.tar")
CHECKPOINT_SHA256 = (
    "363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3")
E3_CONFIG = ROOT / "experiments/entertrack/target_prompt_collaboration_e3.yaml"
SPLIT_FILES = {
    "train": ROOT / "lib/train/data_specs/threemdot/threemdot_train.txt",
    "val": ROOT / "lib/train/data_specs/threemdot/threemdot_val.txt",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("refusing to write empty CSV: {}".format(path))
    columns = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_vector(path, dtype=np.int64):
    values = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip().strip(",")
            if text:
                values.append(text)
    return np.asarray(values, dtype=dtype)


def read_boxes(path):
    boxes = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def systematic_indices(length, count):
    if count < 0 or length < 0 or count > length:
        raise ValueError("invalid systematic sample size")
    if count == 0:
        return []
    indices = [int(math.floor((index + 0.5) * length / count))
               for index in range(count)]
    if len(set(indices)) != count or min(indices) < 0 or max(indices) >= length:
        raise RuntimeError("systematic sampling produced invalid indices")
    return indices


def previous_eligible(mask, frame_id, max_gap=200):
    start = max(0, int(frame_id) - int(max_gap))
    candidates = np.flatnonzero(np.asarray(mask[start:frame_id], dtype=bool))
    if candidates.size == 0:
        return None
    return int(start + candidates[-1])


def _sequence_dir(dataset_root, target_id, view):
    sequence_name = "{}-{}".format(target_id, VIEW_SUFFIX[view])
    return Path(dataset_root) / target_id / sequence_name


def _load_target(dataset_root, target_id):
    output = {}
    lengths = []
    for view in VIEWS:
        sequence_dir = _sequence_dir(dataset_root, target_id, view)
        required = (
            sequence_dir / "groundtruth.txt",
            sequence_dir / "occlusion.txt",
            sequence_dir / "out_of_view.txt",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError("MISSING_DIAGNOSTIC annotation: {}".format(
                sequence_dir))
        boxes = read_boxes(required[0])
        occlusion = read_vector(required[1]).astype(bool)
        out_of_view = read_vector(required[2]).astype(bool)
        if not (len(boxes) == len(occlusion) == len(out_of_view)):
            raise RuntimeError("annotation length mismatch: {}".format(
                sequence_dir))
        valid = (np.isfinite(boxes).all(axis=1)
                 & (boxes[:, 2] > 0) & (boxes[:, 3] > 0))
        visible = (~occlusion) & (~out_of_view) & valid
        output[view] = {
            "sequence_name": sequence_dir.name,
            "boxes": boxes,
            "occlusion": occlusion,
            "out_of_view": out_of_view,
            "valid": valid,
            "visible": visible,
        }
        lengths.append(len(boxes))
    if len(set(lengths)) != 1:
        raise RuntimeError("synchronized view lengths differ: {}".format(
            target_id))
    return output


def _candidate(row, template_frame_id, group, balanced, pair_id=""):
    box = row["box"]
    return {
        "sample_id": "d2p1-{}-{}-{:06d}-{}".format(
            row["split"], row["sequence_name"], row["search_frame_id"],
            group),
        "split": row["split"],
        "target_id": row["target_id"],
        "view": row["view"],
        "sequence_name": row["sequence_name"],
        "template_frame_id": int(template_frame_id),
        "search_frame_id": int(row["search_frame_id"]),
        "group": group,
        "pair_id": pair_id,
        "balanced_primary": bool(balanced),
        "occlusion": int(row["occlusion"]),
        "out_of_view": int(row["out_of_view"]),
        "bbox_valid": bool(row["bbox_valid"]),
        "bbox_x": float(box[0]),
        "bbox_y": float(box[1]),
        "bbox_w": float(box[2]),
        "bbox_h": float(box[3]),
        "causal_max_gap": 200,
        "template_rule": row["template_rule"],
    }


def build_inventory(dataset_root):
    rows = []
    manifest_splits = {}
    annotation_paths = []
    train_targets = set()
    val_targets = set()
    for split in ("train", "val"):
        sequence_names = [line.strip() for line in
                          SPLIT_FILES[split].read_text(encoding="utf-8").splitlines()
                          if line.strip()]
        if any(name.lower().endswith("test") for name in sequence_names):
            raise RuntimeError("test sequence found in allowed split")
        targets = sorted({name.rsplit("-", 1)[0] for name in sequence_names})
        (train_targets if split == "train" else val_targets).update(targets)
        natural_by_stratum = {}
        clean_by_stratum = {}
        no_template_natural = 0
        no_template_clean = 0
        for target_id in targets:
            data = _load_target(dataset_root, target_id)
            for view in VIEWS:
                sequence_dir = _sequence_dir(dataset_root, target_id, view)
                annotation_paths.extend([
                    sequence_dir / "groundtruth.txt",
                    sequence_dir / "occlusion.txt",
                    sequence_dir / "out_of_view.txt",
                ])
            common_visible = np.logical_and.reduce(
                [data[view]["visible"] for view in VIEWS])
            for view in VIEWS:
                item = data[view]
                stratum = (target_id, view)
                natural = []
                clean = []
                natural_mask = (item["occlusion"]
                                & (~item["out_of_view"]) & item["valid"])
                for frame_id in np.flatnonzero(natural_mask):
                    template_id = previous_eligible(item["visible"], frame_id)
                    if template_id is None:
                        no_template_natural += 1
                        continue
                    natural.append({
                        "split": split, "target_id": target_id, "view": view,
                        "sequence_name": item["sequence_name"],
                        "search_frame_id": int(frame_id),
                        "template_frame_id": template_id,
                        "occlusion": True,
                        "out_of_view": bool(item["out_of_view"][frame_id]),
                        "bbox_valid": bool(item["valid"][frame_id]),
                        "box": item["boxes"][frame_id],
                        "template_rule": "latest_prior_same_view_visible",
                    })
                for frame_id in np.flatnonzero(common_visible):
                    if frame_id <= 0:
                        continue
                    template_id = previous_eligible(common_visible, frame_id)
                    if template_id is None:
                        no_template_clean += 1
                        continue
                    clean.append({
                        "split": split, "target_id": target_id, "view": view,
                        "sequence_name": item["sequence_name"],
                        "search_frame_id": int(frame_id),
                        "template_frame_id": template_id,
                        "occlusion": False,
                        "out_of_view": bool(item["out_of_view"][frame_id]),
                        "bbox_valid": bool(item["valid"][frame_id]),
                        "box": item["boxes"][frame_id],
                        "template_rule": "latest_prior_common_visible",
                    })
                natural_by_stratum[stratum] = natural
                clean_by_stratum[stratum] = clean

        split_rows = []
        stratum_summary = []
        for target_id in targets:
            for view in VIEWS:
                stratum = (target_id, view)
                natural = natural_by_stratum[stratum]
                clean = clean_by_stratum[stratum]
                quota = min(len(natural), len(clean))
                natural_selected = set(systematic_indices(len(natural), quota))
                clean_selected = systematic_indices(len(clean), quota)
                for index, item in enumerate(natural):
                    split_rows.append(_candidate(
                        item, item["template_frame_id"], "natural",
                        index in natural_selected))
                for pair_index, clean_index in enumerate(clean_selected):
                    item = clean[clean_index]
                    pair_id = "d2p1-{}-{}-{}-{:05d}".format(
                        split, target_id, view, pair_index)
                    split_rows.append(_candidate(
                        item, item["template_frame_id"], "clean", True,
                        pair_id=pair_id))
                    split_rows.append(_candidate(
                        item, item["template_frame_id"], "synthetic", True,
                        pair_id=pair_id))
                stratum_summary.append({
                    "target_id": target_id,
                    "view": view,
                    "natural_raw": len(natural),
                    "clean_candidates": len(clean),
                    "balanced_quota": quota,
                })
        split_rows.sort(key=lambda row: (
            row["target_id"], row["view"], int(row["search_frame_id"]),
            GROUP_ORDER[row["group"]]))
        rows.extend(split_rows)
        natural_raw = [row for row in split_rows if row["group"] == "natural"]
        natural_balanced = [row for row in natural_raw if row["balanced_primary"]]
        target_counts = {}
        for row in natural_raw:
            target_counts[row["target_id"]] = target_counts.get(row["target_id"], 0) + 1
        shares = sorted(target_counts.values(), reverse=True)
        manifest_splits[split] = {
            "target_count": len(targets),
            "sequence_count": len(sequence_names),
            "inventory_rows": len(split_rows),
            "natural_raw_rows": len(natural_raw),
            "natural_balanced_rows": len(natural_balanced),
            "clean_balanced_rows": sum(
                row["group"] == "clean" for row in split_rows),
            "synthetic_balanced_rows": sum(
                row["group"] == "synthetic" for row in split_rows),
            "natural_no_causal_template": no_template_natural,
            "clean_no_causal_template": no_template_clean,
            "natural_raw_top1_target_share": (
                shares[0] / len(natural_raw) if shares else float("nan")),
            "natural_raw_top3_target_share": (
                sum(shares[:3]) / len(natural_raw) if shares else float("nan")),
            "strata": stratum_summary,
        }
    if train_targets & val_targets:
        raise RuntimeError("train/val target overlap: {}".format(
            sorted(train_targets & val_targets)))
    if not rows:
        raise RuntimeError("empty inventory")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate sample ids")
    pair_counts = {}
    for row in rows:
        if row["pair_id"]:
            pair_counts.setdefault(row["pair_id"], set()).add(row["group"])
    if any(groups != {"clean", "synthetic"}
           for groups in pair_counts.values()):
        raise RuntimeError("incomplete C/S pair")
    annotation_digest = hashlib.sha256()
    for path in sorted(set(annotation_paths)):
        rel = str(Path(path).relative_to(dataset_root))
        annotation_digest.update(rel.encode("utf-8") + b"\0")
        annotation_digest.update(Path(path).read_bytes())
    return rows, {
        "phase": "inventory_freeze_before_model_forward",
        "dataset_root_recorded_as": "env_settings().threemdot_dir",
        "allowed_splits": ["train", "val"],
        "official_test_accessed": False,
        "source_head": "2abaeea8adaa6a0a2b4eaf93460c1e1780336ccf",
        "causal_template_max_gap": 200,
        "balanced_sampling": "midpoint_systematic_per_split_target_view",
        "annotation_aggregate_sha256": annotation_digest.hexdigest(),
        "splits": manifest_splits,
    }


def freeze_inventory(output_dir, dataset_root):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sample_inventory.csv"
    manifest_path = output_dir / "sample_inventory_manifest.json"
    if csv_path.exists() or manifest_path.exists():
        raise RuntimeError("inventory destination already exists")
    rows, manifest = build_inventory(dataset_root)
    staging = Path(tempfile.mkdtemp(prefix="d2-p1-inventory-",
                                    dir=str(output_dir)))
    try:
        staged_csv = staging / csv_path.name
        staged_manifest = staging / manifest_path.name
        write_csv(staged_csv, rows)
        manifest.update({
            "inventory_file": csv_path.name,
            "inventory_rows": len(rows),
            "inventory_sha256": sha256_file(staged_csv),
            "columns": list(rows[0]),
        })
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(str(staged_csv), str(csv_path))
        os.replace(str(staged_manifest), str(manifest_path))
    finally:
        try:
            staging.rmdir()
        except OSError:
            pass
    print(json.dumps(manifest, indent=2, sort_keys=True))


def score_entropy(score_map):
    flat = score_map.flatten(1)
    denominator = flat.sum(dim=1, keepdim=True)
    safe = denominator.isfinite() & (denominator > 1e-12)
    probabilities = flat / denominator.clamp_min(1e-12)
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=1)
    entropy = entropy / math.log(flat.shape[1])
    return torch.where(safe[:, 0], entropy,
                       torch.full_like(entropy, float("nan")))


def prompt_statistics(prompt, topk_scores):
    normalized = torch.nn.functional.normalize(prompt, dim=-1, eps=1e-12)
    cosine = torch.matmul(normalized, normalized.transpose(1, 2))
    mask = ~torch.eye(prompt.shape[1], device=prompt.device,
                      dtype=torch.bool)[None]
    off_diagonal = cosine.masked_select(mask).view(prompt.shape[0], -1)
    norms = prompt.norm(dim=-1)
    return {
        "prompt_topk_score_mean": topk_scores.mean(dim=1),
        "prompt_topk_score_std": topk_scores.std(dim=1, unbiased=False),
        "prompt_topk_score_min": topk_scores.min(dim=1).values,
        "prompt_topk_score_max": topk_scores.max(dim=1).values,
        "prompt_top1_top8_gap": topk_scores[:, 0] - topk_scores[:, -1],
        "prompt_norm_mean": norms.mean(dim=1),
        "prompt_norm_std": norms.std(dim=1, unbiased=False),
        "prompt_pairwise_cos_mean": off_diagonal.mean(dim=1),
        "prompt_pairwise_cos_std": off_diagonal.std(dim=1, unbiased=False),
        "prompt_pairwise_cos_min": off_diagonal.min(dim=1).values,
        "prompt_pairwise_cos_max": off_diagonal.max(dim=1).values,
    }


def _checkpoint_state(path):
    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint.get("net", checkpoint.get("model", checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint
    return {(key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()}


def load_frozen_model(device):
    checkpoint = ROOT / CHECKPOINT_REL
    actual_sha = sha256_file(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise RuntimeError("B0 checkpoint SHA256 mismatch")
    resolved = copy.deepcopy(cfg)
    update_config_from_file(str(E3_CONFIG), base_cfg=resolved)
    model = build_entertrack(resolved, training=True)
    model.eval().to(device)
    incoming = _checkpoint_state(checkpoint)
    current = model.state_dict()
    adapter_prefixes = ("target_prompt_collaboration.",
                        "target_prompt_extractor.")
    core_keys = sorted(key for key in current
                       if not key.startswith(adapter_prefixes))
    mismatch = [key for key in core_keys
                if key not in incoming or not torch.equal(
                    current[key].detach().cpu(), incoming[key].detach().cpu())]
    if mismatch:
        raise RuntimeError("B0 local core identity mismatch: {}".format(
            mismatch[:20]))
    audit = getattr(model, "initialization_audit", {})
    if not audit.get("strict_full_load"):
        raise RuntimeError("E3 strict B0 initialization audit missing")
    if int(model.target_prompt_extractor.prompt_k) != 8:
        raise RuntimeError("prompt K is not frozen at 8")
    return model, {
        "checkpoint": str(CHECKPOINT_REL),
        "checkpoint_sha256": actual_sha,
        "strict_full_load": True,
        "core_key_count": len(core_keys),
        "core_identity_mismatch_count": len(mismatch),
        "fresh_adapter_key_count": audit.get("fresh_adapter_key_count"),
        "prompt_k": 8,
    }


def _frame_path(dataset_root, sequence_name, frame_id):
    target_id = sequence_name.rsplit("-", 1)[0]
    return Path(dataset_root) / target_id / sequence_name / "img" / \
        "{:08d}.jpg".format(int(frame_id) + 1)


def _box_cache(dataset_root, sequence_name, cache):
    if sequence_name not in cache:
        target_id = sequence_name.rsplit("-", 1)[0]
        cache[sequence_name] = read_boxes(
            Path(dataset_root) / target_id / sequence_name / "groundtruth.txt")
    return cache[sequence_name]


def prepare_sample(row, dataset_root, box_cache):
    sequence = row["sequence_name"]
    template_id = int(row["template_frame_id"])
    search_id = int(row["search_frame_id"])
    boxes = _box_cache(dataset_root, sequence, box_cache)
    template_image = opencv_loader(str(_frame_path(
        dataset_root, sequence, template_id)))
    search_image = opencv_loader(str(_frame_path(
        dataset_root, sequence, search_id)))
    if template_image is None or search_image is None:
        raise RuntimeError("image read failed: {}".format(sequence))
    template_box = torch.as_tensor(boxes[template_id], dtype=torch.float32)
    search_box = torch.as_tensor(boxes[search_id], dtype=torch.float32)
    template_crop, _, _ = sample_target(
        template_image, template_box, 2.0, output_sz=128)
    search_crop, resize_factor, _ = sample_target(
        search_image, search_box, 4.0, output_sz=256)
    normalized_box = transform_image_to_crop(
        search_box, search_box, resize_factor,
        torch.tensor([256.0, 256.0]), normalize=True)
    template = ((template_crop.astype(np.float32) / 255.0 - MEAN)
                / STD).transpose(2, 0, 1)
    search = ((search_crop.astype(np.float32) / 255.0 - MEAN)
              / STD).transpose(2, 0, 1)
    search = torch.from_numpy(np.ascontiguousarray(search))
    if row["group"] == "synthetic":
        (x0, y0, x1, y1), _ = _normalized_box_to_pixels(
            normalized_box, 256, 256)
        search = search.clone()
        search[:, y0:y1, x0:x1] = 0.0
    return (torch.from_numpy(np.ascontiguousarray(template)), search,
            normalized_box)


def freeze_representation(output_dir, dataset_root, device, batch_size):
    output_dir = Path(output_dir).resolve()
    inventory_path = output_dir / "sample_inventory.csv"
    inventory_manifest_path = output_dir / "sample_inventory_manifest.json"
    inventory_manifest = json.loads(
        inventory_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(inventory_path) != inventory_manifest["inventory_sha256"]:
        raise RuntimeError("inventory SHA256 mismatch")
    rows = read_csv(inventory_path)
    if len(rows) != inventory_manifest["inventory_rows"]:
        raise RuntimeError("inventory row count mismatch")
    destinations = [
        output_dir / "representation_features.csv",
        output_dir / "prompt_features.npz",
        output_dir / "representation_manifest.json",
    ]
    if any(path.exists() for path in destinations):
        raise RuntimeError("representation destination already exists")
    model, model_audit = load_frozen_model(device)
    adapter_calls = {"count": 0}

    def adapter_hook(module, inputs):
        adapter_calls["count"] += 1

    hook = model.target_prompt_collaboration.register_forward_pre_hook(
        adapter_hook)
    features = []
    prompts = []
    prompt_ids = []
    box_cache = {}
    try:
        for start in range(0, len(rows), batch_size):
            current = rows[start:start + batch_size]
            prepared = [prepare_sample(row, dataset_root, box_cache)
                        for row in current]
            template = torch.stack([item[0] for item in prepared]).to(device)
            search = torch.stack([item[1] for item in prepared]).to(device)
            with torch.inference_mode():
                output = model(
                    template=template, search=search,
                    ce_template_mask=None, ce_keep_rate=None,
                    temperature=100.0, return_last_attn=False,
                    return_atp=True, training=False)
                score = output["score_map"].float()
                flat = score.flatten(1)
                sorted_score = torch.topk(flat, k=2, dim=1).values
                maximum = flat.max(dim=1).values
                minimum = flat.min(dim=1).values
                apce = ((maximum - minimum) ** 2) / (
                    ((flat - minimum[:, None]) ** 2).mean(dim=1) + 1e-8)
                search_tokens = output["backbone_feat"][:, -model.feat_len_s:]
                extraction = model.target_prompt_extractor.extract_with_metadata(
                    search_tokens, score)
                prompt = extraction["prompt"].float()
                prompt_stats = prompt_statistics(
                    prompt, extraction["topk_scores"].float())
                entropy = score_entropy(score)
                boxes = output["pred_boxes"].view(len(current), -1, 4).mean(1)
            prompt_cpu = prompt.detach().cpu().numpy().astype(np.float16)
            prompts.append(prompt_cpu)
            prompt_ids.extend(row["sample_id"] for row in current)
            for index, row in enumerate(current):
                record = {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "target_id": row["target_id"],
                    "view": row["view"],
                    "template_frame_id": int(row["template_frame_id"]),
                    "search_frame_id": int(row["search_frame_id"]),
                    "group": row["group"],
                    "pair_id": row["pair_id"],
                    "balanced_primary": row["balanced_primary"],
                    "max_score": float(maximum[index].item()),
                    "apce": float(apce[index].item()),
                    "score_map_max": float(maximum[index].item()),
                    "score_map_mean": float(flat[index].mean().item()),
                    "score_map_std": float(flat[index].std(
                        unbiased=False).item()),
                    "score_top1_top2_gap": float(
                        sorted_score[index, 0].item()
                        - sorted_score[index, 1].item()),
                    "score_entropy": float(entropy[index].item()),
                    "pred_bbox_x": float(boxes[index, 0].item()),
                    "pred_bbox_y": float(boxes[index, 1].item()),
                    "pred_bbox_w": float(boxes[index, 2].item()),
                    "pred_bbox_h": float(boxes[index, 3].item()),
                    "prompt_valid": bool(extraction["valid"][index].item()),
                }
                for name, values in prompt_stats.items():
                    record[name] = float(values[index].item())
                features.append(record)
            print("D2-P1 representation {}/{}".format(
                min(start + batch_size, len(rows)), len(rows)), flush=True)
    finally:
        hook.remove()
    if adapter_calls["count"] != 0:
        raise RuntimeError("collaboration adapter was invoked")
    if len(features) != len(rows) or prompt_ids != [
            row["sample_id"] for row in rows]:
        raise RuntimeError("representation sample order mismatch")
    prompt_array = np.concatenate(prompts, axis=0)
    if prompt_array.shape != (len(rows), 8, 192):
        raise RuntimeError("unexpected prompt artifact shape")
    staging = Path(tempfile.mkdtemp(prefix="d2-p1-representation-",
                                    dir=str(output_dir)))
    try:
        feature_path = staging / destinations[0].name
        prompt_path = staging / destinations[1].name
        manifest_path = staging / destinations[2].name
        write_csv(feature_path, features)
        np.savez_compressed(
            str(prompt_path),
            sample_id=np.asarray(prompt_ids, dtype="U64"),
            prompt=prompt_array)
        manifest = {
            "phase": "representation_freeze_before_descriptive_analysis",
            "inventory_sha256_verified": inventory_manifest["inventory_sha256"],
            "feature_file": destinations[0].name,
            "feature_rows": len(features),
            "feature_sha256": sha256_file(feature_path),
            "prompt_file": destinations[1].name,
            "prompt_shape": list(prompt_array.shape),
            "prompt_dtype": str(prompt_array.dtype),
            "prompt_sha256": sha256_file(prompt_path),
            "model": model_audit,
            "runtime_audit": {
                "model_eval": not model.training,
                "inference_mode": True,
                "adapter_forward_calls": adapter_calls["count"],
                "collaboration_used": False,
                "gt_passed_to_model": False,
                "gt_used_for_deterministic_crop": True,
                "official_test_accessed": False,
            },
            "score_entropy": "normalized Shannon entropy of score/sum(score)",
            "score_map": "raw sigmoid CENTER 16x16; no Hann window",
            "prompt_k": 8,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        for source, destination in zip(
                (feature_path, prompt_path, manifest_path), destinations):
            os.replace(str(source), str(destination))
    finally:
        try:
            staging.rmdir()
        except OSError:
            pass
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=("inventory", "representation"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root or env_settings().threemdot_dir).resolve()
    if "test" in str(args.output_dir).lower():
        raise RuntimeError("test-named output path is forbidden")
    if args.phase == "inventory":
        freeze_inventory(args.output_dir, dataset_root)
    else:
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        torch.cuda.set_device(args.gpu)
        freeze_representation(
            args.output_dir, dataset_root,
            torch.device("cuda", args.gpu), args.batch_size)


if __name__ == "__main__":
    main()
