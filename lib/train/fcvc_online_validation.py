"""Independent prediction-only online validation on held-out official-train targets."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from lib.models.entertrack.fcvc import build_sender_bundle
from lib.models.entertrack.fcvc.feature_taps import capture_taps, split_template_search
from lib.train.data.processing_utils import sample_target
from lib.train.fcvc_checkpoint import capture_rng_state, restore_rng_state
from lib.train.fcvc_pair_validation import state_digest
from lib.utils.box_ops import clip_box
from tracking.audit_fcvc_scale import read_image, tensorize


VIEWS = {"A": 1, "B": 2, "C": 3}


def _sequence_dir(root, target, view):
    return Path(root) / target / "{}-{}".format(target, VIEWS[view])


def _groundtruth(path):
    return np.loadtxt(str(path), delimiter=",", dtype=float).reshape(-1, 4)


def _map_box(prediction, reference, resize_factor, search_size=256):
    cx_prev = reference[0] + 0.5 * reference[2]
    cy_prev = reference[1] + 0.5 * reference[3]
    cx, cy, width, height = prediction
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + cx_prev - half_side
    cy_real = cy + cy_prev - half_side
    return [cx_real - 0.5 * width, cy_real - 0.5 * height, width, height]


def _prediction_box(prediction, reference, resize_factor, image_shape):
    box = prediction["pred_boxes"].reshape(-1, 4).mean(dim=0)
    box = (box * 256.0 / float(resize_factor)).detach().cpu().tolist()
    return clip_box(
        _map_box(box, reference, resize_factor),
        image_shape[0], image_shape[1], margin=1)


def _local_record(tracker, template, image, state, device):
    search_patch, resize, _ = sample_target(image, state, 4.0, 256)
    search = tensorize(search_patch, device)
    taps = capture_taps(tracker.backbone, template, search)
    template_mid, mid_search = split_template_search(taps.mid_tokens)
    template_high, high_search = split_template_search(taps.final_tokens)
    prediction = tracker.forward_head(taps.final_tokens)
    response = prediction["score_map"].detach()
    return {
        "template_mid": template_mid.detach(),
        "template_high": template_high.detach(),
        "mid_search": mid_search.detach(),
        "high_search": high_search.detach(),
        "response_map": response,
        "confidence_uncertainty": torch.cat(
            (response, torch.full_like(response, 0.5)), dim=1),
        "target_prototype": high_search.detach().mean(dim=1),
        "local_output": prediction,
    }, resize


def _iou(box, target):
    box, target = np.asarray(box, dtype=float), np.asarray(target, dtype=float)
    top_left = np.maximum(box[:2], target[:2])
    bottom_right = np.minimum(box[:2] + box[2:], target[:2] + target[2:])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = size[0] * size[1]
    union = box[2] * box[3] + target[2] * target[3] - intersection
    return float(intersection / union) if union > 0 else 0.0


def _curves(predictions, targets):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    overlaps = np.asarray([_iou(p, t) for p, t in zip(predictions, targets)])
    pred_centers = predictions[:, :2] + 0.5 * predictions[:, 2:]
    target_centers = targets[:, :2] + 0.5 * targets[:, 2:]
    errors = np.linalg.norm(pred_centers - target_centers, axis=1)
    scale = np.sqrt(np.maximum(targets[:, 2] * targets[:, 3], 1e-12))
    return {
        "auc": float(np.mean([
            (overlaps >= threshold).mean()
            for threshold in np.linspace(0.0, 1.0, 21)])),
        "precision": float((errors <= 20.0).mean()),
        "norm_precision": float(((errors / scale) <= 0.2).mean()),
        "overlaps": overlaps.tolist(),
    }


class OnlineValidator:
    def __init__(self, model, frozen_tracker, dataset_root, validation_targets,
                 device, rank=0, world_size=6, dist_module=None,
                 epsilon=1e-6, output_dir=None):
        root_text = str(Path(dataset_root).resolve()).lower()
        if "test" in root_text:
            raise ValueError("online validation refuses a test dataset path")
        self.model = model
        self.fcvc = (
            model.module.fcvc if hasattr(model, "module") else model.fcvc)
        self.tracker = frozen_tracker
        self.dataset_root = Path(dataset_root)
        self.targets = tuple(sorted(validation_targets))
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dist = dist_module
        self.epsilon = float(epsilon)
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def _run_target(self, target, max_frames=None):
        sequences = {view: _sequence_dir(self.dataset_root, target, view)
                     for view in VIEWS}
        groundtruth = {
            view: _groundtruth(path / "groundtruth.txt")
            for view, path in sequences.items()}
        length = min(len(value) for value in groundtruth.values())
        if max_frames is not None:
            length = min(length, int(max_frames))
        states = {view: groundtruth[view][0].tolist() for view in VIEWS}
        templates = {}
        local_predictions = {view: [list(states[view])] for view in VIEWS}
        collab_predictions = {view: [list(states[view])] for view in VIEWS}
        for view in VIEWS:
            image = read_image(sequences[view] / "img" / "00000001.jpg")
            patch, _, _ = sample_target(image, states[view], 2.0, 128)
            templates[view] = tensorize(patch, self.device)
        start = time.perf_counter()
        for frame in range(1, length):
            images, records, resize_factors = {}, {}, {}
            for view in VIEWS:
                images[view] = read_image(
                    sequences[view] / "img" / "{:08d}.jpg".format(frame + 1))
                records[view], resize_factors[view] = _local_record(
                    self.tracker, templates[view], images[view], states[view],
                    self.device)
            for receiver in VIEWS:
                senders = [view for view in VIEWS if view != receiver]
                bundles = tuple(build_sender_bundle(
                    records[sender]["mid_search"],
                    records[sender]["high_search"],
                    records[sender]["response_map"],
                    view_id=torch.full(
                        (1,), VIEWS[sender], device=self.device,
                        dtype=torch.int16),
                    timestamp=torch.full(
                        (1,), frame, device=self.device, dtype=torch.int64),
                ) for sender in senders)
                output = self.fcvc(
                    records[receiver], bundles,
                    forward_head=self.tracker.forward_head)
                reference = list(states[receiver])
                local_box = _prediction_box(
                    records[receiver]["local_output"], reference,
                    resize_factors[receiver], images[receiver].shape[:2])
                collab_box = _prediction_box(
                    output["reported_output"], reference,
                    resize_factors[receiver], images[receiver].shape[:2])
                if not np.isfinite(collab_box).all() or collab_box[2] <= 0 or collab_box[3] <= 0:
                    collab_box = list(local_box)
                # Safe Commit: only the local candidate advances crop/state.
                states[receiver] = list(local_box)
                local_predictions[receiver].append(list(local_box))
                collab_predictions[receiver].append(list(collab_box))
        runtime = time.perf_counter() - start
        rows = []
        for view in VIEWS:
            target_gt = groundtruth[view][:length]
            local_metrics = _curves(local_predictions[view], target_gt)
            collab_metrics = _curves(collab_predictions[view], target_gt)
            deltas = np.asarray(collab_metrics["overlaps"]) - np.asarray(
                local_metrics["overlaps"])
            rows.append({
                "target": target, "view": view, "frames": length,
                "auc_local": local_metrics["auc"],
                "auc_collab": collab_metrics["auc"],
                "precision_local": local_metrics["precision"],
                "precision_collab": collab_metrics["precision"],
                "norm_precision_local": local_metrics["norm_precision"],
                "norm_precision_collab": collab_metrics["norm_precision"],
                "helpful": int((deltas > self.epsilon).sum()),
                "harmful": int((deltas < -self.epsilon).sum()),
                "tied": int((np.abs(deltas) <= self.epsilon).sum()),
                "runtime_seconds": runtime / 3.0,
                "state_source": "local_candidate_only",
                "teacher_enabled": False,
                "gt_after_initialization_input_count": 0,
                "local_predictions": local_predictions[view],
                "collab_predictions": collab_predictions[view],
            })
        return rows

    def run(self, epoch, max_frames=None):
        was_training = self.model.training
        rng_before = capture_rng_state()
        params_before = state_digest(self.fcvc.state_dict())
        local_rows = []
        self.model.eval()
        self.tracker.eval()
        try:
            with torch.inference_mode():
                for target_index, target in enumerate(self.targets):
                    if target_index % self.world_size == self.rank:
                        local_rows.extend(self._run_target(target, max_frames=max_frames))
        finally:
            restore_rng_state(rng_before)
            self.model.train(was_training)
        gathered = [None] * self.world_size
        if self.dist is not None:
            self.dist.all_gather_object(gathered, local_rows)
        else:
            gathered = [local_rows]
        rows = [row for part in gathered for row in part]
        if not rows:
            raise RuntimeError("online validation produced no rows")
        if self.rank == 0 and self.output_dir is not None:
            epoch_dir = self.output_dir / "epoch_{:02d}".format(int(epoch))
            epoch_dir.mkdir(parents=True, exist_ok=True)
            for row in rows:
                sequence = "{}-{}".format(row["target"], VIEWS[row["view"]])
                np.savetxt(
                    str(epoch_dir / (sequence + "_local.txt")),
                    np.asarray(row["local_predictions"], dtype=float),
                    delimiter=",", fmt="%.6f")
                np.savetxt(
                    str(epoch_dir / (sequence + "_collab.txt")),
                    np.asarray(row["collab_predictions"], dtype=float),
                    delimiter=",", fmt="%.6f")

        def mean(field, selected=rows):
            return float(np.mean([row[field] for row in selected]))

        per_target, per_view = {}, {}
        for target in sorted({row["target"] for row in rows}):
            selected = [row for row in rows if row["target"] == target]
            per_target[target] = {
                field: mean(field, selected) for field in (
                    "auc_local", "auc_collab", "precision_local",
                    "precision_collab", "norm_precision_local",
                    "norm_precision_collab")}
        for view in sorted({row["view"] for row in rows}):
            selected = [row for row in rows if row["view"] == view]
            per_view[view] = {
                field: mean(field, selected) for field in (
                    "auc_local", "auc_collab", "precision_local",
                    "precision_collab", "norm_precision_local",
                    "norm_precision_collab")}
        helpful = sum(row["helpful"] for row in rows)
        harmful = sum(row["harmful"] for row in rows)
        tied = sum(row["tied"] for row in rows)
        cases = helpful + harmful + tied
        metrics = {
            "epoch": int(epoch), "cases": cases,
            "auc_local": mean("auc_local"), "auc_collab": mean("auc_collab"),
            "precision_local": mean("precision_local"),
            "precision_collab": mean("precision_collab"),
            "norm_precision_local": mean("norm_precision_local"),
            "norm_precision_collab": mean("norm_precision_collab"),
            "helpful_rate": helpful / cases, "harmful_rate": harmful / cases,
            "tied_rate": tied / cases, "per_target": per_target,
            "per_view": per_view,
            "runtime": {
                "seconds": sum(row["runtime_seconds"] for row in rows),
                "sequence_count": len(rows),
                "independent_tracker_instances": len(rows),
            },
        }
        for stem in ("auc", "precision", "norm_precision"):
            metrics[stem + "_delta"] = (
                metrics[stem + "_collab"] - metrics[stem + "_local"])
        isolation = {
            "parameters_unchanged": params_before == state_digest(self.fcvc.state_dict()),
            "rng_restored": state_digest(rng_before) == state_digest(capture_rng_state()),
            "teacher_called": False,
            "gt_after_initialization_input_count": 0,
            "state_source": "local_candidate_only",
            "test_dataset_accessed": False,
        }
        if self.rank == 0 and self.output_dir is not None:
            epoch_dir = self.output_dir / "epoch_{:02d}".format(int(epoch))
            (epoch_dir / "summary.json").write_text(
                json.dumps({
                    "metrics": metrics,
                    "isolation": isolation,
                    "teacher_enabled": False,
                    "gt_roi_enabled": False,
                    "test_dataset_accessed": False,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        if self.dist is not None:
            self.dist.barrier()
        return metrics, isolation
