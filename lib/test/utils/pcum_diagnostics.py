import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F


DIAGNOSTIC_COLUMNS = [
    "diagnostic_label",
    "uses_gt_visible_mask",
    "sequence_name",
    "frame_id",
    "current_uav",
    "remote_uav_ids",
    "remote_uav_count",
    "local_bbox",
    "raw_collaborative_bbox",
    "final_bbox",
    "gt_bbox",
    "local_iou",
    "raw_collaborative_iou",
    "final_iou",
    "instant_delta_iou",
    "fallback_delta_iou",
    "final_delta_iou",
    "local_confidence",
    "local_score_max",
    "local_apce",
    "local_response_entropy",
    "local_bbox_motion_distance",
    "raw_collaborative_score_max",
    "raw_collaborative_apce",
    "raw_collaborative_response_entropy",
    "remote_confidences",
    "prompt_similarities",
    "remote_visibility_gt",
    "remote_participated",
    "alignment_gate_mean",
    "alignment_gate_std",
    "fusion_gate_mean",
    "fusion_gate_std",
    "fusion_gate_min",
    "fusion_gate_max",
    "prompt_norm",
    "aligned_prompt_norm",
    "final_source",
    "fallback_triggered",
    "fallback_reason",
]


def diagnostic_filename(tracker_name, parameter_name, run_id, sequence_name, uav_id):
    def clean(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))

    run_label = "none" if run_id is None else str(run_id)
    return "{}__{}__run{}__{}__uav-{}__pcum_frame_diagnostics.csv".format(
        clean(tracker_name),
        clean(parameter_name),
        clean(run_label),
        clean(sequence_name),
        clean(uav_id),
    )


def visibility_for_remote_selection(
    diagnostics_enabled,
    use_gt_visible_mask,
    sequences,
    frame_id,
):
    if not diagnostics_enabled or not use_gt_visible_mask:
        return None
    return [bool(sequence.target_visible[frame_id]) for sequence in sequences]


def _json_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def json_value(value):
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def as_float(value, default=float("nan")):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        value = value.detach().reshape(-1)[0].cpu().item()
    return float(value)


def bbox_iou_xywh(first, second, eps=1e-12):
    if first is None or second is None:
        return float("nan")
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 4 or second.size < 4:
        return float("nan")
    if not np.isfinite(first[:4]).all() or not np.isfinite(second[:4]).all():
        return float("nan")
    if first[2] <= 0 or first[3] <= 0 or second[2] <= 0 or second[3] <= 0:
        return float("nan")

    top_left = np.maximum(first[:2], second[:2])
    bottom_right = np.minimum(first[:2] + first[2:4], second[:2] + second[2:4])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = float(size[0] * size[1])
    union = float(first[2] * first[3] + second[2] * second[3] - intersection)
    return intersection / max(union, eps)


def bbox_center_distance(first, second):
    if first is None or second is None:
        return float("nan")
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 4 or second.size < 4:
        return float("nan")
    first_center = first[:2] + 0.5 * first[2:4]
    second_center = second[:2] + 0.5 * second[2:4]
    return float(np.linalg.norm(second_center - first_center))


def normalized_response_entropy(response, eps=1e-12):
    if response is None:
        return float("nan")
    values = response.detach().float().reshape(-1).clamp_min(0.0)
    if values.numel() <= 1:
        return 0.0
    probabilities = values + float(eps)
    probabilities = probabilities / probabilities.sum().clamp_min(float(eps))
    entropy = -(probabilities * probabilities.log()).sum()
    return float((entropy / math.log(values.numel())).cpu().item())


def tensor_stats(value):
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    values = value.detach().float().reshape(-1)
    return {
        "mean": float(values.mean().cpu().item()),
        "std": float(values.std(unbiased=False).cpu().item()),
        "min": float(values.min().cpu().item()),
        "max": float(values.max().cpu().item()),
    }


def prompt_norm(prompt):
    if prompt is None or not torch.is_tensor(prompt) or prompt.numel() == 0:
        return float("nan")
    values = prompt.detach().float()
    return float(values.norm(dim=-1).mean().cpu().item())


def prompt_cosine_similarity(local_prompt, remote_prompt, aligner=None):
    if local_prompt is None or remote_prompt is None:
        return float("nan")
    with torch.no_grad():
        local = local_prompt.detach()
        remote = remote_prompt.detach().to(device=local.device, dtype=local.dtype)
        if aligner is not None:
            local = aligner.local_norm(local)
            remote = aligner.remote_norm(remote)
        if remote.shape[1] != local.shape[1]:
            remote = F.interpolate(
                remote.transpose(1, 2),
                size=local.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        similarity = F.cosine_similarity(local, remote, dim=-1, eps=1e-6).mean()
        return float(similarity.detach().cpu().item())


def build_frame_diagnostic_row(
    diagnostic_label,
    uses_gt_visible_mask,
    sequence_name,
    frame_id,
    current_uav,
    remote_uav_ids,
    local,
    raw_collaborative,
    final,
    gt_bbox,
    previous_bbox=None,
    remote_confidences=None,
    prompt_similarities=None,
    remote_visibility_gt=None,
    remote_participated=None,
    final_source="unknown",
    fallback_triggered=False,
    fallback_reason="none",
):
    local = local or {}
    raw_collaborative = raw_collaborative or local
    final = final or raw_collaborative
    local_bbox = local.get("bbox", None)
    raw_bbox = raw_collaborative.get("bbox", local_bbox)
    final_bbox = final.get("bbox", raw_bbox)

    local_iou = bbox_iou_xywh(local_bbox, gt_bbox)
    raw_iou = bbox_iou_xywh(raw_bbox, gt_bbox)
    final_iou = bbox_iou_xywh(final_bbox, gt_bbox)
    alignment = raw_collaborative

    return {
        "diagnostic_label": diagnostic_label,
        "uses_gt_visible_mask": int(bool(uses_gt_visible_mask)),
        "sequence_name": sequence_name,
        "frame_id": int(frame_id),
        "current_uav": current_uav,
        "remote_uav_ids": "|".join(remote_uav_ids),
        "remote_uav_count": len(remote_uav_ids),
        "local_bbox": json_value(local_bbox),
        "raw_collaborative_bbox": json_value(raw_bbox),
        "final_bbox": json_value(final_bbox),
        "gt_bbox": json_value(list(gt_bbox) if gt_bbox is not None else None),
        "local_iou": local_iou,
        "raw_collaborative_iou": raw_iou,
        "final_iou": final_iou,
        "instant_delta_iou": raw_iou - local_iou,
        "fallback_delta_iou": final_iou - raw_iou,
        "final_delta_iou": final_iou - local_iou,
        "local_confidence": local.get("confidence", float("nan")),
        "local_score_max": local.get("score_max", float("nan")),
        "local_apce": local.get("apce", float("nan")),
        "local_response_entropy": local.get("response_entropy", float("nan")),
        "local_bbox_motion_distance": bbox_center_distance(previous_bbox, local_bbox),
        "raw_collaborative_score_max": raw_collaborative.get("score_max", float("nan")),
        "raw_collaborative_apce": raw_collaborative.get("apce", float("nan")),
        "raw_collaborative_response_entropy": raw_collaborative.get(
            "response_entropy", float("nan")
        ),
        "remote_confidences": json_value(remote_confidences or {}),
        "prompt_similarities": json_value(prompt_similarities or {}),
        "remote_visibility_gt": json_value(remote_visibility_gt or {}),
        "remote_participated": json_value(remote_participated or {}),
        "alignment_gate_mean": alignment.get("alignment_gate_mean", float("nan")),
        "alignment_gate_std": alignment.get("alignment_gate_std", float("nan")),
        "fusion_gate_mean": alignment.get("fusion_gate_mean", float("nan")),
        "fusion_gate_std": alignment.get("fusion_gate_std", float("nan")),
        "fusion_gate_min": alignment.get("fusion_gate_min", float("nan")),
        "fusion_gate_max": alignment.get("fusion_gate_max", float("nan")),
        "prompt_norm": alignment.get("prompt_norm", float("nan")),
        "aligned_prompt_norm": alignment.get("aligned_prompt_norm", float("nan")),
        "final_source": final_source,
        "fallback_triggered": int(bool(fallback_triggered)),
        "fallback_reason": fallback_reason,
    }


class PCUMDiagnosticHooks:
    """Capture PCUM scalar telemetry without changing module outputs."""

    def __init__(self, network, enabled=False):
        self.enabled = bool(enabled)
        self._handles = []
        self.reset()
        if not self.enabled:
            return

        pcum = getattr(network, "pcum", None)
        if pcum is None:
            return
        self._handles.append(pcum.aligner.register_forward_hook(self._alignment_hook))
        if getattr(pcum.fusion, "mode", None) == "gated_add":
            self._handles.append(pcum.fusion.gate.register_forward_hook(self._fusion_gate_hook))

    def reset(self):
        self.alignment = tensor_stats(None)
        self.fusion = tensor_stats(None)

    def _alignment_hook(self, module, inputs, output):
        gate = output.get("gate", None) if isinstance(output, dict) else None
        self.alignment = tensor_stats(gate)

    def _fusion_gate_hook(self, module, inputs, output):
        self.fusion = tensor_stats(torch.sigmoid(output.detach()))

    def snapshot(self):
        return {
            "alignment_gate_mean": self.alignment["mean"],
            "alignment_gate_std": self.alignment["std"],
            "fusion_gate_mean": self.fusion["mean"],
            "fusion_gate_std": self.fusion["std"],
            "fusion_gate_min": self.fusion["min"],
            "fusion_gate_max": self.fusion["max"],
        }

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def handle_count(self):
        return len(self._handles)
