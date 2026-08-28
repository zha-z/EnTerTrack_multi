"""Runtime methods for E3 target-semantic prompt collaboration.

Kept outside the large legacy tracker class so V1 code and checkpoint behavior
remain untouched.  ``install_target_prompt_runtime`` binds three explicit
methods onto the tracker class at import time.
"""

import copy

import torch

from lib.utils.box_ops import clip_box


FP32_BYTES_PER_SENDER = 8 * 192 * 4
FP16_BYTES_PER_SENDER = 8 * 192 * 2


def _scalar(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("E3 diagnostic value must be scalar")
        return float(value.detach().cpu().item())
    return float(value)


def target_prompt_local_candidate(self, image):
    """Run one local branch and construct its prediction-only K=8 packet."""
    if not self.target_prompt_collaboration_enabled:
        raise RuntimeError("E3 local candidate requested while disabled")
    self._target_prompt_local_forward_count = int(getattr(
        self, "_target_prompt_local_forward_count", 0)) + 1
    candidate = self._run_candidate(
        image=image,
        search_factor=self.params.search_factor,
        return_score=True)
    feature = candidate["out_dict"].get("backbone_feat")
    score_map = candidate["out_dict"].get("score_map")
    if not torch.is_tensor(feature) or feature.dim() != 3:
        raise RuntimeError("E3 local candidate is missing [B,L,C] feature")
    if feature.shape[0] != 1 or feature.shape[1] < int(self.network.feat_len_s):
        raise RuntimeError("E3 local feature violates the batch/token contract")
    if not torch.is_tensor(score_map) or tuple(score_map.shape) != (1, 1, 16, 16):
        raise RuntimeError("E3 local candidate requires [1,1,16,16] raw score")
    search = feature[:, -int(self.network.feat_len_s):]
    with torch.no_grad():
        extraction = self.network.target_prompt_extractor.extract_with_metadata(
            search.detach(), score_map.detach())
    topk_scores = extraction["topk_scores"].float()
    packet = {
        "prompt": extraction["prompt"].detach(),
        "topk_score_mean": _scalar(topk_scores.mean()),
        "topk_score_min": _scalar(topk_scores.min()),
        "topk_score_max": _scalar(topk_scores.max()),
        "valid": extraction["valid"].detach(),
        "prompt_k": int(self.network.target_prompt_extractor.prompt_k),
        "source": "local",
        "uses_gt": False,
    }
    candidate["target_prompt_packet"] = packet
    return candidate


def target_prompt_candidate(
        self, local_candidate, remote_candidates, receiver_view,
        sender_views, frame_id, target_id=""):
    """Fuse two synchronized sender-local K=8 packets and rerun only the head."""
    if not self.target_prompt_collaboration_enabled:
        raise RuntimeError("E3 candidate requested while disabled")
    receiver_view = str(receiver_view).upper()
    sender_views = tuple(str(value).upper() for value in sender_views)
    remote_candidates = tuple(remote_candidates)
    if receiver_view not in ("A", "B", "C"):
        raise ValueError("E3 receiver must be A, B, or C")
    expected = tuple(view for view in ("A", "B", "C")
                     if view != receiver_view)
    if sender_views != expected or len(remote_candidates) != 2:
        raise ValueError("E3 requires canonical two-sender order {}".format(
            expected))

    feature = local_candidate["out_dict"].get("backbone_feat")
    if not torch.is_tensor(feature):
        raise RuntimeError("E3 local candidate is missing backbone_feat")
    local_search = feature[:, -int(self.network.feat_len_s):]
    prompts = []
    valid = []
    packets = []
    for sender_view, candidate in zip(sender_views, remote_candidates):
        packet = candidate.get("target_prompt_packet")
        if not isinstance(packet, dict) or packet.get("source") != "local":
            raise RuntimeError(
                "E3 sender {} packet must come from its local branch".format(
                    sender_view))
        prompt = packet.get("prompt")
        if not torch.is_tensor(prompt) or tuple(prompt.shape) != (1, 8, 192):
            raise RuntimeError("E3 sender prompt must have shape [1,8,192]")
        prompts.append(prompt.to(
            device=local_search.device, dtype=local_search.dtype))
        packet_valid = bool(packet["valid"].reshape(-1)[0].item())
        valid.append(packet_valid)
        packets.append(packet)
    remote_prompts = torch.stack(prompts, dim=1)
    remote_valid = torch.tensor(
        [valid], device=local_search.device, dtype=torch.bool)

    state_digest_before = self.fcvc_persistent_state_digest()
    with torch.no_grad():
        head_output = self.network(
            template=None,
            search=None,
            training=False,
            collaboration_feature=feature.detach(),
            target_prompt_remote_tokens=remote_prompts,
            target_prompt_remote_valid=remote_valid)
    state_digest_after = self.fcvc_persistent_state_digest()
    if state_digest_after != state_digest_before:
        raise RuntimeError("E3 head-only forward mutated persistent state")
    collaboration = head_output.get("target_prompt_collaboration")
    if not isinstance(collaboration, dict):
        raise RuntimeError("E3 output is missing collaboration diagnostics")

    pred_box, pred_boxes, max_score, response = self._decode_prediction(
        head_output, local_candidate["resize_factor"], return_score=True)
    image = local_candidate["image"]
    height, width = image.shape[:2]
    target_bbox = clip_box(
        self.map_box_back(
            pred_box,
            local_candidate["resize_factor"],
            reference_bbox=local_candidate["crop_bbox"]),
        height, width, margin=10)
    if self.save_all_boxes:
        all_boxes = self.map_box_back_batch(
            pred_boxes * self.params.search_size
            / local_candidate["resize_factor"],
            local_candidate["resize_factor"],
            reference_bbox=local_candidate["crop_bbox"])
        output = {"target_bbox": target_bbox,
                  "all_boxes": all_boxes.view(-1).tolist()}
    else:
        output = {"target_bbox": target_bbox}

    diagnostics = {
        "frame_id": int(frame_id),
        "target_id": str(target_id),
        "receiver_view": receiver_view,
        "prompt_k": 8,
        "sender_view_0": sender_views[0],
        "sender_view_1": sender_views[1],
        "sender_0_prompt_norm": _scalar(prompts[0].flatten(1).norm(dim=1)),
        "sender_1_prompt_norm": _scalar(prompts[1].flatten(1).norm(dim=1)),
        "residual_norm": _scalar(collaboration["residual_norm"]),
        "relative_residual_norm": _scalar(
            collaboration["relative_residual_norm"]),
        "residual_scale": _scalar(collaboration["residual_scale"]),
        "valid_remote_count": int(
            collaboration["valid_remote_count"].reshape(-1)[0].item()),
        "used_remote": bool(collaboration["used_remote"]),
        "reported_output_source": "target_prompt_collaboration_e3",
        "state_output_source": "local",
        "sender_prompt_source": "local",
        "payload_fp32_bytes_per_sender": FP32_BYTES_PER_SENDER,
        "payload_fp16_bytes_per_sender": FP16_BYTES_PER_SENDER,
        "uses_gt": False,
        "persistent_state_digest_before": state_digest_before,
        "persistent_state_digest_after_collaboration": state_digest_after,
    }
    for slot, packet in enumerate(packets):
        diagnostics.update({
            "sender_{}_topk_score_mean".format(slot): packet["topk_score_mean"],
            "sender_{}_topk_score_min".format(slot): packet["topk_score_min"],
            "sender_{}_topk_score_max".format(slot): packet["topk_score_max"],
        })

    candidate = dict(local_candidate)
    candidate.update({
        "output": output,
        "target_bbox": target_bbox,
        "max_score": max_score,
        "apce": self.calAPCE(response),
        "response": response,
        "out_dict": head_output,
        "pred_boxes": pred_boxes,
        "used_remote": bool(collaboration["used_remote"]),
        "target_prompt_collaboration_diagnostics": diagnostics,
        "_target_prompt_local_candidate": local_candidate,
    })
    return candidate


def target_prompt_finalize_frame(
        self, local_candidate, collaborative_candidate,
        info=None, debug_name=""):
    """Always report E3 and always commit the local candidate state."""
    if not self.target_prompt_collaboration_enabled:
        raise RuntimeError("E3 finalize requested while disabled")
    if not self.target_prompt_collaboration_safe_commit:
        raise RuntimeError("E3 finalize requires SAFE_COMMIT")
    self.frame_id += 1
    self._commit_state_from_candidate(
        local_candidate, info=info, debug_name=debug_name)
    output = copy.deepcopy(collaborative_candidate["output"])
    diagnostics = dict(collaborative_candidate[
        "target_prompt_collaboration_diagnostics"])
    diagnostics["persistent_state_digest_after_commit"] = (
        self.fcvc_persistent_state_digest())
    diagnostics["next_crop_state_digest"] = self.fcvc_next_crop_digest()
    output["target_prompt_collaboration_diagnostics"] = diagnostics
    return (output, collaborative_candidate["max_score"],
            collaborative_candidate["apce"])


def install_target_prompt_runtime(tracker_class):
    tracker_class.target_prompt_local_candidate = target_prompt_local_candidate
    tracker_class.target_prompt_candidate = target_prompt_candidate
    tracker_class.target_prompt_finalize_frame = target_prompt_finalize_frame
