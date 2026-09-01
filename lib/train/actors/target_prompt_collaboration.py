"""Three-MDOT actor helper for E3 target-semantic prompt collaboration."""

import torch

from .plain_collaboration import validate_synchronized_abc_metadata
from lib.train.target_prompt_asymmetric_degradation import (
    apply_e3_d1_asymmetric_degradation)
from lib.train.target_prompt_d2_s1_source_degradation import (
    apply_e3_d2_s1_source_degradation)


def build_flat_remote_prompts(prompts, num_views=3):
    """Map view-major [V*B,K,C] prompts to [V*B,2,K,C]."""
    if prompts.dim() != 3:
        raise ValueError("prompts must have shape [V*B,K,C]")
    total_batch = prompts.shape[0]
    if num_views != 3 or total_batch % num_views != 0:
        raise ValueError("E3 requires a view-major batch divisible by three")
    batch_size = total_batch // num_views
    per_view = [
        prompts[index * batch_size:(index + 1) * batch_size]
        for index in range(num_views)]
    receivers = []
    for receiver in range(num_views):
        senders = [index for index in range(num_views) if index != receiver]
        receivers.append(torch.stack(
            [per_view[index] for index in senders], dim=1))
    return torch.cat(receivers, dim=0)


def build_flat_remote_valid(valid, num_views=3):
    if valid.dim() != 1:
        raise ValueError("valid must have shape [V*B]")
    return build_flat_remote_prompts(
        valid[:, None, None], num_views=num_views)[:, :, 0, 0].bool()


def forward_target_prompt_collaboration(actor, net, data):
    """Run one frozen local ABC pass, extract prompts, then train E3 only."""
    target = net.module if hasattr(net, "module") else net
    adapter = getattr(target, "target_prompt_collaboration", None)
    extractor = getattr(target, "target_prompt_extractor", None)
    if adapter is None or extractor is None:
        raise RuntimeError("E3 actor requires adapter and extractor")
    if any((getattr(target, "plain_collaboration", None) is not None,
            getattr(target, "pcum", None) is not None,
            getattr(target, "c3r", None) is not None,
            getattr(target, "search_prompt_gate", None) is not None)):
        raise RuntimeError("E3 actor forbids V1/PCUM/C3R/search prompt")
    if not actor._get_cfg_value(
            "TRAIN.MULTIVIEW.REQUIRE_ALL_VIEWS_VISIBLE", False):
        raise RuntimeError("E3 requires the common-visible sampler")
    if not actor._get_cfg_value(
            "TRAIN.MULTIVIEW.CANONICAL_VIEW_ORDER", False):
        raise RuntimeError("E3 requires canonical A/B/C view order")

    num_views = min(actor._num_views(data), 3)
    if num_views != 3:
        raise ValueError("E3 requires exactly three synchronized views")
    effective_data, degradation_audit = apply_e3_d1_asymmetric_degradation(
        data, actor.cfg, training=bool(target.training))
    effective_data, source_degradation_audit = \
        apply_e3_d2_s1_source_degradation(
            effective_data, actor.cfg, training=bool(target.training))
    with torch.no_grad():
        local_output = actor._forward_flat_views(
            target, effective_data, num_views, remote_prompts=None,
            model_training=False)
        feature = local_output["backbone_feat"].detach()
        search_tokens = feature[:, -target.feat_len_s:]
        extraction = extractor.extract_with_metadata(
            search_tokens, local_output["score_map"].detach())

    batch_size = search_tokens.shape[0] // num_views
    validate_synchronized_abc_metadata(
        data, num_views=num_views, batch_size=batch_size)
    remote_prompts = build_flat_remote_prompts(
        extraction["prompt"], num_views=num_views)
    remote_valid = build_flat_remote_valid(
        extraction["valid"], num_views=num_views)
    if actor._get_cfg_value(
            "TRAIN.TARGET_PROMPT_COLLABORATION.DETACH_REMOTE", True):
        remote_prompts = remote_prompts.detach()

    output = net(
        template=None,
        search=None,
        training=True,
        collaboration_feature=feature,
        target_prompt_remote_tokens=remote_prompts,
        target_prompt_remote_valid=remote_valid)
    output["pcum_flat_multiview"] = True
    output["num_views"] = num_views
    output["target_prompt_k"] = int(extractor.prompt_k)
    output["e3_d1_degradation"] = degradation_audit
    output["e3_d2_s1_degradation"] = source_degradation_audit
    output["target_prompt_local_output"] = {
        key: value.detach()
        for key, value in local_output.items()
        if key in ("pred_boxes", "score_map", "size_map", "offset_map")}
    return output
