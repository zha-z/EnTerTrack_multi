"""Three-MDOT actor helpers for Plain Collaboration V1."""

import torch


def _metadata_item(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Expected scalar collaboration metadata")
        return value.detach().cpu().item()
    return value


def validate_synchronized_abc_metadata(data, num_views, batch_size):
    """Fail closed when available view/frame metadata is not synchronized."""
    if num_views != 3:
        raise ValueError("Plain Collaboration V1 requires exactly three views")

    view_ids = data.get("view_ids", None)
    if view_ids is not None:
        if len(view_ids) < num_views:
            raise ValueError("view_ids does not contain three views")
        for view_index, expected in enumerate(("A", "B", "C")):
            if len(view_ids[view_index]) != batch_size:
                raise ValueError("view_ids batch size mismatch")
            actual = {
                str(_metadata_item(value)).upper()
                for value in view_ids[view_index]
            }
            if actual != {expected}:
                raise ValueError(
                    "Non-canonical view order: expected {}, got {}".format(
                        expected, sorted(actual)))

    frame_ids = data.get("search_frame_ids", None)
    if frame_ids is not None:
        frame_slots = len(frame_ids)
        if frame_slots not in (1, num_views):
            raise ValueError(
                "search_frame_ids must contain one shared frame slot or "
                "one slot per view")
        for frame_slot in range(frame_slots):
            if len(frame_ids[frame_slot]) != batch_size:
                raise ValueError("search_frame_ids batch size mismatch")

        # The common-visible sampler stores one shared frame id per target;
        # LTRLoader(stack_dim=1) therefore collates it as [1, B].  Independent
        # sampling stores [V, B], in which case synchronization must be checked
        # explicitly before remote features are exchanged.
        if frame_slots == num_views:
            for batch_index in range(batch_size):
                synchronized = {
                    int(_metadata_item(frame_ids[view][batch_index]))
                    for view in range(num_views)
                }
                if len(synchronized) != 1:
                    raise ValueError(
                        "Remote views use different search_frame_ids at batch {}"
                        .format(batch_index))

    target_ids = data.get("target_id", None)
    if target_ids is not None and len(target_ids) != batch_size:
        raise ValueError("target_id batch size mismatch")


def build_flat_remote_tokens(search_tokens, num_views=3):
    """Map view-major [V*B,L,C] into two sender slots per receiver."""
    if search_tokens.dim() != 3:
        raise ValueError("search_tokens must have shape [V*B,L,C]")
    total_batch = search_tokens.shape[0]
    if num_views != 3 or total_batch % num_views != 0:
        raise ValueError("Expected a view-major batch divisible by three")
    batch_size = total_batch // num_views
    per_view = [
        search_tokens[view * batch_size:(view + 1) * batch_size]
        for view in range(num_views)
    ]
    receiver_remotes = []
    for receiver in range(num_views):
        senders = [view for view in range(num_views) if view != receiver]
        receiver_remotes.append(torch.stack(
            [per_view[sender] for sender in senders], dim=1))
    return torch.cat(receiver_remotes, dim=0)


def forward_plain_collaboration(actor, net, data):
    """Run one frozen local pass and one DDP-visible adapter/head pass."""
    target = net.module if hasattr(net, "module") else net
    adapter = getattr(target, "plain_collaboration", None)
    if adapter is None:
        raise RuntimeError("V1 actor path requires model.plain_collaboration")
    if getattr(target, "pcum", None) is not None:
        raise RuntimeError("V1 actor path forbids PCUM")
    if getattr(target, "c3r", None) is not None:
        raise RuntimeError("V1 actor path forbids C3R")
    if getattr(target, "search_prompt_gate", None) is not None:
        raise RuntimeError("V1 actor path forbids the search prompt gate")
    if not actor._get_cfg_value(
            "TRAIN.MULTIVIEW.REQUIRE_ALL_VIEWS_VISIBLE", False):
        raise RuntimeError("V1 requires the common-visible sampler")
    if not actor._get_cfg_value(
            "TRAIN.MULTIVIEW.CANONICAL_VIEW_ORDER", False):
        raise RuntimeError("V1 requires canonical A/B/C view order")

    num_views = min(actor._num_views(data), 3)
    if num_views != 3:
        raise ValueError("V1 requires exactly three synchronized views")

    # The local core is frozen and deliberately bypasses DDP.  The following
    # head-only call goes through ``net`` so adapter gradients are synchronized.
    with torch.no_grad():
        local_output = actor._forward_flat_views(
            target, data, num_views, remote_prompts=None,
            model_training=False)
    feature = local_output["backbone_feat"].detach()
    search_tokens = feature[:, -target.feat_len_s:]
    batch_size = search_tokens.shape[0] // num_views
    validate_synchronized_abc_metadata(
        data, num_views=num_views, batch_size=batch_size)
    remote_tokens = build_flat_remote_tokens(
        search_tokens, num_views=num_views)
    if actor._get_cfg_value(
            "TRAIN.PLAIN_COLLABORATION.DETACH_REMOTE", True):
        remote_tokens = remote_tokens.detach()
    remote_valid = torch.ones(
        remote_tokens.shape[:2], device=remote_tokens.device,
        dtype=torch.bool)

    output = net(
        template=None,
        search=None,
        training=True,
        collaboration_feature=feature,
        plain_remote_tokens=remote_tokens,
        plain_remote_valid=remote_valid,
    )
    output["pcum_flat_multiview"] = True
    output["num_views"] = num_views
    output["plain_collaboration_local_output"] = {
        key: value.detach()
        for key, value in local_output.items()
        if key in ("pred_boxes", "score_map", "size_map", "offset_map")
    }
    return output
