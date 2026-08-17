import torch.nn as nn


def _unwrap_model(net):
    return net.module if hasattr(net, "module") else net


def _cfg_bool(cfg, path, default=False):
    node = cfg
    for part in path.split("."):
        if not hasattr(node, part):
            return default
        node = getattr(node, part)
    return bool(node)


def _canonical_name(name):
    return name[len("module."):] if name.startswith("module.") else name


def _is_backbone_name(name):
    return _canonical_name(name).startswith("backbone.")


def _is_pcum_name(name):
    return _canonical_name(name).startswith("pcum.")


def _is_head_or_other_name(name):
    canonical = _canonical_name(name)
    return not canonical.startswith(("backbone.", "pcum."))


def pcum_ranking_strict_freeze_enabled(cfg):
    return (
        _cfg_bool(cfg, "TRAIN.PCUM_RANKING.FREEZE_BACKBONE", False)
        or _cfg_bool(cfg, "TRAIN.PCUM_RANKING.FREEZE_HEAD", False)
    )


def pcum_remote_suppression_only_enabled(cfg):
    return _cfg_bool(cfg, "TRAIN.PCUM_RANKING.REMOTE_SUPPRESSION_ONLY", False)


def partial_adaptation_enabled(cfg):
    return _cfg_bool(cfg, "TRAIN.PARTIAL_ADAPTATION.ENABLED", False)


def _partial_backbone_block_indices(cfg):
    node = cfg
    for part in "TRAIN.PARTIAL_ADAPTATION.BACKBONE_BLOCKS".split("."):
        if not hasattr(node, part):
            return []
        node = getattr(node, part)
    return [int(index) for index in node]


def _is_partial_backbone_trainable_name(canonical, cfg):
    if not canonical.startswith("backbone."):
        return False
    for block_index in _partial_backbone_block_indices(cfg):
        if canonical.startswith("backbone.blocks.%d." % block_index):
            return True
    if _cfg_bool(cfg, "TRAIN.PARTIAL_ADAPTATION.TRAIN_FINAL_NORM", False):
        if canonical.startswith("backbone.norm."):
            return True
    return False


def _is_partial_head_trainable_name(canonical, cfg):
    if not _cfg_bool(cfg, "TRAIN.PARTIAL_ADAPTATION.TRAIN_HEAD", False):
        return False
    return not canonical.startswith(("backbone.", "pcum."))


def _is_partial_pcum_trainable_name(canonical, cfg):
    if not _cfg_bool(cfg, "MODEL.PCUM.ENABLED", False):
        return False
    return canonical.startswith("pcum.")


def apply_partial_adaptation_freeze(net, cfg, verbose=False):
    """Freeze everything except the declared paired-adaptation parameter sets."""
    target = _unwrap_model(net)
    trainable_names = []
    frozen_names = []
    illegal_trainable = []
    counts = {
        "trainable": 0,
        "last_backbone": 0,
        "head": 0,
        "pcum": 0,
        "frozen": 0,
    }

    for name, parameter in target.named_parameters():
        canonical = _canonical_name(name)
        should_train_backbone = _is_partial_backbone_trainable_name(canonical, cfg)
        should_train_head = _is_partial_head_trainable_name(canonical, cfg)
        should_train_pcum = _is_partial_pcum_trainable_name(canonical, cfg)
        should_train = should_train_backbone or should_train_head or should_train_pcum
        parameter.requires_grad = bool(should_train)

        if should_train:
            trainable_names.append(canonical)
            counts["trainable"] += parameter.numel()
            if should_train_backbone:
                counts["last_backbone"] += parameter.numel()
            elif should_train_head:
                counts["head"] += parameter.numel()
            elif should_train_pcum:
                counts["pcum"] += parameter.numel()
        else:
            frozen_names.append(canonical)
            counts["frozen"] += parameter.numel()

    for name, parameter in target.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical = _canonical_name(name)
        allowed = (
            _is_partial_backbone_trainable_name(canonical, cfg)
            or _is_partial_head_trainable_name(canonical, cfg)
            or _is_partial_pcum_trainable_name(canonical, cfg)
        )
        if not allowed:
            illegal_trainable.append(canonical)

    if illegal_trainable:
        raise RuntimeError(
            "Partial adaptation freeze violation: illegal trainable parameters: %s"
            % ", ".join(illegal_trainable[:20])
        )
    if counts["last_backbone"] <= 0:
        raise RuntimeError("Partial adaptation left no trainable last-backbone parameters")
    if _cfg_bool(cfg, "TRAIN.PARTIAL_ADAPTATION.TRAIN_HEAD", False) and counts["head"] <= 0:
        raise RuntimeError("Partial adaptation left no trainable head parameters")
    if _cfg_bool(cfg, "MODEL.PCUM.ENABLED", False) and counts["pcum"] <= 0:
        raise RuntimeError("Partial adaptation left no trainable PCUM parameters")

    if verbose:
        print("Partial adaptation freeze summary:")
        print("  TRAIN.PARTIAL_ADAPTATION.BACKBONE_BLOCKS=%s" % _partial_backbone_block_indices(cfg))
        print("  TRAIN.PARTIAL_ADAPTATION.TRAIN_FINAL_NORM=%s" % _cfg_bool(cfg, "TRAIN.PARTIAL_ADAPTATION.TRAIN_FINAL_NORM", False))
        print("  TRAIN.PARTIAL_ADAPTATION.TRAIN_HEAD=%s" % _cfg_bool(cfg, "TRAIN.PARTIAL_ADAPTATION.TRAIN_HEAD", False))
        print("  trainable_parameter_count=%d" % counts["trainable"])
        print("  last_backbone_parameter_count=%d" % counts["last_backbone"])
        print("  head_parameter_count=%d" % counts["head"])
        print("  pcum_parameter_count=%d" % counts["pcum"])
        print("  optimizer_parameter_names=%s" % trainable_names)

    return {
        "counts": counts,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }


def _is_remote_suppression_gate_name(name):
    return _canonical_name(name).startswith("pcum.remote_suppression_gate.")


def apply_pcum_ranking_freeze(net, cfg, verbose=False):
    """Apply strict PCUM-only freeze rules before optimizer construction."""
    target = _unwrap_model(net)
    freeze_backbone = _cfg_bool(
        cfg, "TRAIN.PCUM_RANKING.FREEZE_BACKBONE",
        _cfg_bool(cfg, "TRAIN.FREEZE_BACKBONE", False),
    )
    freeze_head = _cfg_bool(
        cfg, "TRAIN.PCUM_RANKING.FREEZE_HEAD",
        _cfg_bool(cfg, "TRAIN.FREEZE_HEAD", False),
    )
    remote_suppression_only = pcum_remote_suppression_only_enabled(cfg)

    counts = {
        "trainable": 0,
        "frozen_backbone": 0,
        "frozen_head": 0,
        "pcum_trainable": 0,
    }
    trainable_names = []
    frozen_backbone_names = []
    frozen_head_names = []
    illegal_trainable = []

    for name, parameter in target.named_parameters():
        canonical = _canonical_name(name)
        if freeze_backbone and _is_backbone_name(canonical):
            parameter.requires_grad = False
            frozen_backbone_names.append(canonical)
        elif freeze_head and _is_head_or_other_name(canonical):
            parameter.requires_grad = False
            frozen_head_names.append(canonical)
        elif remote_suppression_only and _is_pcum_name(canonical):
            parameter.requires_grad = _is_remote_suppression_gate_name(canonical)
        elif _is_pcum_name(canonical):
            parameter.requires_grad = True

        if parameter.requires_grad:
            trainable_names.append(canonical)
            counts["trainable"] += parameter.numel()
            if _is_pcum_name(canonical):
                counts["pcum_trainable"] += parameter.numel()
            if freeze_backbone and _is_backbone_name(canonical):
                illegal_trainable.append(canonical)
            if freeze_head and _is_head_or_other_name(canonical):
                illegal_trainable.append(canonical)
            if remote_suppression_only and not _is_remote_suppression_gate_name(canonical):
                illegal_trainable.append(canonical)
        else:
            if _is_backbone_name(canonical):
                counts["frozen_backbone"] += parameter.numel()
            elif _is_head_or_other_name(canonical):
                counts["frozen_head"] += parameter.numel()

    if illegal_trainable:
        examples = ", ".join(illegal_trainable[:20])
        raise RuntimeError(
            "Strict PCUM freeze violation: illegal trainable parameters: %s"
            % examples
        )
    if (freeze_backbone or freeze_head) and counts["pcum_trainable"] <= 0:
        raise RuntimeError("Strict PCUM freeze left no trainable PCUM parameters")

    set_pcum_frozen_modules_eval(target, cfg)
    assert_pcum_frozen_batchnorm_eval(target, cfg)

    if verbose:
        print("PCUM ranking strict freeze summary:")
        print("  TRAIN.PCUM_RANKING.FREEZE_BACKBONE=%s" % freeze_backbone)
        print("  TRAIN.PCUM_RANKING.FREEZE_HEAD=%s" % freeze_head)
        print("  TRAIN.PCUM_RANKING.REMOTE_SUPPRESSION_ONLY=%s" % remote_suppression_only)
        print("  trainable_parameter_count=%d" % counts["trainable"])
        print("  frozen_backbone_count=%d" % counts["frozen_backbone"])
        print("  frozen_head_count=%d" % counts["frozen_head"])
        print("  pcum_trainable_parameter_count=%d" % counts["pcum_trainable"])
        print("  optimizer_parameter_names=%s" % trainable_names)
        if frozen_backbone_names:
            print("  frozen_backbone_examples=%s" % frozen_backbone_names[:10])
        if frozen_head_names:
            print("  frozen_head_examples=%s" % frozen_head_names[:10])

    return {
        "counts": counts,
        "trainable_names": trainable_names,
        "frozen_backbone_names": frozen_backbone_names,
        "frozen_head_names": frozen_head_names,
    }


def _module_should_be_eval(module_name, cfg):
    freeze_backbone = _cfg_bool(
        cfg, "TRAIN.PCUM_RANKING.FREEZE_BACKBONE",
        _cfg_bool(cfg, "TRAIN.FREEZE_BACKBONE", False),
    )
    freeze_head = _cfg_bool(
        cfg, "TRAIN.PCUM_RANKING.FREEZE_HEAD",
        _cfg_bool(cfg, "TRAIN.FREEZE_HEAD", False),
    )
    canonical = _canonical_name(module_name)
    if canonical == "":
        return False
    if freeze_backbone and (canonical == "backbone" or canonical.startswith("backbone.")):
        return True
    if freeze_head and not (canonical == "pcum" or canonical.startswith("pcum.")):
        return True
    return False


def set_pcum_frozen_modules_eval(net, cfg):
    target = _unwrap_model(net)
    for module_name, module in target.named_modules():
        if _module_should_be_eval(module_name, cfg):
            module.eval()


def assert_pcum_frozen_batchnorm_eval(net, cfg):
    target = _unwrap_model(net)
    bad = []
    for module_name, module in target.named_modules():
        if not _module_should_be_eval(module_name, cfg):
            continue
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            bad.append(module_name)
    if bad:
        raise RuntimeError(
            "Frozen BatchNorm modules are still in train mode: %s"
            % ", ".join(bad[:20])
        )


def assert_optimizer_has_only_pcum_params(net, cfg):
    if partial_adaptation_enabled(cfg):
        return
    target = _unwrap_model(net)
    freeze_backbone = _cfg_bool(cfg, "TRAIN.PCUM_RANKING.FREEZE_BACKBONE", False)
    freeze_head = _cfg_bool(cfg, "TRAIN.PCUM_RANKING.FREEZE_HEAD", False)
    remote_suppression_only = pcum_remote_suppression_only_enabled(cfg)
    if not (freeze_backbone or freeze_head or remote_suppression_only):
        return
    illegal = []
    for name, parameter in target.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical = _canonical_name(name)
        if freeze_backbone and _is_backbone_name(canonical):
            illegal.append(canonical)
        if freeze_head and _is_head_or_other_name(canonical):
            illegal.append(canonical)
        if remote_suppression_only and not _is_remote_suppression_gate_name(canonical):
            illegal.append(canonical)
    if illegal:
        raise RuntimeError(
            "Optimizer would include frozen parameters: %s"
            % ", ".join(illegal[:20])
        )
