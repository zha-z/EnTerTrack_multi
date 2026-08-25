"""Strict adapter-only freeze and optimizer audit for Plain Collaboration V1."""


def _unwrap(net):
    return net.module if hasattr(net, "module") else net


def plain_collaboration_training_enabled(cfg):
    return bool(
        getattr(getattr(cfg.MODEL, "PLAIN_COLLABORATION", None), "ENABLED", False)
        and getattr(getattr(cfg.TRAIN, "PLAIN_COLLABORATION", None), "ENABLED", False)
    )


def apply_plain_collaboration_freeze(net, cfg, verbose=False):
    if not plain_collaboration_training_enabled(cfg):
        return {"trainable_names": [], "trainable_parameters": 0,
                "frozen_parameters": 0}
    if not bool(getattr(cfg.TRAIN.PLAIN_COLLABORATION, "FREEZE_LOCAL", True)):
        raise RuntimeError(
            "Plain Collaboration V1 requires FREEZE_LOCAL=true")
    target = _unwrap(net)
    if getattr(target, "plain_collaboration", None) is None:
        raise RuntimeError(
            "Plain collaboration training enabled but the model has no adapter")

    trainable_names = []
    frozen_names = []
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in target.named_parameters():
        should_train = name.startswith("plain_collaboration.")
        parameter.requires_grad = should_train
        if should_train:
            trainable_names.append(name)
            trainable_parameters += parameter.numel()
        else:
            frozen_names.append(name)
            frozen_parameters += parameter.numel()
    if not trainable_names:
        raise RuntimeError("Plain collaboration freeze left no trainable parameters")
    target.plain_collaboration_freeze_local = True
    target.train(target.training)
    report = {
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
    }
    if verbose:
        print("Plain Collaboration V1 strict freeze summary:")
        print("  trainable_parameter_count={}".format(trainable_parameters))
        print("  frozen_local_parameter_count={}".format(frozen_parameters))
        print("  optimizer_parameter_names={}".format(trainable_names))
    return report


def assert_plain_collaboration_optimizer_membership(net, optimizer):
    target = _unwrap(net)
    names_by_id = {
        id(parameter): name for name, parameter in target.named_parameters()
    }
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError(
            "Plain collaboration optimizer contains duplicate parameters")
    trainable_ids = {
        id(parameter) for parameter in target.parameters()
        if parameter.requires_grad
    }
    if set(optimizer_ids) != trainable_ids:
        raise RuntimeError(
            "Plain collaboration optimizer does not exactly cover trainable parameters")
    illegal = [
        names_by_id.get(identifier, "<unknown>")
        for identifier in optimizer_ids
        if not names_by_id.get(identifier, "").startswith(
            "plain_collaboration.")
    ]
    if illegal:
        raise RuntimeError(
            "Optimizer contains non-collaboration parameters: {}".format(
                illegal[:20]))
    return [names_by_id[identifier] for identifier in optimizer_ids]
