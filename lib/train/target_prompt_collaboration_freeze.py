"""Strict adapter-only freeze and optimizer audit for E3."""


def _unwrap(net):
    return net.module if hasattr(net, "module") else net


def target_prompt_collaboration_training_enabled(cfg):
    return bool(
        getattr(getattr(
            cfg.MODEL, "TARGET_PROMPT_COLLABORATION", None),
            "ENABLED", False)
        and getattr(getattr(
            cfg.TRAIN, "TARGET_PROMPT_COLLABORATION", None),
            "ENABLED", False))


def apply_target_prompt_collaboration_freeze(net, cfg, verbose=False):
    if not target_prompt_collaboration_training_enabled(cfg):
        return {"trainable_names": [], "trainable_parameters": 0,
                "frozen_parameters": 0}
    if not bool(getattr(
            cfg.TRAIN.TARGET_PROMPT_COLLABORATION,
            "FREEZE_LOCAL", True)):
        raise RuntimeError("E3 requires FREEZE_LOCAL=true")
    target = _unwrap(net)
    if target.target_prompt_collaboration is None:
        raise RuntimeError("E3 training is enabled without the adapter")
    if target.target_prompt_extractor is None:
        raise RuntimeError("E3 training is enabled without the extractor")
    if sum(parameter.numel()
           for parameter in target.target_prompt_extractor.parameters()) != 0:
        raise RuntimeError("TargetPromptExtractor must have zero parameters")

    trainable_names = []
    frozen_names = []
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in target.named_parameters():
        should_train = name.startswith("target_prompt_collaboration.")
        parameter.requires_grad = should_train
        if should_train:
            trainable_names.append(name)
            trainable_parameters += parameter.numel()
        else:
            frozen_names.append(name)
            frozen_parameters += parameter.numel()
    if not trainable_names:
        raise RuntimeError("E3 freeze left no trainable parameters")
    target.target_prompt_collaboration_freeze_local = True
    target.train(target.training)
    report = {
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "extractor_parameters": 0,
    }
    if verbose:
        print("Target Prompt E3 strict freeze summary:")
        print("  trainable_parameter_count={}".format(trainable_parameters))
        print("  frozen_local_parameter_count={}".format(frozen_parameters))
        print("  extractor_parameter_count=0")
        print("  optimizer_parameter_names={}".format(trainable_names))
    return report


def assert_target_prompt_optimizer_membership(net, optimizer):
    target = _unwrap(net)
    names_by_id = {
        id(parameter): name for name, parameter in target.named_parameters()}
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("E3 optimizer contains duplicate parameters")
    trainable_ids = {
        id(parameter) for parameter in target.parameters()
        if parameter.requires_grad}
    if set(optimizer_ids) != trainable_ids:
        raise RuntimeError("E3 optimizer does not exactly cover trainable parameters")
    illegal = [
        names_by_id.get(identifier, "<unknown>")
        for identifier in optimizer_ids
        if not names_by_id.get(identifier, "").startswith(
            "target_prompt_collaboration.")]
    if illegal:
        raise RuntimeError("E3 optimizer contains non-adapter parameters: {}".format(
            illegal[:20]))
    return [names_by_id[identifier] for identifier in optimizer_ids]
