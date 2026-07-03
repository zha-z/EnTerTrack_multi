def build_optimizer_param_groups(net, cfg, verbose=False):
    """Build disjoint head, backbone, and PCUM parameter groups."""
    named_trainable = [
        (name, parameter) for name, parameter in net.named_parameters()
        if parameter.requires_grad
    ]

    def canonical_name(name):
        return name[len("module."):] if name.startswith("module.") else name

    pcum_named = [
        (name, parameter) for name, parameter in named_trainable
        if canonical_name(name).startswith("pcum.")
    ]
    backbone_named = [
        (name, parameter) for name, parameter in named_trainable
        if canonical_name(name).startswith("backbone.")
    ]
    other_named = [
        (name, parameter) for name, parameter in named_trainable
        if not canonical_name(name).startswith(("backbone.", "pcum."))
    ]

    group_specs = [
        ("head_and_other", other_named, cfg.TRAIN.LR * 0.1),
        ("backbone", backbone_named,
         cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER),
        ("pcum", pcum_named,
         getattr(cfg.TRAIN, "PCUM_LR", cfg.TRAIN.LR * 0.1)),
    ]
    groups = [
        {
            "params": [parameter for _, parameter in named],
            "lr": float(lr),
            "group_name": group_name,
        }
        for group_name, named, lr in group_specs if named
    ]

    grouped_ids = [
        id(parameter) for group in groups for parameter in group["params"]
    ]
    trainable_ids = [id(parameter) for _, parameter in named_trainable]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("A parameter appears in more than one optimizer group")
    if set(grouped_ids) != set(trainable_ids):
        raise RuntimeError("Optimizer groups do not cover all trainable parameters")

    if verbose:
        print("Optimizer parameter groups:")
        for group_name, named, lr in group_specs:
            if not named:
                continue
            parameter_count = sum(parameter.numel() for _, parameter in named)
            categories = sorted(set(
                canonical_name(name).split(".", 1)[0] for name, _ in named
            ))
            examples = [name for name, _ in named[:5]]
            print(
                "  %s: tensors=%d parameters=%d lr=%.8g categories=%s examples=%s"
                % (group_name, len(named), parameter_count, lr, categories, examples)
            )
    return groups
