"""Strict collaboration-only freeze and optimizer audits for C3R."""

from typing import Dict, List


def _unwrap(net):
    return net.module if hasattr(net, "module") else net


def c3r_training_enabled(cfg) -> bool:
    return bool(
        getattr(getattr(cfg.MODEL, "C3R", None), "ENABLED", False)
        and getattr(getattr(cfg.TRAIN, "C3R", None), "ENABLED", False)
    )


def apply_c3r_freeze(net, cfg, verbose: bool = False) -> Dict[str, object]:
    if not c3r_training_enabled(cfg):
        return {"trainable_names": [], "trainable_parameters": 0,
                "frozen_parameters": 0}
    if not bool(getattr(cfg.TRAIN.C3R, "FREEZE_LOCAL", True)):
        raise RuntimeError("C3R phase 1 requires TRAIN.C3R.FREEZE_LOCAL=true")
    target = _unwrap(net)
    if getattr(target, "c3r", None) is None:
        raise RuntimeError("C3R training enabled but the model has no c3r module")
    trainable_names: List[str] = []
    frozen_names: List[str] = []
    trainable_parameters = 0
    frozen_parameters = 0
    variant = str(getattr(cfg.MODEL.C3R, "VARIANT", "c1")).lower()
    for name, parameter in target.named_parameters():
        should_train = name.startswith("c3r.")
        if variant in ("c0", "a1") and name.startswith("c3r.reliability."):
            should_train = False
        parameter.requires_grad = should_train
        if should_train:
            trainable_names.append(name)
            trainable_parameters += parameter.numel()
        else:
            frozen_names.append(name)
            frozen_parameters += parameter.numel()
    if not trainable_names:
        raise RuntimeError("C3R strict freeze left no trainable parameters")
    target.c3r_freeze_local = True
    target.train(target.training)
    bad_local = [
        name for name, parameter in target.named_parameters()
        if parameter.requires_grad and not name.startswith("c3r.")
    ]
    if bad_local:
        raise RuntimeError("Non-C3R parameters remain trainable: {}".format(
            bad_local[:20]))
    report = {
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
    }
    if verbose:
        print("C3R strict freeze summary:")
        print("  trainable_parameter_count={}".format(trainable_parameters))
        print("  frozen_local_parameter_count={}".format(frozen_parameters))
        print("  optimizer_parameter_names={}".format(trainable_names))
    return report


def assert_c3r_optimizer_membership(net, optimizer) -> List[str]:
    target = _unwrap(net)
    named = {id(parameter): name for name, parameter in target.named_parameters()}
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("C3R optimizer contains duplicate parameters")
    trainable_ids = {
        id(parameter) for parameter in target.parameters() if parameter.requires_grad
    }
    if set(optimizer_ids) != trainable_ids:
        raise RuntimeError("C3R optimizer does not exactly cover trainable parameters")
    illegal = [named.get(identifier, "<unknown>") for identifier in optimizer_ids
               if not named.get(identifier, "").startswith("c3r.")]
    if illegal:
        raise RuntimeError("C3R optimizer contains non-C3R parameters: {}".format(
            illegal[:20]))
    return [named[identifier] for identifier in optimizer_ids]
