import os

# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
import torch
import torch.nn as nn

# train pipeline related
from lib.train.trainers import LTRTrainer

# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP

# base functions
from .base_functions import *

# models
#from lib.models.ostrack import build_ostrack, build_ostrack_prompt
from lib.models.entertrack import build_entertrack

# actors
#from lib.train.actors import OSTrackActor, OSTrackActorPrompt
from lib.train.actors import EnTeRTrackActorThreeMDOT, EnTeRTrackActorTeacher

# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def use_grouped_multiview_loader(cfg):
    """Return whether training must preserve synchronized multi-view groups.

    TRAIN.MULTIVIEW.ENABLED is the method-neutral switch used by paired
    controls. The PCUM flag remains as a compatibility fallback so existing
    collaborative configurations resolve to the same loader as before.
    """
    train_cfg = getattr(cfg, "TRAIN", None)
    multiview_cfg = getattr(train_cfg, "MULTIVIEW", None)
    pcum_cfg = getattr(train_cfg, "PCUM", None)
    return bool(
        getattr(multiview_cfg, "ENABLED", False)
        or getattr(pcum_cfg, "USE_REAL_MULTIVIEW", False)
    )


def run(settings):
    if getattr(settings, "config_name", "") == "fcvc_full":
        from lib.train.fcvc_pipeline import run as run_fcvc
        return run_fcvc(settings)

    settings.description = "Training script for EnTeRTrack / OSTrack on MDOT / ThreeMDOT"

    # ------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)

    config_module = importlib.import_module(
        "lib.config.%s.config" % settings.script_name
    )

    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)

    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print("\n")

    # update settings based on cfg
    update_settings(settings, cfg)

    # ------------------------------------------------------------
    # 2. Log directory
    # ------------------------------------------------------------
    log_dir = os.path.join(settings.save_dir, "logs")

    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    settings.log_file = os.path.join(
        log_dir,
        "%s-%s.log" % (settings.script_name, settings.config_name)
    )

    # ------------------------------------------------------------
    # 3. Build dataloaders
    # ------------------------------------------------------------
    if settings.script_name == "ostrack":
        loader_train, loader_val = build_dataloaders_mdot(cfg, settings)

    elif settings.script_name == "entertrack":
        if use_grouped_multiview_loader(cfg):
            loader_train, loader_val = build_dataloaders_threemdot(cfg, settings)
        else:
            # Single-UAV fine-tuning: treat ThreeMDOT views as independent videos.
            # No auxiliary-view templates/searches are sampled.
            loader_train, loader_val = build_dataloaders(cfg, settings)

    elif settings.script_name == "entertrack_teacher":
        loader_train, loader_val = build_dataloaders_threemdot(cfg, settings)

    else:
        raise ValueError("illegal script name: %s" % settings.script_name)

    if (
        "RepVGG" in cfg.MODEL.BACKBONE.TYPE
        or "swin" in cfg.MODEL.BACKBONE.TYPE
        or "LightTrack" in cfg.MODEL.BACKBONE.TYPE
    ):
        cfg.ckpt_dir = settings.save_dir

    # ------------------------------------------------------------
    # 4. Create network
    # ------------------------------------------------------------
    if settings.script_name == "ostrack":
        net = build_ostrack(cfg)

    elif settings.script_name == "ostrack_three":
        net = build_ostrack_prompt(cfg)

    elif settings.script_name in ["entertrack", "entertrack_teacher"]:
        # 单机 EnTeRTrack，不使用 teacher，不使用 prompt
        net = build_entertrack(cfg, training=True)

    else:
        raise ValueError("illegal script name: %s" % settings.script_name)

    # ------------------------------------------------------------
    # 5. Distributed training
    # ------------------------------------------------------------
    net.cuda()

    if settings.local_rank != -1:
        net = DDP(
            net,
            device_ids=[settings.local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False
        )
        settings.device = torch.device("cuda:%d" % settings.local_rank)

    else:
        settings.device = torch.device("cuda:0")

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # ------------------------------------------------------------
    # 6. Loss functions and Actors
    # ------------------------------------------------------------
    focal_loss = FocalLoss()

    if settings.script_name == "ostrack":
        objective = {
            "giou": giou_loss,
            "l1": l1_loss,
            "focal": focal_loss,
            "cls": BCEWithLogitsLoss()
        }

        loss_weight = {
            "giou": cfg.TRAIN.GIOU_WEIGHT,
            "l1": cfg.TRAIN.L1_WEIGHT,
            "focal": cfg.TRAIN.FOCAL_WEIGHT,
            "cls": 1.0
        }

        actor = OSTrackActorMDOT(
            net=net,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )

    elif settings.script_name == "ostrack_three":
        objective = {
            "giou": giou_loss,
            "l1": l1_loss,
            "focal": focal_loss,
            "cls": BCEWithLogitsLoss(),
            "prompt_consist": nn.MSELoss()
        }

        loss_weight = {
            "giou": cfg.TRAIN.GIOU_WEIGHT,
            "l1": cfg.TRAIN.L1_WEIGHT,
            "focal": 1.0,
            "cls": 1.0,
            "prompt_consist": getattr(cfg.TRAIN, "PROMPT_CONSIST_WEIGHT", 1.0)
        }

        actor = OSTrackActorPrompt(
            net=net,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )

    elif settings.script_name == "entertrack":
        # --------------------------------------------------------
        # 单机 EnTeRTrack 微调：
        # 不使用 teacher；
        # 不使用 prompt；
        # 不使用 distillation；
        # FLOPs constraint 在 EnTeRTrackActorThreeMDOT 内部计算。
        # --------------------------------------------------------
        objective = {
            "giou": giou_loss,
            "l1": l1_loss,
            "focal": focal_loss,
            "cls": BCEWithLogitsLoss()
        }

        loss_weight = {
            "giou": cfg.TRAIN.GIOU_WEIGHT,
            "l1": cfg.TRAIN.L1_WEIGHT,
            "focal": cfg.TRAIN.FOCAL_WEIGHT,
            "cls": 1.0
        }

        actor = EnTeRTrackActorThreeMDOT(
            net=net,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )
    elif settings.script_name == "entertrack_teacher":
        objective = {
            "giou": giou_loss,
            "l1": l1_loss,
            "focal": focal_loss,
            "cls": BCEWithLogitsLoss()
        }

        loss_weight = {
            "giou": cfg.TRAIN.GIOU_WEIGHT,
            "l1": cfg.TRAIN.L1_WEIGHT,
            "focal": cfg.TRAIN.FOCAL_WEIGHT,
            "cls": 1.0
        }

        actor = EnTeRTrackActorTeacher(
            net=net,
            objective=objective,
            loss_weight=loss_weight,
            settings=settings,
            cfg=cfg
        )

    else:
        raise ValueError("illegal script name: %s" % settings.script_name)

    # ------------------------------------------------------------
    # 7. Optimizer, scheduler, trainer
    # ------------------------------------------------------------
    if getattr(cfg.TRAIN, "PROMPT_ONLY", False):
        target_net = net.module if hasattr(net, "module") else net
        for name, param in target_net.named_parameters():
            param.requires_grad = "search_prompt_gate" in name
        if is_main_process():
            trainable = [name for name, param in target_net.named_parameters() if param.requires_grad]
            print("PROMPT_ONLY trainable parameters:", trainable)

    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    if settings.local_rank in [-1, 0]:
        target_net = net.module if hasattr(net, "module") else net
        pcum = getattr(target_net, "pcum", None)
        print("Initialization checkpoint:", cfg.MODEL.PRETRAIN_FILE)
        print("Optimizer state restored:", bool(getattr(cfg.TRAIN, "RESUME", False)))
        print("Scheduler state restored:", bool(getattr(cfg.TRAIN, "RESUME", False)))
        if pcum is not None:
            raw_scale = pcum.fusion.residual_scale.detach().float().item()
            effective_scale = pcum.fusion._residual_scale().detach().float().item()
            print("Initial PCUM residual scale: raw=%.8f effective=%.8f" % (
                raw_scale, effective_scale))
            if getattr(pcum.fusion, "mode", None) == "film":
                print("Film inactive parameters: pcum.fusion.gate, pcum.fusion.prompt_proj")

    use_amp = getattr(cfg.TRAIN, "AMP", False)

    trainer = LTRTrainer(
        actor,
        [loader_train, loader_val],
        optimizer,
        settings,
        lr_scheduler,
        use_amp=use_amp
    )

    # ------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------
    trainer.train(
        cfg.TRAIN.EPOCH,
        load_latest=getattr(cfg.TRAIN, "RESUME", False),
        fail_safe=False
    )
