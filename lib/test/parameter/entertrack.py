from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.entertrack.config import cfg, update_config_from_file
from easydict import EasyDict as edict


def parameters(yaml_name: str):
    params = TrackerParams()
    params.param_name = yaml_name
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/entertrack/%s.yaml' % yaml_name)
    if yaml_name == "fcvc_full":
        legacy_file = os.path.join(
            prj_dir, 'experiments/entertrack/entertrack_threemdot_lasot_ft_cons.yaml')
        update_config_from_file(legacy_file)
        from lib.train.fcvc_config import load_resolved_config

        resolved = load_resolved_config(yaml_file)
        cfg.MODEL.COLLABORATION = edict(resolved["MODEL"]["COLLABORATION"])
        cfg.MODEL.SAFE_COMMIT = edict(resolved["MODEL"]["SAFE_COMMIT"])
        for key, value in resolved["MODEL"]["FCVC"].items():
            setattr(cfg.MODEL.FCVC, key, value)
        cfg.TEST.EPOCH = resolved["TEST"]["EPOCH"]
        cfg.TEST.SAFE_COMMIT = resolved["TEST"]["SAFE_COMMIT"]
    else:
        update_config_from_file(yaml_file)
    params.cfg = cfg
    #print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    checkpoint_save_dir = getattr(cfg.TEST, "SAVE_DIR", "") or save_dir
    checkpoint_name = getattr(cfg.TEST, "CHECKPOINT_NAME", "") or yaml_name
    if yaml_name == "fcvc_full":
        params.checkpoint = os.path.join(
            "/data/zjy/multi/output/entertrack_single_lasot",
            "checkpoints/train/entertrack/entertrack_threemdot/"
            "EnTeRTrack_ep0021.pth.tar",
        )
        params.fcvc_checkpoint = os.path.join(
            prj_dir, "output/entertrack/fcvc_full/checkpoints/fcvc_student_epoch30.pth")
        params.fcvc_checkpoint_epoch = 30
    else:
        params.checkpoint = os.path.join(checkpoint_save_dir, "checkpoints/train/entertrack/%s/EnTeRTrack_ep%04d.pth.tar" %
                                         (checkpoint_name, cfg.TEST.EPOCH))
    temporal_cfg = getattr(cfg.TEST, "TEMPORAL_GATE", None)
    params.temporal_gate_checkpoint = str(getattr(
        temporal_cfg, "CHECKPOINT", "") or "")
    if (params.temporal_gate_checkpoint
            and not os.path.isabs(params.temporal_gate_checkpoint)):
        params.temporal_gate_checkpoint = os.path.join(
            prj_dir, params.temporal_gate_checkpoint)
    params.temporal_gate_checkpoint_sha256 = str(getattr(
        temporal_cfg, "CHECKPOINT_SHA256", "") or "")

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
