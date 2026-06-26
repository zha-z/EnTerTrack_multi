from easydict import EasyDict as edict
import yaml
import copy

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.USE_SEARCH_PROMPT = False
cfg.MODEL.PROMPT_HIDDEN_DIM = 32
cfg.MODEL.PROMPT_INIT_SCALE = 0.1

cfg.MODEL.PCUM = edict()
cfg.MODEL.PCUM.ENABLED = False
cfg.MODEL.PCUM.LOCAL_ONLY = True
cfg.MODEL.PCUM.TOKEN_DIM = 192
cfg.MODEL.PCUM.PROMPT_DIM = 192
cfg.MODEL.PCUM.NUM_PROMPTS = 4
cfg.MODEL.PCUM.TOPK = 16
cfg.MODEL.PCUM.SALIENCY_SOURCE = "feature_norm"
cfg.MODEL.PCUM.FUSION = "gated_add"
cfg.MODEL.PCUM.ALIGN_GATE = "cosine"
cfg.MODEL.PCUM.FUSION_INIT_SCALE = 0.1
cfg.MODEL.PCUM.FUSION_SCALE_MAX = 0.0

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = [2, 5, 8, 11]

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FOCAL_WEIGHT = 5.0
cfg.TRAIN.PROMPT_CONSIST_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 10
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.RESUME = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.05  # drop path rate for ViT backbone
cfg.TRAIN.FLOPS_START_EPOCH = 0
cfg.TRAIN.FLOPS_END_EPOCH = 0
cfg.TRAIN.INITIAL_FLOPS_TARGET = 10e8
cfg.TRAIN.MAX_FLOPS_TARGET = 7e8
cfg.TRAIN.FLOPS_WEIGHT = 0.0
cfg.TRAIN.PROMPT_DROP_PROB = 0.3
cfg.TRAIN.PROMPT_NOISE_STD = 0.08
cfg.TRAIN.PROMPT_WRONG_PROB = 0.1
cfg.TRAIN.PROMPT_GATE_WEIGHT = 0.01
cfg.TRAIN.PROMPT_ONLY = False
cfg.TRAIN.PCUM_CONSIST_WEIGHT = 0.0
cfg.TRAIN.PCUM = edict()
cfg.TRAIN.PCUM.ALIGN_LOSS_WEIGHT = 0.0
cfg.TRAIN.PCUM.USE_PSEUDO_REMOTE = True
cfg.TRAIN.PCUM.STOP_GRAD_TEACHER = True
cfg.TRAIN.PCUM.USE_REAL_MULTIVIEW = False
cfg.TRAIN.PCUM.DETACH_REAL_REMOTE = True
cfg.TRAIN.PCUM.REAL_MULTIVIEW_LOSS_MODE = "all_views"
cfg.TRAIN.PCUM.REQUIRE_ALL_VIEWS_VISIBLE = False
cfg.TRAIN.PCUM.USE_REMOTE_VISIBLE_MASK = False
cfg.TRAIN.PCUM.CANONICAL_VIEW_ORDER = False
cfg.TRAIN.PCUM.REAL_MULTIVIEW_LOSS_WEIGHTS = []
cfg.TRAIN.PCUM.REMOTE_DROPOUT_PROB = 0.0

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.SCHEDULER.MILESTONES = [40, 60]
cfg.TRAIN.SCHEDULER.GAMMA = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
# cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT"]
# cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["LASOT_VAL"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 3
cfg.DATA.SEARCH.SCALE_JITTER = 0.25
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.SAVE_DIR = ""
cfg.TEST.CHECKPOINT_NAME = ""
cfg.TEST.USE_SEARCH_PROMPT = False
cfg.TEST.PROMPT_SELF_SCORE_THR = 0.25
cfg.TEST.PROMPT_SELF_APCE_THR = 100.0
cfg.TEST.PROMPT_PEER_SCORE_THR = 0.35
cfg.TEST.PROMPT_PEER_APCE_THR = 120.0
cfg.TEST.PROMPT_LARGE_SEARCH_FACTOR = 6.0
cfg.TEST.PCUM = edict()
cfg.TEST.PCUM.USE_REMOTE = False
cfg.TEST.PCUM.REMOTE_SCORE_THR = 0.0
cfg.TEST.PCUM.REMOTE_APCE_THR = 0.0
cfg.TEST.PCUM.MIN_REMOTE_PROMPTS = 1
cfg.TEST.PCUM.USE_REMOTE_ONLY_WHEN_LOCAL_LOW = False
cfg.TEST.PCUM.USE_REMOTE_VISIBLE_MASK = False
cfg.TEST.PCUM.LOCAL_LOW_SCORE_THR = 0.25
cfg.TEST.PCUM.LOCAL_LOW_APCE_THR = 100.0
cfg.TEST.PCUM.LOCAL_LOW_MODE = "or"
cfg.TEST.PCUM.KEEP_LOCAL_IF_REMOTE_WORSE = True
cfg.TEST.PCUM.REMOTE_SCORE_MAX_DROP = 0.05
cfg.TEST.PCUM.KEEP_LOCAL_IF_REMOTE_CONFIDENCE_WORSE = False
cfg.TEST.PCUM.REMOTE_CONFIDENCE_MAX_DROP = 0.02
cfg.TEST.PCUM.SAVE_DECISION_LOG = False
cfg.TEST.PCUM.USE_MOTION_REDETECT = False
cfg.TEST.PCUM.MOTION_REDETECT_SEARCH_FACTOR = 6.0
cfg.TEST.PCUM.MOTION_REDETECT_MIN_REMOTE = 1
cfg.TEST.PCUM.MOTION_REDETECT_MIN_RELIABILITY = 0.25
cfg.TEST.PCUM.MOTION_REDETECT_MAX_NORM_MOTION = 2.0
cfg.TEST.PCUM.MOTION_REDETECT_APCE_NORM = 200.0
cfg.TEST.PCUM.MOTION_REDETECT_USE_LOCAL_CANDIDATE = True
cfg.TEST.PCUM.MOTION_REDETECT_LOCAL_MIN_GAIN = 0.0
cfg.TEST.PCUM.USE_MOTION_CONFIDENCE = False
cfg.TEST.COOP = edict()
cfg.TEST.COOP.ENABLED = False
cfg.TEST.COOP.FUSION = "none"
cfg.TEST.COOP.PAYLOAD = "bbox_score"
cfg.TEST.COOP.SEND_INTERVAL = 1
cfg.TEST.COOP.BANDWIDTH_LIMIT_BYTES_PER_FRAME = None
cfg.TEST.COOP.PACKET_LOSS = 0.0
cfg.TEST.COOP.DELAY_FRAMES = 0
cfg.TEST.COOP.MAX_NEIGHBORS = None
cfg.TEST.COOP.SEED = 0
cfg.TEST.COOP.SAVE_STATS = True

_DEFAULT_CFG = copy.deepcopy(cfg)


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            cfg.clear()
            cfg.update(copy.deepcopy(_DEFAULT_CFG))
            _update_config(cfg, exp_config)
