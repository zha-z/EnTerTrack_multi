import math
import os
import copy
import hashlib
import json
import numpy as np
import torch
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
from PIL import Image, ImageDraw

from lib.models.entertrack import build_entertrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.tracker.data_utils import Preprocessor
from lib.test.tracker.motion_state import MotionStateManager
from lib.test.tracker.mcr_redetection import (
    MCRRedetectionManager,
    RedetectionCandidate,
    bbox_center,
)
from lib.test.utils.pcum_diagnostics import (
    PCUMDiagnosticHooks,
    normalized_response_entropy,
    prompt_norm,
)
from lib.test.utils.hann import hann2d
from lib.test.utils.c3r_inference import (
    C3RReceiverContext,
    build_packet_record,
    collaborate_local_candidate,
    diagnostic_row as c3r_diagnostic_row,
)
from lib.models.entertrack.c3r import MessageAccounting
from lib.models.entertrack.temporal_gate import (
    TemporalGateRuntime,
    load_temporal_gate_checkpoint,
)
from lib.models.entertrack.fcvc.structures import FrameTrackingResult
from lib.models.entertrack.fcvc import (
    FCVCConfig,
    FCVCModel,
    build_sender_bundle,
    load_fcvc_checkpoint,
)
from lib.models.entertrack.fcvc.feature_taps import capture_taps, split_template_search
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


RELIABILITY_SELECTOR_MODES = ("none", "deterministic")


FCVC_PERSISTENT_STATE_REGISTRY = (
    {
        "owner": "EnTeRTrack",
        "function": "__init__/initialize/_commit_state_from_candidate",
        "name": "state",
        "dtype_shape": "list[4] xywh",
        "init": "None then init_bbox",
        "read": "_run_candidate map_box_back get_redetection_context",
        "write": "initialize _commit_state_from_candidate",
        "source": "local in FCVC",
        "next_crop": True,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "E0/C1 unchanged; FCVC commits local state_output only",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_track_single/c3r_track_with_packets/pcum_track_with_remote",
        "name": "frame_id",
        "dtype_shape": "int scalar",
        "init": "0",
        "read": "diagnostics packet context motion/MCR",
        "write": "per-frame increment",
        "source": "local runtime clock",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "FCVC increments before predict like E0 runtime",
    },
    {
        "owner": "EnTeRTrack",
        "function": "initialize",
        "name": "z_dict1",
        "dtype_shape": "NestedTensor template",
        "init": "preprocessed template crop",
        "read": "_network_forward",
        "write": "initialize",
        "source": "local initialization",
        "next_crop": False,
        "template_memory": True,
        "packet_sender": False,
        "digest": True,
        "behavior": "unchanged; FCVC never rewrites from collaborative branch",
    },
    {
        "owner": "EnTeRTrack",
        "function": "initialize",
        "name": "z_patch_arr",
        "dtype_shape": "image array",
        "init": "template crop array",
        "read": "debug visualization",
        "write": "initialize",
        "source": "local initialization",
        "next_crop": False,
        "template_memory": True,
        "packet_sender": False,
        "digest": True,
        "behavior": "unchanged",
    },
    {
        "owner": "EnTeRTrack",
        "function": "initialize",
        "name": "box_mask_z",
        "dtype_shape": "tensor/bool mask or None",
        "init": "CE mask or None",
        "read": "_network_forward",
        "write": "initialize",
        "source": "local initialization",
        "next_crop": False,
        "template_memory": True,
        "packet_sender": False,
        "digest": True,
        "behavior": "unchanged",
    },
    {
        "owner": "EnTeRTrack",
        "function": "__init__/close_pcum_diagnostics",
        "name": "_pcum_diagnostic_hooks",
        "dtype_shape": "object diagnostic buffers",
        "init": "PCUMDiagnosticHooks",
        "read": "_run_candidate diagnostics",
        "write": "reset/snapshot/remove",
        "source": "local diagnostic path",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": False,
        "digest": True,
        "behavior": "FCVC local payload freezes diagnostics before collaboration",
    },
    {
        "owner": "EnTeRTrack",
        "function": "pcum_track_with_remote",
        "name": "last_pcum_diagnostic",
        "dtype_shape": "dict or None",
        "init": "None",
        "read": "diagnostic output",
        "write": "pcum diagnostics only",
        "source": "selected PCUM candidate outside FCVC",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": False,
        "digest": True,
        "behavior": "FCVC does not use this as state input",
    },
    {
        "owner": "EnTeRTrack",
        "function": "__init__/c3r_track_with_packets",
        "name": "c3r_last_frame_by_sender",
        "dtype_shape": "dict[int,int]",
        "init": "{}",
        "read": "C3R context and Temporal Gate",
        "write": "C3R packet runtime",
        "source": "packet source local state",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "C1 unchanged; FCVC sender state must be local provenance",
    },
    {
        "owner": "EnTeRTrack",
        "function": "__init__/c3r_track_with_packets",
        "name": "c3r_message_accounting",
        "dtype_shape": "MessageAccounting object",
        "init": "MessageAccounting()",
        "read": "accounting report",
        "write": "record_frame",
        "source": "packet accounting",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "not updated by FCVC reported output",
    },
    {
        "owner": "EnTeRTrack",
        "function": "__init__/reset_temporal_gate",
        "name": "temporal_gate_runtime",
        "dtype_shape": "TemporalGateRuntime or None",
        "init": "None or checkpoint sidecar",
        "read": "C3R gate_provider",
        "write": "gate_for/reset",
        "source": "C1 sidecar only",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "FCVC runtime path excludes Temporal Gate",
    },
    {
        "owner": "EnTeRTrack",
        "function": "__init__/initialize/_apply_mcr",
        "name": "mcr_manager",
        "dtype_shape": "MCRRedetectionManager or None",
        "init": "config object",
        "read": "redetection context/process",
        "write": "reset/process internal histories",
        "source": "local in FCVC",
        "next_crop": True,
        "template_memory": False,
        "packet_sender": False,
        "digest": True,
        "behavior": "FCVC commit uses local candidate/payload only",
    },
    {
        "owner": "EnTeRTrack",
        "function": "__init__/apply_confirmed_redetection",
        "name": "mcr_updates_allowed",
        "dtype_shape": "bool",
        "init": "True",
        "read": "get_redetection_context",
        "write": "initialize/apply_confirmed_redetection",
        "source": "local MCR result in FCVC",
        "next_crop": True,
        "template_memory": False,
        "packet_sender": False,
        "digest": True,
        "behavior": "reported_output cannot change it",
    },
    {
        "owner": "EnTeRTrack",
        "function": "__init__/initialize/_attach_motion_shadow_diagnostics",
        "name": "motion_state_manager",
        "dtype_shape": "MotionStateManager or None",
        "init": "config object",
        "read": "diagnostic output",
        "write": "reset/update_prediction_only histories",
        "source": "local in FCVC",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": False,
        "digest": True,
        "behavior": "FCVC updates motion/confidence from local state_output",
    },
    {
        "owner": "EnTeRTrack",
        "function": "__init__",
        "name": "output_window",
        "dtype_shape": "tensor [feat_sz,feat_sz]",
        "init": "hann2d",
        "read": "_decode_prediction",
        "write": "constructor only",
        "source": "configuration",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": False,
        "digest": True,
        "behavior": "unchanged constant",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.crop_bbox",
        "dtype_shape": "list[4] xywh",
        "init": "pre-frame state/reference bbox",
        "read": "map_box_back C3R/FCVC sender payload",
        "write": "candidate creation",
        "source": "local payload in FCVC",
        "next_crop": True,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "frozen before collaborative branch",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.resize_factor",
        "dtype_shape": "float",
        "init": "sample_target result",
        "read": "box mapping and sender payload",
        "write": "candidate creation",
        "source": "local payload in FCVC",
        "next_crop": True,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "frozen before collaborative branch",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.search_factor",
        "dtype_shape": "float",
        "init": "params/search override",
        "read": "diagnostics next crop audit",
        "write": "candidate creation",
        "source": "local payload in FCVC",
        "next_crop": True,
        "template_memory": False,
        "packet_sender": False,
        "digest": True,
        "behavior": "frozen before collaborative branch",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.max_score",
        "dtype_shape": "tensor/scalar",
        "init": "local response decode",
        "read": "confidence/history/diagnostics",
        "write": "candidate creation",
        "source": "local payload in FCVC",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "reported score is display-only",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.apce",
        "dtype_shape": "tensor/scalar",
        "init": "calAPCE(local response)",
        "read": "motion/confidence diagnostics",
        "write": "candidate creation",
        "source": "local payload in FCVC",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "FCVC motion/confidence histories consume local apce",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.response",
        "dtype_shape": "tensor [B,H,W] or [B,1,H,W]",
        "init": "windowed local score map",
        "read": "APCE entropy quality packet",
        "write": "candidate creation",
        "source": "local payload in FCVC",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "frozen before collaborative branch",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.out_dict",
        "dtype_shape": "dict of network tensors",
        "init": "network forward output",
        "read": "debug C3R packet diagnostics",
        "write": "candidate creation",
        "source": "local payload in FCVC for sender bundle",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "collaborative out_dict not used for commit",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.local_prompt",
        "dtype_shape": "tensor or None",
        "init": "local forward",
        "read": "PCUM prompt exchange",
        "write": "candidate creation",
        "source": "local payload in FCVC",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "sender bundle/prompt source remains local",
    },
    {
        "owner": "EnTeRTrack",
        "function": "_run_candidate",
        "name": "candidate.remote_states",
        "dtype_shape": "dict/list or None",
        "init": "runtime input",
        "read": "diagnostics only",
        "write": "candidate creation",
        "source": "remote input, not commit state",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": False,
        "digest": True,
        "behavior": "FCVC excludes from local commit payload unless local source",
    },
    {
        "owner": "evaluation runner",
        "function": "lib/test/evaluation/tracker.py three-view loops",
        "name": "per-view tracker objects",
        "dtype_shape": "tracker instances",
        "init": "one per view",
        "read": "receiver/sender orchestration",
        "write": "each tracker commits own state",
        "source": "local state_output",
        "next_crop": True,
        "template_memory": True,
        "packet_sender": True,
        "digest": True,
        "behavior": "FCVC mock runner keeps A/B/C and target streams isolated",
    },
    {
        "owner": "evaluation runner",
        "function": "message exchange",
        "name": "sender packet/bundle queue",
        "dtype_shape": "dict/list payloads",
        "init": "per frame local candidates",
        "read": "receiver collaboration",
        "write": "after local branch",
        "source": "local payload only",
        "next_crop": False,
        "template_memory": False,
        "packet_sender": True,
        "digest": True,
        "behavior": "reported_output is not enqueued",
    },
)


def validate_reliability_selector(mode):
    mode = str(mode).lower()
    if mode not in RELIABILITY_SELECTOR_MODES:
        raise ValueError(
            "Unsupported PCUM reliability selector: {}".format(mode)
        )
    return mode


def deterministic_reliability_selector_decision(
    local_confidence,
    collaborative_confidence,
    collaborative_motion_reliability,
    margin=0.0,
    motion_threshold=0.0,
):
    """Prediction-only deterministic choice between local and collaborative."""
    local_confidence = float(local_confidence)
    collaborative_confidence = float(collaborative_confidence)
    collaborative_motion_reliability = float(collaborative_motion_reliability)
    margin = float(margin)
    motion_threshold = float(motion_threshold)
    confidence_delta = collaborative_confidence - local_confidence
    use_collaborative = (
        confidence_delta >= margin
        and collaborative_motion_reliability >= motion_threshold
    )
    return {
        "use_collaborative": bool(use_collaborative),
        "local_confidence": local_confidence,
        "collaborative_confidence": collaborative_confidence,
        "confidence_delta": confidence_delta,
        "collaborative_motion_reliability": collaborative_motion_reliability,
        "margin": margin,
        "motion_threshold": motion_threshold,
    }


class EnTeRTrack(BaseTracker):
    """
    ThreeMDOT-compatible single-view EnTeRTrack tracker.

    当前版本：
    1. 使用 EnTeRTrack 作为单机主干；
    2. 只使用单模板 + 单搜索区域；
    3. 不使用 prompt；
    4. 不使用 B/C 机模板；
    5. 不使用 B/C 机搜索区域；
    6. 保留 ThreeMDOT 测试代码中可能调用的接口。
    """

    def __init__(self, params, dataset_name):
        super(EnTeRTrack, self).__init__(params)

        self.cfg = params.cfg
        mcr_cfg = getattr(self.cfg.TEST, "MCR", None)
        self.mcr_config_name = str(getattr(params, "param_name", "unknown"))
        mcr_enabled = bool(getattr(mcr_cfg, "ENABLED", False))
        mcr_shadow_only = bool(getattr(mcr_cfg, "SHADOW_ONLY", True))
        mcr_global_enabled = bool(getattr(mcr_cfg, "GLOBAL_ENABLED", False))
        if not mcr_enabled:
            mcr_mode = "DISABLED"
        elif mcr_shadow_only:
            mcr_mode = "SHADOW"
        else:
            mcr_mode = "ACTIVE"
        print(
            "[MCR MODE]\n"
            "config={}\n"
            "enabled={}\n"
            "shadow_only={}\n"
            "global_enabled={}\n"
            "mode={}".format(
                self.mcr_config_name,
                str(mcr_enabled).lower(),
                str(mcr_shadow_only).lower(),
                str(mcr_global_enabled).lower(),
                mcr_mode,
            )
        )

        network = build_entertrack(params.cfg, training=False)
        self._load_network(network, self.params.checkpoint)

        self.network = network.cuda()
        self.network.eval()
        plain_model_enabled = bool(getattr(
            getattr(self.cfg.MODEL, "PLAIN_COLLABORATION", None),
            "ENABLED", False))
        plain_test_enabled = bool(getattr(
            getattr(self.cfg.TEST, "PLAIN_COLLABORATION", None),
            "ENABLED", False))
        if plain_model_enabled != plain_test_enabled:
            raise RuntimeError(
                "Plain Collaboration MODEL/TEST enable flags must match")
        self.plain_collaboration_enabled = bool(
            plain_model_enabled and plain_test_enabled)
        plain_test_cfg = getattr(
            self.cfg.TEST, "PLAIN_COLLABORATION", None)
        self.plain_collaboration_safe_commit = bool(getattr(
            plain_test_cfg, "SAFE_COMMIT", False))
        self.plain_collaboration_counterfactual_diagnostics = bool(getattr(
            plain_test_cfg, "SAVE_COUNTERFACTUAL_DIAGNOSTICS", False))
        self.plain_collaboration_sender_counterfactual_diagnostics = bool(
            getattr(
                plain_test_cfg,
                "SAVE_SENDER_COUNTERFACTUAL_DIAGNOSTICS",
                False,
            ))
        self.plain_collaboration_target_consistency_diagnostics = bool(
            getattr(
                plain_test_cfg,
                "SAVE_TARGET_CONSISTENCY_DIAGNOSTICS",
                False,
            ))
        self._plain_collaboration_local_forward_count = 0
        if self.plain_collaboration_safe_commit and not self.plain_collaboration_enabled:
            raise RuntimeError(
                "Plain Collaboration SAFE_COMMIT requires collaboration inference")
        if (self.plain_collaboration_counterfactual_diagnostics
                and not self.plain_collaboration_enabled):
            raise RuntimeError(
                "Plain Collaboration counterfactual diagnostics require inference")
        if (self.plain_collaboration_sender_counterfactual_diagnostics
                and not self.plain_collaboration_enabled):
            raise RuntimeError(
                "Plain Collaboration sender counterfactuals require inference")
        if (self.plain_collaboration_sender_counterfactual_diagnostics
                and not self.plain_collaboration_safe_commit):
            raise RuntimeError(
                "Plain Collaboration sender counterfactuals require SAFE_COMMIT")
        if (self.plain_collaboration_target_consistency_diagnostics
                and not self.plain_collaboration_enabled):
            raise RuntimeError(
                "Plain Collaboration target consistency diagnostics require inference")
        if (self.plain_collaboration_target_consistency_diagnostics
                and not self.plain_collaboration_safe_commit):
            raise RuntimeError(
                "Plain Collaboration target consistency diagnostics require SAFE_COMMIT")
        if (self.plain_collaboration_target_consistency_diagnostics
                and not self.plain_collaboration_sender_counterfactual_diagnostics):
            raise RuntimeError(
                "Plain Collaboration target consistency diagnostics require "
                "sender counterfactual diagnostics")
        if self.plain_collaboration_enabled:
            if self.network.plain_collaboration is None:
                raise RuntimeError(
                    "Plain Collaboration inference requires the V1 adapter")
            if any((
                    self.network.pcum is not None,
                    self.network.c3r is not None,
                    self.network.search_prompt_gate is not None)):
                raise RuntimeError(
                    "Plain Collaboration inference is exclusive with "
                    "PCUM/C3R/search prompt")
            if int(self.network.feat_len_s) != 256:
                raise RuntimeError(
                    "Plain Collaboration V1 requires 256 search tokens")
        self.fcvc_enabled = bool(
            str(getattr(getattr(self.cfg.MODEL, "COLLABORATION", None), "TYPE", "")).lower() == "fcvc"
            and bool(getattr(getattr(self.cfg.MODEL, "FCVC", None), "ENABLED", False))
        )
        self.fcvc_model = None
        if self.fcvc_enabled:
            self.fcvc_model = self._load_fcvc_sidecar(params).cuda().eval()
        self.c3r_enabled = bool(
            getattr(getattr(self.cfg.MODEL, "C3R", None), "ENABLED", False)
            and getattr(getattr(self.cfg.TEST, "C3R", None), "ENABLED", False)
        )
        if self.c3r_enabled and self.network.c3r is None:
            raise RuntimeError("formal C3R inference requires network.c3r")
        self.c3r_last_frame_by_sender = {}
        self.c3r_message_accounting = MessageAccounting()
        temporal_model_enabled = bool(getattr(
            getattr(self.cfg.MODEL, "TEMPORAL_GATE", None), "ENABLED", False))
        temporal_test_enabled = bool(getattr(
            getattr(self.cfg.TEST, "TEMPORAL_GATE", None), "ENABLED", False))
        if temporal_model_enabled != temporal_test_enabled:
            raise RuntimeError("Temporal Gate MODEL/TEST enable flags must match")
        self.temporal_gate_enabled = bool(
            self.c3r_enabled and temporal_model_enabled and temporal_test_enabled)
        if temporal_model_enabled and not self.c3r_enabled:
            raise RuntimeError("Temporal Gate requires the frozen C1 packet path")
        self.temporal_gate_runtime = None
        if self.temporal_gate_enabled:
            checkpoint = str(getattr(params, "temporal_gate_checkpoint", "") or "")
            if not checkpoint or not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    "enabled Temporal Gate requires a separate sidecar checkpoint")
            sidecar = load_temporal_gate_checkpoint(
                checkpoint,
                expected_sha256=str(getattr(
                    params, "temporal_gate_checkpoint_sha256", "") or ""),
                map_location="cpu",
            ).cuda().eval()
            self.temporal_gate_runtime = TemporalGateRuntime(sidecar)
        self.no_gt_inference = bool(getattr(params, "no_gt_inference", False))
        self.c3r_instrumentation_enabled = bool(getattr(
            params, "c3r_instrumentation", False))
        self.c3r_instrumentation_fold_id = int(getattr(
            params, "instrumentation_fold_id", -1))
        self.temporal_gate_rollout_capture = bool(getattr(
            params, "temporal_gate_rollout_capture", False))
        self.temporal_gate_counterfactual_diagnostics = bool(getattr(
            params, "temporal_gate_counterfactual_diagnostics", False))
        self.remote_information_diagnostics = bool(getattr(
            params, "remote_information_diagnostics", False))
        if (self.temporal_gate_counterfactual_diagnostics
                and not self.temporal_gate_rollout_capture):
            raise RuntimeError(
                "counterfactual diagnostics require rollout capture")
        if (self.remote_information_diagnostics
                and not self.temporal_gate_counterfactual_diagnostics):
            raise RuntimeError(
                "remote-information diagnostics require counterfactual diagnostics")
        self._temporal_gate_backbone_forward_count = 0
        if self.c3r_instrumentation_enabled and not self.c3r_enabled:
            raise RuntimeError("C3R instrumentation requires formal C3R inference")

        diagnostics_cfg = getattr(
            getattr(self.cfg.TEST, "PCUM", None),
            "FRAME_DIAGNOSTICS",
            None,
        )
        self.pcum_diagnostics_enabled = bool(
            getattr(diagnostics_cfg, "ENABLED", False)
        )
        self._pcum_diagnostic_hooks = PCUMDiagnosticHooks(
            self.network,
            enabled=self.pcum_diagnostics_enabled,
        )
        self.last_pcum_diagnostic = None
        self.remote_weight_diagnostics_enabled = bool(getattr(
            getattr(self.cfg.MODEL, "PCUM", None),
            "REMOTE_WEIGHT_DIAGNOSTICS",
            True,
        ))
        motion_cfg = getattr(self.cfg.TEST, "MOTION_STATE", None)
        self.motion_state_enabled = bool(getattr(motion_cfg, "ENABLED", False))
        self.motion_state_log_enabled = bool(
            self.motion_state_enabled
            and getattr(motion_cfg, "LOG_ENABLED", False)
        )
        if self.motion_state_enabled and not bool(
            getattr(motion_cfg, "SHADOW_ONLY", True)
        ):
            raise ValueError("M0 motion state supports SHADOW_ONLY=true only")
        self.motion_state_manager = (
            MotionStateManager.from_config(motion_cfg)
            if self.motion_state_enabled else None
        )
        self.mcr_manager = MCRRedetectionManager(mcr_cfg) if mcr_cfg is not None else None
        self.mcr_enabled = bool(self.mcr_manager is not None and self.mcr_manager.enabled)
        self.mcr_updates_allowed = True

        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(),
            centered=True
        ).cuda()

        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0

        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        self.save_all_boxes = params.save_all_boxes

        # 单机版本只保留一个模板
        self.z_dict1 = None
        self.z_patch_arr = None
        self.box_mask_z = None

    # ------------------------------------------------------------
    # Network loading
    # ------------------------------------------------------------
    def _load_network(self, network, checkpoint_path):
        """
        兼容普通 checkpoint / DDP checkpoint。
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "net" in checkpoint:
            state_dict = checkpoint["net"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        try:
            network.load_state_dict(state_dict, strict=True)

        except RuntimeError:
            if (getattr(network, "c3r", None) is not None
                    or getattr(network, "plain_collaboration", None) is not None):
                raise
            new_state_dict = {}

            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[len("module."):]] = v
                else:
                    new_state_dict[k] = v

            missing_keys, unexpected_keys = network.load_state_dict(
                new_state_dict,
                strict=False
            )

            print("Load checkpoint with strict=False")
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

    def _load_fcvc_sidecar(self, params):
        checkpoint = str(getattr(params, "fcvc_checkpoint", "") or "")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                "FCVC inference requires fcvc_student_epoch30 sidecar: {}".format(
                    checkpoint))
        raw_cfg = self.cfg.MODEL.FCVC
        if hasattr(raw_cfg, "items"):
            cfg_items = raw_cfg.items()
        else:
            cfg_items = (
                (name, getattr(raw_cfg, name))
                for name in dir(raw_cfg) if name.isupper()
            )
        model_cfg = {
            str(key).lower(): value
            for key, value in cfg_items
            if str(key).lower() in FCVCConfig.__dataclass_fields__
        }
        fcvc = FCVCModel(FCVCConfig(**model_cfg))
        result = load_fcvc_checkpoint(fcvc, checkpoint, strict=False)
        missing = sorted(result.missing_keys)
        unexpected = sorted(result.unexpected_keys)
        non_teacher_missing = [
            key for key in missing if not key.startswith("teacher.")
        ]
        if non_teacher_missing or unexpected:
            raise RuntimeError(
                "FCVC sidecar load failed: missing_non_teacher={}, unexpected={}".format(
                    non_teacher_missing, unexpected))
        fcvc.sidecar_checkpoint = checkpoint
        print("Load FCVC sidecar from: {}".format(checkpoint))
        print("FCVC sidecar missing teacher-only keys: {}".format(len(missing)))
        return fcvc

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def initialize(self, image, info: dict):
        """
        单机初始化：只使用当前视角 image 和 init_bbox。
        """
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            info["init_bbox"],
            self.params.template_factor,
            output_sz=self.params.template_size
        )

        self.z_patch_arr = z_patch_arr

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                info["init_bbox"],
                resize_factor,
                template.tensors.device
            ).squeeze(1)

            self.box_mask_z = generate_mask_cond(
                self.cfg,
                1,
                template.tensors.device,
                template_bbox
            )

        self.state = info["init_bbox"]
        self.frame_id = 0
        output = {}
        if self.mcr_manager is not None:
            self.mcr_manager.reset(initial_bbox=info["init_bbox"])
        self.mcr_updates_allowed = True
        if self.motion_state_manager is not None:
            initial_record = self.motion_state_manager.reset(
                initial_bbox=info["init_bbox"],
                image_size=image.shape[:2],
            )
            if self.motion_state_log_enabled:
                output["motion_state_diagnostics"] = initial_record
        if self.save_all_boxes:
            all_boxes_save = info["init_bbox"] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            output["all_boxes"] = all_boxes_save
        return output or None

    def multi_initialize(self, image_a, image_b, init_info_a, init_info_b):
        """
        ThreeMDOT 双机接口兼容。

        当前单机 EnTeRTrack 版本只使用 A 机：
            image_a + init_info_a
        """
        return self.initialize(image_a, init_info_a)

    def three_multi_initialize(
        self,
        image_a,
        image_b,
        image_c,
        init_info_a,
        init_info_b,
        init_info_c
    ):
        """
        ThreeMDOT 三机接口兼容。

        当前单机 EnTeRTrack 版本只使用 A 机：
            image_a + init_info_a
        """
        return self.initialize(image_a, init_info_a)

    # ------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------
    def _network_forward(self, search_tensor, prompt_map=None, prompt_gate_input=None,
                         remote_prompts=None, remote_states=None):
        """
        EnTeRTrack forward.

        注意：
        推理阶段显式传 training=False。
        """
        if self.temporal_gate_counterfactual_diagnostics:
            self._temporal_gate_backbone_forward_count += 1
        out_dict = self.network.forward(
            template=self.z_dict1.tensors,
            search=search_tensor,
            ce_template_mask=self.box_mask_z,
            return_last_attn=False,
            return_atp=True,
            training=False,
            prompt_map=prompt_map,
            prompt_gate_input=prompt_gate_input,
            remote_prompts=remote_prompts,
            remote_states=remote_states
        )

        return out_dict

    def _to_float(self, x):
        return x.item() if torch.is_tensor(x) else float(x)

    def _build_prompt_map(self, peer_state=None):
        """
        Build a local search-coordinate prompt map.

        Cross-view image coordinates are not geometrically aligned, so peer boxes
        are used for confidence/scale only. The spatial center remains the local
        search center.
        """
        feat_sz = self.feat_sz
        device = self.output_window.device
        yy, xx = torch.meshgrid(
            torch.arange(feat_sz, device=device, dtype=torch.float32),
            torch.arange(feat_sz, device=device, dtype=torch.float32),
            indexing="ij"
        )

        center = (feat_sz - 1) * 0.5
        sigma = max(float(feat_sz) / 6.0, 1.0)
        if peer_state is not None and peer_state.get("bbox") is not None:
            pw = max(float(peer_state["bbox"][2]), 1.0)
            ph = max(float(peer_state["bbox"][3]), 1.0)
            scale = max((pw * ph) ** 0.5, 1.0)
            sigma = max(min(scale / 32.0, float(feat_sz) / 3.0), 1.0)

        prompt_map = torch.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * sigma * sigma))
        prompt_map = prompt_map / prompt_map.max().clamp(min=1e-6)
        return prompt_map.view(1, 1, feat_sz, feat_sz)

    def _build_prompt_gate_input(self, self_score, self_apce, peer_states):
        valid_peers = [
            p for p in peer_states
            if p is not None
            and p.get("score", 0.0) >= getattr(self.cfg.TEST, "PROMPT_PEER_SCORE_THR", 0.35)
            and p.get("apce", 0.0) >= getattr(self.cfg.TEST, "PROMPT_PEER_APCE_THR", 120.0)
        ]

        if not valid_peers:
            return None, None

        best_peer = max(valid_peers, key=lambda p: p.get("score", 0.0) * p.get("apce", 0.0))
        self_area = max(float(self.state[2] * self.state[3]), 1.0) if self.state is not None else 1.0
        peer_box = best_peer.get("bbox", None)
        peer_area = max(float(peer_box[2] * peer_box[3]), 1.0) if peer_box is not None else self_area
        scale_ratio = max(min((peer_area / self_area) ** 0.5, 4.0), 0.25) / 4.0

        gate_input = torch.tensor([[
            float(self_score),
            float(self_apce) / 200.0,
            float(best_peer.get("score", 0.0)),
            float(best_peer.get("apce", 0.0)) / 200.0,
            scale_ratio,
            min(len(valid_peers), 2) / 2.0,
        ]], device=self.output_window.device, dtype=torch.float32)

        return self._build_prompt_map(best_peer), gate_input

    def _should_use_prompt(self, self_score, self_apce, peer_states):
        if not getattr(self.cfg.TEST, "USE_SEARCH_PROMPT", False):
            return False

        self_low = (
            float(self_score) < getattr(self.cfg.TEST, "PROMPT_SELF_SCORE_THR", 0.25)
            or float(self_apce) < getattr(self.cfg.TEST, "PROMPT_SELF_APCE_THR", 100.0)
        )
        if not self_low:
            return False

        _, gate_input = self._build_prompt_gate_input(self_score, self_apce, peer_states)
        return gate_input is not None

    def track_with_peer_prompts(self, image, info, self_score, self_apce, peer_states):
        prompt_map, gate_input = self._build_prompt_gate_input(self_score, self_apce, peer_states)
        if prompt_map is None:
            return None

        search_factor = getattr(self.cfg.TEST, "PROMPT_LARGE_SEARCH_FACTOR", self.params.search_factor)
        return self._track_single(
            image=image,
            info=info,
            search_factor=search_factor,
            return_score_apce=True,
            debug_name="prompt",
            prompt_map=prompt_map,
            prompt_gate_input=gate_input
        )

    def _decode_prediction(self, out_dict, resize_factor, return_score=False):
        """
        Decode EnTeRTrack output to box in search-image coordinate.

        返回：
            pred_box: [cx, cy, w, h] in original-image coordinate offset before map_box_back
            pred_boxes: all query boxes
            max_score: max response score
            response: hann-windowed response map
        """
        pred_score_map = out_dict["score_map"]
        response = self.output_window * pred_score_map

        max_score = response.flatten(1).max(dim=1)[0]

        if "size_map" in out_dict and "offset_map" in out_dict:
            if return_score:
                try:
                    pred_boxes, max_score_from_head = self.network.box_head.cal_bbox(
                        response,
                        out_dict["size_map"],
                        out_dict["offset_map"],
                        return_score=True
                    )
                    max_score = max_score_from_head
                except TypeError:
                    pred_boxes = self.network.box_head.cal_bbox(
                        response,
                        out_dict["size_map"],
                        out_dict["offset_map"]
                    )
            else:
                pred_boxes = self.network.box_head.cal_bbox(
                    response,
                    out_dict["size_map"],
                    out_dict["offset_map"]
                )

        else:
            # 兼容 CORNER head 或只有 pred_boxes 的情况
            pred_boxes = out_dict["pred_boxes"]

        pred_boxes = pred_boxes.view(-1, 4)

        pred_box = (
            pred_boxes.mean(dim=0) * self.params.search_size / resize_factor
        ).tolist()

        return pred_box, pred_boxes, max_score, response

    def _run_candidate(
        self,
        image,
        search_factor=None,
        prompt_map=None,
        prompt_gate_input=None,
        remote_prompts=None,
        remote_states=None,
        return_score=True,
        reference_bbox=None,
    ):
        """
        Run one forward pass from the current state without committing it.

        This is used by test-time PCUM: first collect local prompts for all
        UAVs, then run a second pass with peer prompts and commit only the
        selected candidate.
        """
        H, W, _ = image.shape

        if search_factor is None:
            search_factor = self.params.search_factor

        crop_bbox = self.state if reference_bbox is None else reference_bbox
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            crop_bbox,
            search_factor,
            output_sz=self.params.search_size
        )

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        if self.pcum_diagnostics_enabled:
            self._pcum_diagnostic_hooks.reset()

        with torch.no_grad():
            out_dict = self._network_forward(
                search.tensors,
                prompt_map=prompt_map,
                prompt_gate_input=prompt_gate_input,
                remote_prompts=remote_prompts,
                remote_states=remote_states
            )

        pred_box, pred_boxes, max_score, response = self._decode_prediction(
            out_dict,
            resize_factor,
            return_score=return_score
        )

        target_bbox = clip_box(
            self.map_box_back(pred_box, resize_factor, reference_bbox=crop_bbox),
            H,
            W,
            margin=10
        )

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size / resize_factor,
                resize_factor,
                reference_bbox=crop_bbox,
            )
            output = {
                "target_bbox": target_bbox,
                "all_boxes": all_boxes.view(-1).tolist()
            }
        else:
            output = {"target_bbox": target_bbox}

        candidate = {
            "output": output,
            "target_bbox": target_bbox,
            "prev_bbox": list(self.state) if self.state is not None else None,
            "crop_bbox": list(crop_bbox),
            "resize_factor": float(resize_factor),
            "search_factor": float(search_factor),
            "max_score": max_score,
            "apce": self.calAPCE(response),
            "response": response,
            "out_dict": out_dict,
            "pred_boxes": pred_boxes,
            "x_patch_arr": x_patch_arr,
            "image": image,
            "local_prompt": out_dict.get("local_prompt", None),
            "aligned_prompt": out_dict.get("aligned_prompt", None),
            "align_gate": out_dict.get("pcum", {}).get("align_gate", None)
                if isinstance(out_dict.get("pcum", None), dict) else None,
            "used_remote": remote_prompts is not None,
            "remote_states": remote_states,
        }
        pcum_output = out_dict.get("pcum", None)
        if isinstance(pcum_output, dict):
            candidate["remote_weights"] = pcum_output.get("remote_weights", None)
            candidate["remote_quality"] = pcum_output.get("remote_quality", None)
            candidate["remote_aggregation_diagnostics"] = pcum_output.get(
                "remote_aggregation_diagnostics", None
            )
            candidate["remote_suppression"] = pcum_output.get(
                "remote_suppression", None)
            candidate["remote_delta_norm"] = pcum_output.get(
                "remote_delta_norm", None)
            candidate["suppressed_delta_norm"] = pcum_output.get(
                "suppressed_delta_norm", None)
            candidate["remote_suppression_active_ratio"] = pcum_output.get(
                "remote_suppression_active_ratio", None)
        if self.pcum_diagnostics_enabled:
            candidate["diagnostics"] = {
                **self._pcum_diagnostic_hooks.snapshot(),
                "response_entropy": normalized_response_entropy(response),
                "prompt_norm": prompt_norm(candidate["local_prompt"]),
                "aligned_prompt_norm": prompt_norm(candidate["aligned_prompt"]),
            }
        return candidate

    def fcvc_local_candidate(self, image, search_factor=None):
        if not self.fcvc_enabled or self.fcvc_model is None:
            raise RuntimeError("fcvc_local_candidate requires loaded FCVC sidecar")
        H, W, _ = image.shape
        if search_factor is None:
            search_factor = self.params.search_factor
        crop_bbox = self.state
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, crop_bbox, search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            taps = capture_taps(
                self.network.backbone, self.z_dict1.tensors, search.tensors)
            template_mid, mid_search = split_template_search(taps.mid_tokens)
            template_high, high_search = split_template_search(taps.final_tokens)
            out_dict = self.network.forward_head(taps.final_tokens)
        pred_box, pred_boxes, max_score, response = self._decode_prediction(
            out_dict, resize_factor, return_score=True)
        target_bbox = clip_box(
            self.map_box_back(pred_box, resize_factor, reference_bbox=crop_bbox),
            H, W, margin=10)
        output = {"target_bbox": target_bbox}
        local_record = {
            "template_mid": template_mid.detach(),
            "template_high": template_high.detach(),
            "mid_search": mid_search.detach(),
            "high_search": high_search.detach(),
            "response_map": out_dict["score_map"].detach(),
            "confidence_uncertainty": torch.cat(
                (
                    out_dict["score_map"].detach(),
                    torch.full_like(out_dict["score_map"].detach(), 0.5),
                ),
                dim=1,
            ),
            "target_prototype": high_search.detach().mean(dim=1),
            "local_output": out_dict,
        }
        return {
            "output": output,
            "target_bbox": target_bbox,
            "prev_bbox": list(self.state) if self.state is not None else None,
            "crop_bbox": list(crop_bbox),
            "resize_factor": float(resize_factor),
            "search_factor": float(search_factor),
            "max_score": max_score,
            "apce": self.calAPCE(response),
            "response": response,
            "out_dict": out_dict,
            "pred_boxes": pred_boxes,
            "x_patch_arr": x_patch_arr,
            "image": image,
            "local_record": local_record,
            "_fcvc_provenance": "local",
        }

    def fcvc_sender_bundle(self, local_candidate, view_id, frame_id):
        local = local_candidate["local_record"]
        return build_sender_bundle(
            local["mid_search"],
            local["high_search"],
            local["response_map"],
            local_bbox=torch.tensor(
                [local_candidate["target_bbox"]],
                device=local["high_search"].device,
                dtype=torch.float32,
            ),
            view_id=torch.full(
                (1,), int(view_id), device=local["high_search"].device,
                dtype=torch.int16),
            timestamp=torch.full(
                (1,), int(frame_id), device=local["high_search"].device,
                dtype=torch.int64),
        )

    def fcvc_collaborative_candidate(self, local_candidate, sender_bundles):
        if not self.fcvc_enabled or self.fcvc_model is None:
            raise RuntimeError("fcvc_collaborative_candidate requires loaded FCVC sidecar")
        image = local_candidate["image"]
        H, W = image.shape[:2]
        with torch.no_grad():
            fcvc_out = self.fcvc_model(
                local_candidate["local_record"],
                tuple(sender_bundles),
                forward_head=self.network.forward_head,
            )
        pred_box, pred_boxes, max_score, response = self._decode_prediction(
            fcvc_out["reported_output"],
            local_candidate["resize_factor"],
            return_score=True,
        )
        target_bbox = clip_box(
            self.map_box_back(
                pred_box,
                local_candidate["resize_factor"],
                reference_bbox=local_candidate["crop_bbox"],
            ),
            H,
            W,
            margin=10,
        )
        candidate = dict(local_candidate)
        candidate.update({
            "output": {"target_bbox": target_bbox},
            "target_bbox": target_bbox,
            "max_score": max_score,
            "apce": self.calAPCE(response),
            "response": response,
            "out_dict": fcvc_out["reported_output"],
            "pred_boxes": pred_boxes,
            "used_remote": bool(fcvc_out.get("used_remote", False)),
            "fcvc_diagnostics": {
                "used_remote": bool(fcvc_out.get("used_remote", False)),
                "reason": fcvc_out.get("reason", "ok"),
            },
            "_fcvc_provenance": "collaborative",
        })
        return candidate

    def fcvc_finalize_frame(self, local_candidate, collaborative_candidate,
                            info=None, debug_name=""):
        frame_result = FrameTrackingResult(
            local_candidate=self._fcvc_mark_candidate(local_candidate, "local"),
            collaborative_candidate=self._fcvc_mark_candidate(
                collaborative_candidate, "collaborative"),
            state_output=self._fcvc_mark_candidate(local_candidate, "local"),
            reported_output=self._fcvc_mark_candidate(
                collaborative_candidate, "collaborative"),
            local_runtime_payload=self.fcvc_freeze_local_runtime_payload(
                local_candidate),
            local_diagnostics={},
            collaborative_diagnostics={
                "report_fallback": False,
                "fallback_reason": "ok",
            },
        )
        output = self.fcvc_commit_frame_result(
            frame_result, info=info, debug_name=debug_name)
        return output, collaborative_candidate["max_score"], collaborative_candidate["apce"]

    def get_redetection_context(self):
        """Return prediction-only MCR context without exposing annotations."""
        if self.mcr_manager is None:
            return None
        return {
            "current_bbox": list(self.state) if self.state is not None else None,
            "motion_center": self.mcr_manager.motion.predicted_center,
            "last_reliable_center": self.mcr_manager.motion.last_reliable_center,
            "updates_allowed": self.mcr_manager.switcher.updates_allowed,
        }

    def run_redetection_candidate(
        self,
        image,
        center,
        scale,
        anchor_type,
        reference_bbox,
        remote_prompts=None,
        remote_states=None,
    ):
        """Run a side-effect-free local enlarged search around ``center``."""
        reference_bbox = list(reference_bbox)
        search_bbox = list(reference_bbox)
        search_bbox[0] = float(center[0]) - 0.5 * float(search_bbox[2])
        search_bbox[1] = float(center[1]) - 0.5 * float(search_bbox[3])
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state(self.output_window.device)
        saved_diagnostic = self.last_pcum_diagnostic
        saved_alignment = copy.deepcopy(self._pcum_diagnostic_hooks.alignment)
        saved_fusion = copy.deepcopy(self._pcum_diagnostic_hooks.fusion)
        try:
            raw = self._run_candidate(
                image=image,
                search_factor=float(self.params.search_factor) * float(scale),
                remote_prompts=remote_prompts,
                remote_states=remote_states,
                return_score=True,
                reference_bbox=search_bbox,
            )
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, self.output_window.device)
            self.last_pcum_diagnostic = saved_diagnostic
            self._pcum_diagnostic_hooks.alignment = saved_alignment
            self._pcum_diagnostic_hooks.fusion = saved_fusion

        remote_quality, _, _, _ = self._motion_remote_diagnostics(raw)
        if remote_quality is None:
            quality = raw.get("remote_quality", None)
            if torch.is_tensor(quality) and quality.numel() > 0:
                remote_quality = float(quality.detach().float().mean().cpu().item())
        return RedetectionCandidate(
            bbox=list(raw["target_bbox"]),
            visual_score=max(0.0, min(1.0, self._score_value(raw["max_score"]))),
            apce=self._score_value(raw["apce"]),
            response_entropy=normalized_response_entropy(raw["response"]),
            anchor_type=str(anchor_type),
            scale=float(scale),
            remote_score=remote_quality,
            remote_diagnostics=raw.get("remote_aggregation_diagnostics", None),
            feature=raw.get("local_prompt", None),
            search_region={
                "center": [float(center[0]), float(center[1])],
                "crop_bbox": search_bbox,
                "search_factor": float(self.params.search_factor) * float(scale),
                "resize_factor": raw.get("resize_factor"),
            },
            source=raw,
        )

    def apply_confirmed_redetection(self, main_candidate, mcr_result):
        """Return the candidate to commit after an optional active MCR switch."""
        self.mcr_updates_allowed = bool(mcr_result.get("updates_allowed", True))
        if not mcr_result.get("switched", False):
            return main_candidate
        selected = mcr_result.get("candidate", None)
        if selected is None or selected.source is None:
            return main_candidate
        return selected.source

    def _apply_mcr(self, image, candidate, remote_prompts=None, remote_states=None):
        if not self.mcr_enabled:
            return candidate, None
        remote_quality, _, _, _ = self._motion_remote_diagnostics(candidate)
        result = self.mcr_manager.process(
            frame_id=self.frame_id,
            current_bbox=candidate["target_bbox"],
            current_visual_score=max(0.0, min(1.0, self._score_value(candidate["max_score"]))),
            current_apce=self._score_value(candidate["apce"]),
            image_size=image.shape[:2],
            current_remote_score=remote_quality,
            current_remote_diagnostics=candidate.get("remote_aggregation_diagnostics", None),
            search_callback=lambda center, scale, anchor_type, reference_bbox: (
                self.run_redetection_candidate(
                    image=image,
                    center=center,
                    scale=scale,
                    anchor_type=anchor_type,
                    reference_bbox=reference_bbox,
                    remote_prompts=remote_prompts,
                    remote_states=remote_states,
                )
            ),
        )
        diagnostic = result["diagnostic"]
        if diagnostic is not None:
            diagnostic["tracker_parameter_name"] = self.mcr_config_name
        return self.apply_confirmed_redetection(candidate, result), diagnostic

    def _commit_candidate(self, candidate, info=None, debug_name=""):
        return self._commit_state_from_candidate(
            candidate, info=info, debug_name=debug_name)["output"]

    def _commit_state_from_candidate(self, candidate, info=None, debug_name=""):
        self.state = candidate["target_bbox"]

        if self.debug:
            self._debug_vis(
                image=candidate.get("image", None),
                info=info,
                x_patch_arr=candidate["x_patch_arr"],
                pred_score_map=candidate["out_dict"]["score_map"],
                response=candidate["response"],
                out_dict=candidate["out_dict"],
                debug_name=debug_name
            )

        return {
            "output": candidate["output"],
            "state_output": candidate,
        }

    @staticmethod
    def _fcvc_clone_runtime_value(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                copy.deepcopy(key): EnTeRTrack._fcvc_clone_runtime_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [EnTeRTrack._fcvc_clone_runtime_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(EnTeRTrack._fcvc_clone_runtime_value(item) for item in value)
        try:
            return copy.deepcopy(value)
        except Exception:
            return repr(value)

    @staticmethod
    def _fcvc_canonicalize_for_digest(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            return {
                "kind": "tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            }
        if hasattr(value, "tensors") and torch.is_tensor(value.tensors):
            return {
                "kind": type(value).__name__,
                "tensors": EnTeRTrack._fcvc_canonicalize_for_digest(value.tensors),
            }
        if hasattr(value, "__dict__") and not isinstance(value, type):
            public = {
                key: item for key, item in vars(value).items()
                if not key.startswith("_")
                and key not in ("logger", "visdom", "writer")
            }
            return EnTeRTrack._fcvc_canonicalize_for_digest(public)
        if isinstance(value, dict):
            return {
                str(key): EnTeRTrack._fcvc_canonicalize_for_digest(value[key])
                for key in sorted(value, key=lambda item: str(item))
            }
        if isinstance(value, (list, tuple)):
            return [EnTeRTrack._fcvc_canonicalize_for_digest(item)
                    for item in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            if isinstance(value, float) and not math.isfinite(value):
                return {"kind": "nonfinite", "repr": repr(value)}
            return value
        return repr(value)

    def fcvc_persistent_state_snapshot(self):
        """Return the registered semantic runtime state for Safe Commit audit."""
        snapshot = {
            "state": self._fcvc_clone_runtime_value(getattr(self, "state", None)),
            "frame_id": int(getattr(self, "frame_id", 0)),
            "z_dict1": self._fcvc_clone_runtime_value(getattr(self, "z_dict1", None)),
            "z_patch_arr": self._fcvc_clone_runtime_value(getattr(self, "z_patch_arr", None)),
            "box_mask_z": self._fcvc_clone_runtime_value(getattr(self, "box_mask_z", None)),
            "last_pcum_diagnostic": self._fcvc_clone_runtime_value(
                getattr(self, "last_pcum_diagnostic", None)),
            "c3r_last_frame_by_sender": self._fcvc_clone_runtime_value(
                getattr(self, "c3r_last_frame_by_sender", {})),
            "mcr_updates_allowed": bool(getattr(self, "mcr_updates_allowed", True)),
            "output_window": self._fcvc_clone_runtime_value(
                getattr(self, "output_window", None)),
        }
        for name in (
                "c3r_message_accounting", "temporal_gate_runtime",
                "mcr_manager", "motion_state_manager",
                "_pcum_diagnostic_hooks"):
            snapshot[name] = self._fcvc_clone_runtime_value(
                getattr(self, name, None))
        return snapshot

    def fcvc_persistent_state_digest(self):
        payload = self._fcvc_canonicalize_for_digest(
            self.fcvc_persistent_state_snapshot())
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def fcvc_next_crop_digest(self):
        crop_source = {
            "state": self._fcvc_clone_runtime_value(getattr(self, "state", None)),
            "search_factor": float(getattr(self.params, "search_factor", 0.0))
                if hasattr(self, "params") else 0.0,
            "search_size": int(getattr(self.params, "search_size", 0))
                if hasattr(self, "params") else 0,
        }
        encoded = json.dumps(
            self._fcvc_canonicalize_for_digest(crop_source),
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def fcvc_sender_source_digest(self, local_runtime_payload):
        fields = {
            key: local_runtime_payload.get(key)
            for key in (
                "target_bbox", "max_score", "apce", "response",
                "out_dict", "local_prompt", "crop_bbox", "resize_factor",
                "frame_id", "provenance")
        }
        encoded = json.dumps(
            self._fcvc_canonicalize_for_digest(fields),
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def fcvc_freeze_local_runtime_payload(self, local_candidate):
        payload = {
            "provenance": "local",
            "frame_id": int(getattr(self, "frame_id", 0)),
            "target_bbox": local_candidate.get("target_bbox"),
            "output": local_candidate.get("output"),
            "max_score": local_candidate.get("max_score"),
            "apce": local_candidate.get("apce"),
            "response": local_candidate.get("response"),
            "out_dict": local_candidate.get("out_dict"),
            "pred_boxes": local_candidate.get("pred_boxes"),
            "prev_bbox": local_candidate.get("prev_bbox"),
            "crop_bbox": local_candidate.get("crop_bbox"),
            "resize_factor": local_candidate.get("resize_factor"),
            "search_factor": local_candidate.get("search_factor"),
            "x_patch_arr": local_candidate.get("x_patch_arr"),
            "local_prompt": local_candidate.get("local_prompt"),
            "aligned_prompt": local_candidate.get("aligned_prompt"),
            "align_gate": local_candidate.get("align_gate"),
            "remote_states": None,
            "template_update_decision": local_candidate.get(
                "template_update_decision", None),
            "memory_update_decision": local_candidate.get(
                "memory_update_decision", None),
            "motion_update": {
                "bbox": local_candidate.get("target_bbox"),
                "max_score": local_candidate.get("max_score"),
                "apce": local_candidate.get("apce"),
            },
            "sender_bundle_source": "local",
            "state_pre_digest": self.fcvc_persistent_state_digest(),
            "next_crop_pre_digest": self.fcvc_next_crop_digest(),
        }
        frozen = self._fcvc_clone_runtime_value(payload)
        frozen["sender_source_digest"] = self.fcvc_sender_source_digest(frozen)
        return frozen

    @staticmethod
    def _fcvc_mark_candidate(candidate, provenance):
        marked = EnTeRTrack._fcvc_clone_runtime_value(candidate)
        marked["_fcvc_provenance"] = str(provenance)
        return marked

    @staticmethod
    def _fcvc_candidate_is_valid_for_report(candidate):
        try:
            bbox = candidate["target_bbox"]
            if len(bbox) != 4:
                return False, "bbox length"
            values = [float(item) for item in bbox]
            if not all(math.isfinite(item) for item in values):
                return False, "nonfinite bbox"
            if values[2] <= 0.0 or values[3] <= 0.0:
                return False, "nonpositive bbox size"
            for key in ("max_score", "apce"):
                if key in candidate:
                    value = candidate[key]
                    if torch.is_tensor(value):
                        if not torch.isfinite(value).all():
                            return False, "nonfinite {}".format(key)
                    elif not math.isfinite(float(value)):
                        return False, "nonfinite {}".format(key)
            return True, "ok"
        except Exception as exc:
            return False, "invalid candidate: {}".format(exc)

    def fcvc_predict_frame(self, image=None, info=None, local_candidate=None,
                           collaborative_candidate=None, collaborative_fn=None,
                           search_factor=None, debug_assertions=False):
        """Create local/collaborative candidates without committing state."""
        pre_digest = self.fcvc_persistent_state_digest()
        if local_candidate is None:
            local_candidate = self._run_candidate(
                image=image,
                search_factor=search_factor,
                return_score=True,
            )
        local_candidate = self._fcvc_mark_candidate(local_candidate, "local")
        local_payload = self.fcvc_freeze_local_runtime_payload(local_candidate)
        if debug_assertions:
            if local_payload.get("provenance") != "local":
                raise AssertionError("local runtime payload must be local")
            if self.fcvc_persistent_state_digest() != pre_digest:
                raise AssertionError("predict_frame must not commit state")

        if collaborative_candidate is None and collaborative_fn is not None:
            collaborative_candidate = collaborative_fn(local_payload)
        if collaborative_candidate is None:
            collaborative_candidate = local_candidate
        collaborative_candidate = self._fcvc_mark_candidate(
            collaborative_candidate,
            "collaborative"
            if collaborative_candidate is not local_candidate else "local")
        valid, reason = self._fcvc_candidate_is_valid_for_report(
            collaborative_candidate)
        if not valid:
            collaborative_candidate = self._fcvc_mark_candidate(
                local_candidate, "local")
            collaborative_diagnostics = {
                "report_fallback": True,
                "fallback_reason": reason,
            }
        else:
            collaborative_diagnostics = {
                "report_fallback": False,
                "fallback_reason": "ok",
            }
        result = FrameTrackingResult(
            local_candidate=local_candidate,
            collaborative_candidate=collaborative_candidate,
            state_output=local_candidate,
            reported_output=collaborative_candidate,
            local_runtime_payload=local_payload,
            local_diagnostics={
                "state_pre_digest": pre_digest,
                "next_crop_pre_digest": local_payload["next_crop_pre_digest"],
                "sender_source_digest": local_payload["sender_source_digest"],
            },
            collaborative_diagnostics=collaborative_diagnostics,
        )
        if debug_assertions:
            if result.state_output.get("_fcvc_provenance") != "local":
                raise AssertionError("state_output provenance must be local")
            if result.reported_output.get("_fcvc_provenance") not in (
                    "collaborative", "local"):
                raise AssertionError("reported_output provenance invalid")
        return result

    def fcvc_commit_state(self, state_output, local_runtime_payload,
                          info=None, debug_name="", debug_assertions=False):
        """Commit only the local state_output and local runtime payload."""
        if debug_assertions:
            if state_output.get("_fcvc_provenance") != "local":
                raise AssertionError(
                    "FCVC commit_state refuses collaborative provenance")
            if local_runtime_payload.get("provenance") != "local":
                raise AssertionError("local payload provenance must be local")
        committed = self._commit_state_from_candidate(
            state_output, info=info, debug_name=debug_name)
        return committed["output"]

    def fcvc_emit_report(self, reported_output, debug_assertions=False):
        """Return the report candidate output without mutating state."""
        before = self.fcvc_persistent_state_digest() if debug_assertions else None
        valid, reason = self._fcvc_candidate_is_valid_for_report(reported_output)
        if not valid:
            raise ValueError("invalid FCVC reported output: {}".format(reason))
        output = self._fcvc_clone_runtime_value(reported_output["output"])
        if debug_assertions and self.fcvc_persistent_state_digest() != before:
            raise AssertionError("emit_report mutated persistent state")
        return output

    def fcvc_commit_frame_result(self, frame_result, info=None, debug_name="",
                                 debug_assertions=False):
        """Commit local state and emit collaborative report under Safe Commit."""
        pre_frame_digest = self.fcvc_persistent_state_digest()
        report_candidate = frame_result.reported_output
        valid, reason = self._fcvc_candidate_is_valid_for_report(report_candidate)
        if not valid:
            report_candidate = frame_result.local_candidate
            report_fallback_reason = reason
        else:
            report_fallback_reason = frame_result.collaborative_diagnostics.get(
                "fallback_reason", "ok")

        self.fcvc_commit_state(
            frame_result.state_output,
            frame_result.local_runtime_payload,
            info=info,
            debug_name=debug_name,
            debug_assertions=debug_assertions,
        )
        reported = self._fcvc_clone_runtime_value(report_candidate["output"])
        reported = self._attach_motion_shadow_diagnostics(
            frame_result.state_output, reported)
        reported["fcvc_safe_commit"] = {
            "state_output_provenance": frame_result.state_output.get(
                "_fcvc_provenance"),
            "reported_output_provenance": report_candidate.get(
                "_fcvc_provenance"),
            "pre_frame_digest": pre_frame_digest,
            "post_commit_digest": self.fcvc_persistent_state_digest(),
            "next_crop_digest": self.fcvc_next_crop_digest(),
            "sender_source_digest": frame_result.local_runtime_payload[
                "sender_source_digest"],
            "report_fallback_reason": report_fallback_reason,
        }
        after_report_digest = self.fcvc_persistent_state_digest()
        if debug_assertions:
            if reported["fcvc_safe_commit"]["post_commit_digest"] != after_report_digest:
                raise AssertionError("reported output changed persistent state")
            if self.state != frame_result.state_output["target_bbox"]:
                raise AssertionError("next crop state must come from local")
            if frame_result.local_runtime_payload.get("sender_bundle_source") != "local":
                raise AssertionError("sender bundle source must be local")
        return reported

    def _motion_remote_diagnostics(self, candidate):
        diagnostics = candidate.get("remote_aggregation_diagnostics", None)
        if not isinstance(diagnostics, dict):
            return None, None, None, None

        def scalar(name):
            value = diagnostics.get(name, None)
            if value is None:
                return None
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return None
                value = value.detach().reshape(-1)[0].cpu().item()
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        return (
            scalar("remote_quality_mean"),
            scalar("remote_weight_entropy"),
            scalar("remote_weight_max"),
            scalar("valid_remote_count"),
        )

    def _attach_motion_shadow_diagnostics(self, candidate, output):
        """Update M0 diagnostics without changing any tracking value."""
        if self.motion_state_manager is None:
            return output
        remote_quality, remote_entropy, remote_max, valid_remote = (
            self._motion_remote_diagnostics(candidate)
        )
        image = candidate.get("image", None)
        image_size = image.shape[:2] if image is not None else None
        record = self.motion_state_manager.update_prediction_only(
            frame_id=self.frame_id,
            predicted_bbox=candidate["target_bbox"],
            max_score=candidate.get("max_score", None),
            apce=candidate.get("apce", None),
            response=candidate.get("response", None),
            image_size=image_size,
            remote_quality=remote_quality,
            remote_weight_entropy=remote_entropy,
            remote_max_weight=remote_max,
            valid_remote_count=valid_remote,
        )
        if self.motion_state_log_enabled:
            output["motion_state_diagnostics"] = record
        return output

    def _score_value(self, score):
        return score.item() if torch.is_tensor(score) else float(score)

    def _candidate_confidence(self, candidate):
        if candidate is None:
            return 0.0
        pcum_test_cfg = getattr(self.cfg.TEST, "PCUM", None)
        apce_norm = max(
            float(getattr(pcum_test_cfg, "MOTION_REDETECT_APCE_NORM", 200.0)),
            1e-6
        )
        score = max(0.0, min(1.0, self._score_value(candidate["max_score"])))
        apce = max(0.0, min(1.0, self._score_value(candidate["apce"]) / apce_norm))
        return score * apce

    def _candidate_motion_reliability(self, candidate):
        if candidate is None:
            return 0.0
        prev_bbox = candidate.get("prev_bbox", None)
        target_bbox = candidate.get("target_bbox", None)
        if prev_bbox is None or target_bbox is None:
            motion_score = 1.0
        else:
            try:
                px = float(prev_bbox[0]) + 0.5 * float(prev_bbox[2])
                py = float(prev_bbox[1]) + 0.5 * float(prev_bbox[3])
                cx = float(target_bbox[0]) + 0.5 * float(target_bbox[2])
                cy = float(target_bbox[1]) + 0.5 * float(target_bbox[3])
                scale = max(
                    (float(prev_bbox[2]) * float(prev_bbox[3])) ** 0.5,
                    1.0,
                )
                normalized_motion = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / scale
            except Exception:
                normalized_motion = 0.0
            pcum_test_cfg = getattr(self.cfg.TEST, "PCUM", None)
            max_norm_motion = max(float(getattr(
                pcum_test_cfg,
                "MOTION_REDETECT_MAX_NORM_MOTION",
                2.0,
            )), 1e-6)
            motion_score = max(0.0, min(1.0, 1.0 - normalized_motion / max_norm_motion))

        return self._candidate_confidence(candidate) * motion_score

    def _selector_record(
        self,
        selector_mode,
        decision=None,
        num_remote_prompts=0,
        selected_source=0.0,
    ):
        if decision is None:
            decision = {
                "use_collaborative": False,
                "local_confidence": float("nan"),
                "collaborative_confidence": float("nan"),
                "confidence_delta": float("nan"),
                "collaborative_motion_reliability": float("nan"),
                "margin": float("nan"),
                "motion_threshold": float("nan"),
            }
        selector_enabled = 1.0 if selector_mode == "deterministic" else 0.0
        return [
            selector_enabled,
            1.0 if decision.get("use_collaborative", False) else 0.0,
            float(decision.get("local_confidence", float("nan"))),
            float(decision.get("collaborative_confidence", float("nan"))),
            float(decision.get("confidence_delta", float("nan"))),
            float(decision.get("collaborative_motion_reliability", float("nan"))),
            float(decision.get("margin", float("nan"))),
            float(decision.get("motion_threshold", float("nan"))),
            float(num_remote_prompts),
            float(selected_source),
        ]

    def _remote_aggregation_record(self, candidate):
        if candidate is None or not self.remote_weight_diagnostics_enabled:
            return None
        diagnostics = candidate.get("remote_aggregation_diagnostics", None)
        weights = candidate.get("remote_weights", None)
        if not isinstance(diagnostics, dict) or not torch.is_tensor(weights):
            return None

        weights = weights.detach().float().reshape(weights.shape[0], -1)[0].cpu()
        state = candidate.get("remote_states", None)
        uav_indices = None
        if isinstance(state, dict):
            uav_indices = state.get("per_remote_uav_indices", None)
        if torch.is_tensor(uav_indices):
            uav_indices = uav_indices.detach().reshape(uav_indices.shape[0], -1)[0].cpu()
        else:
            uav_indices = torch.arange(weights.numel(), dtype=torch.long)

        global_weights = torch.zeros(3, dtype=torch.float32)
        for slot, weight in enumerate(weights):
            if slot >= uav_indices.numel():
                break
            index = int(uav_indices[slot].item())
            if 0 <= index < global_weights.numel():
                global_weights[index] = float(weight.item())

        def scalar(name, default=float("nan")):
            value = diagnostics.get(name, None)
            if value is None:
                return float(default)
            if torch.is_tensor(value):
                value = value.detach().reshape(-1)[0].cpu().item()
            return float(value)

        selected_slot = int(scalar("selected_remote_index", 0.0))
        selected_uav = (
            int(uav_indices[selected_slot].item())
            if 0 <= selected_slot < uav_indices.numel() else -1
        )
        return [
            scalar("remote_weight_entropy"),
            scalar("remote_weight_max"),
            scalar("remote_weight_mean"),
            float(selected_uav),
            scalar("valid_remote_count"),
            scalar("remote_quality_mean"),
            scalar("remote_quality_min"),
            scalar("remote_quality_max"),
            scalar("fallback_to_uniform", 0.0),
            float(global_weights[0].item()),
            float(global_weights[1].item()),
            float(global_weights[2].item()),
        ]

    def _remote_suppression_record(self, candidate):
        if candidate is None:
            return None

        def scalar(name, default=float("nan")):
            value = candidate.get(name, None)
            if value is None:
                return float(default)
            if torch.is_tensor(value):
                value = value.detach().reshape(-1)[0].cpu().item()
            return float(value)

        suppress = scalar("remote_suppression")
        if not math.isfinite(suppress):
            return None
        return [
            suppress,
            1.0 - suppress,
            scalar("remote_delta_norm"),
            scalar("suppressed_delta_norm"),
            scalar("remote_suppression_active_ratio", 0.0),
        ]

    def _diagnostic_candidate_snapshot(self, candidate):
        if candidate is None:
            return None
        diagnostics = candidate.get("diagnostics", {})
        return {
            "bbox": list(candidate["target_bbox"]),
            "score_max": self._score_value(candidate["max_score"]),
            "apce": self._score_value(candidate["apce"]),
            "confidence": self._candidate_confidence(candidate),
            "response_entropy": diagnostics.get("response_entropy", float("nan")),
            "alignment_gate_mean": diagnostics.get("alignment_gate_mean", float("nan")),
            "alignment_gate_std": diagnostics.get("alignment_gate_std", float("nan")),
            "fusion_gate_mean": diagnostics.get("fusion_gate_mean", float("nan")),
            "fusion_gate_std": diagnostics.get("fusion_gate_std", float("nan")),
            "fusion_gate_min": diagnostics.get("fusion_gate_min", float("nan")),
            "fusion_gate_max": diagnostics.get("fusion_gate_max", float("nan")),
            "prompt_norm": diagnostics.get("prompt_norm", float("nan")),
            "aligned_prompt_norm": diagnostics.get("aligned_prompt_norm", float("nan")),
        }

    def close_pcum_diagnostics(self):
        self._pcum_diagnostic_hooks.remove()

    def pcum_local_candidate(self, image, search_factor=None):
        return self._run_candidate(
            image=image,
            search_factor=search_factor,
            return_score=True
        )

    def plain_collaboration_local_candidate(self, image):
        """Run exactly one frozen local backbone/head pass for a V1 frame."""
        if not self.plain_collaboration_enabled:
            raise RuntimeError(
                "Plain Collaboration local candidate requested while disabled")
        self._plain_collaboration_local_forward_count = int(getattr(
            self, "_plain_collaboration_local_forward_count", 0)) + 1
        candidate = self._run_candidate(
            image=image,
            search_factor=self.params.search_factor,
            return_score=True,
        )
        feature = candidate["out_dict"].get("backbone_feat", None)
        if not torch.is_tensor(feature) or feature.dim() != 3:
            raise RuntimeError(
                "Plain Collaboration local candidate is missing [B,L,C] feature")
        if feature.shape[0] != 1:
            raise RuntimeError(
                "Plain Collaboration inference requires batch size one per view")
        if feature.shape[1] < int(self.network.feat_len_s):
            raise RuntimeError(
                "Plain Collaboration local feature has too few search tokens")
        if not bool(torch.isfinite(feature).all().item()):
            raise RuntimeError(
                "Plain Collaboration local feature contains non-finite values")
        return candidate

    @staticmethod
    def _plain_scalar(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(
                    "Plain Collaboration diagnostic value must be scalar")
            return float(value.detach().cpu().item())
        return float(value)

    def plain_collaboration_candidate(
            self, local_candidate, remote_candidates, receiver_view,
            sender_views, frame_id, target_id=""):
        """Fuse synchronized remote final-search features and rerun the head."""
        if not self.plain_collaboration_enabled:
            raise RuntimeError(
                "Plain Collaboration candidate requested while disabled")
        remote_candidates = tuple(remote_candidates)
        sender_views = tuple(str(value).upper() for value in sender_views)
        receiver_view = str(receiver_view).upper()
        if receiver_view not in ("A", "B", "C"):
            raise ValueError("Plain Collaboration receiver must be A, B, or C")
        expected_senders = tuple(
            view for view in ("A", "B", "C") if view != receiver_view)
        sender_diagnostic = bool(getattr(
            self,
            "plain_collaboration_sender_counterfactual_diagnostics",
            False,
        ))
        if sender_diagnostic:
            canonical_subset = tuple(
                view for view in expected_senders if view in sender_views)
            if (len(remote_candidates) not in (1, 2)
                    or len(remote_candidates) != len(sender_views)
                    or len(set(sender_views)) != len(sender_views)
                    or sender_views != canonical_subset):
                raise ValueError(
                    "Sender counterfactual requires a canonical one/two-sender "
                    "subset of {}".format(expected_senders))
        elif sender_views != expected_senders or len(remote_candidates) != 2:
            raise ValueError(
                "Plain Collaboration requires canonical two-sender order {}"
                .format(expected_senders))

        feature = local_candidate["out_dict"].get("backbone_feat", None)
        if not torch.is_tensor(feature):
            raise RuntimeError("Local candidate is missing backbone_feat")
        feat_len_s = int(self.network.feat_len_s)
        local_search = feature[:, -feat_len_s:]
        remote_search = []
        for sender_view, candidate in zip(sender_views, remote_candidates):
            remote_feature = candidate["out_dict"].get("backbone_feat", None)
            if not torch.is_tensor(remote_feature):
                raise RuntimeError(
                    "Remote {} candidate is missing backbone_feat".format(
                        sender_view))
            if tuple(remote_feature.shape) != tuple(feature.shape):
                raise RuntimeError(
                    "Remote {} feature shape does not match receiver {}"
                    .format(sender_view, receiver_view))
            remote_search.append(remote_feature[:, -feat_len_s:].detach().to(
                device=local_search.device, dtype=local_search.dtype))
        remote_tokens = torch.stack(remote_search, dim=1)
        remote_valid = torch.isfinite(remote_tokens).flatten(2).all(dim=2)
        if not bool(remote_valid.all().item()):
            raise RuntimeError(
                "Plain Collaboration requires finite remote senders")

        state_digest_before = None
        state_audit_enabled = bool(
            getattr(self, "plain_collaboration_counterfactual_diagnostics", False)
            or sender_diagnostic)
        if state_audit_enabled:
            state_digest_before = self.fcvc_persistent_state_digest()
        with torch.no_grad():
            head_output = self.network(
                template=None,
                search=None,
                training=False,
                collaboration_feature=feature.detach(),
                plain_remote_tokens=remote_tokens,
                plain_remote_valid=remote_valid,
            )
        state_digest_after = None
        if state_audit_enabled:
            state_digest_after = self.fcvc_persistent_state_digest()
            if state_digest_after != state_digest_before:
                raise RuntimeError(
                    "Plain Collaboration forward mutated persistent runtime state")
        collaboration = head_output.get("plain_collaboration", None)
        if not isinstance(collaboration, dict):
            raise RuntimeError(
                "Plain Collaboration head output is missing diagnostics")
        valid_remote_count = collaboration["valid_remote_count"]
        if valid_remote_count.numel() != 1 or int(
                valid_remote_count.detach().cpu().item()) != len(remote_candidates):
            raise RuntimeError(
                "Plain Collaboration remote count did not match the branch")
        if not bool(collaboration.get("used_remote", False)):
            raise RuntimeError("Plain Collaboration adapter bypassed remote input")

        pred_box, pred_boxes, max_score, response = self._decode_prediction(
            head_output,
            local_candidate["resize_factor"],
            return_score=True,
        )
        image = local_candidate["image"]
        height, width = image.shape[:2]
        crop_bbox = local_candidate["crop_bbox"]
        target_bbox = clip_box(
            self.map_box_back(
                pred_box,
                local_candidate["resize_factor"],
                reference_bbox=crop_bbox,
            ),
            height,
            width,
            margin=10,
        )
        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size
                / local_candidate["resize_factor"],
                local_candidate["resize_factor"],
                reference_bbox=crop_bbox,
            )
            output = {
                "target_bbox": target_bbox,
                "all_boxes": all_boxes.view(-1).tolist(),
            }
        else:
            output = {"target_bbox": target_bbox}

        weights = collaboration["remote_weights"].detach().reshape(-1).cpu()
        weight_values = [float(value.item()) for value in weights]
        diagnostics = {
            "frame_id": int(frame_id),
            "receiver_view": receiver_view,
            "sender_view_0": sender_views[0],
            "sender_view_1": sender_views[1] if len(sender_views) > 1 else "",
            "used_remote": True,
            "valid_remote_count": len(remote_candidates),
            "search_token_count": int(local_search.shape[1]),
            "sender_weight_0": weight_values[0],
            "sender_weight_1": (
                weight_values[1] if len(weight_values) > 1 else float("nan")),
            "residual_norm": self._plain_scalar(
                collaboration["residual_norm"]),
            "relative_residual_norm": self._plain_scalar(
                collaboration["relative_residual_norm"]),
            "residual_scale": self._plain_scalar(
                collaboration["residual_scale"]),
        }
        def candidate_metrics(source):
            bbox = [float(value) for value in source["target_bbox"]]
            center_x = bbox[0] + 0.5 * bbox[2]
            center_y = bbox[1] + 0.5 * bbox[3]
            return {
                "bbox": bbox,
                "center_x": center_x,
                "center_y": center_y,
                "width": bbox[2],
                "height": bbox[3],
                "area": bbox[2] * bbox[3],
                "max_score": self._score_value(source["max_score"]),
                "apce": self._score_value(source["apce"]),
                "entropy": normalized_response_entropy(source.get("response")),
            }

        counterfactual = None
        if getattr(self, "plain_collaboration_counterfactual_diagnostics", False):
            local_metrics = candidate_metrics(local_candidate)
            collaborative_metrics = {
                "bbox": [float(value) for value in target_bbox],
                "center_x": float(target_bbox[0] + 0.5 * target_bbox[2]),
                "center_y": float(target_bbox[1] + 0.5 * target_bbox[3]),
                "width": float(target_bbox[2]),
                "height": float(target_bbox[3]),
                "area": float(target_bbox[2] * target_bbox[3]),
                "max_score": self._score_value(max_score),
                "apce": self._score_value(self.calAPCE(response)),
                "entropy": normalized_response_entropy(response),
            }
            sender_metrics = [candidate_metrics(item) for item in remote_candidates]
            previous_bbox = local_candidate.get("prev_bbox")
            if previous_bbox is None:
                previous_center = (local_metrics["center_x"], local_metrics["center_y"])
                previous_area = local_metrics["area"]
            else:
                previous_center = (
                    float(previous_bbox[0] + 0.5 * previous_bbox[2]),
                    float(previous_bbox[1] + 0.5 * previous_bbox[3]),
                )
                previous_area = float(previous_bbox[2] * previous_bbox[3])
            eps = 1e-12
            counterfactual = {
                "frame_id": int(frame_id),
                "target_id": str(target_id),
                "receiver_view": receiver_view,
                "sender_view_0": sender_views[0],
                "sender_view_1": sender_views[1],
                "uses_gt": False,
                "safe_commit": bool(getattr(
                    self, "plain_collaboration_safe_commit", False)),
                "valid_remote_count": 2,
                "search_token_count": int(local_search.shape[1]),
                "sender_weight_0": diagnostics["sender_weight_0"],
                "sender_weight_1": diagnostics["sender_weight_1"],
                "residual_norm": diagnostics["residual_norm"],
                "relative_residual_norm": diagnostics["relative_residual_norm"],
                "residual_scale": diagnostics["residual_scale"],
                "local_center_displacement": math.hypot(
                    local_metrics["center_x"] - previous_center[0],
                    local_metrics["center_y"] - previous_center[1]),
                "local_scale_change": math.log(
                    max(local_metrics["area"], eps) / max(previous_area, eps)),
                "local_collab_center_displacement": math.hypot(
                    collaborative_metrics["center_x"] - local_metrics["center_x"],
                    collaborative_metrics["center_y"] - local_metrics["center_y"]),
                "local_collab_scale_difference": math.log(
                    max(collaborative_metrics["area"], eps)
                    / max(local_metrics["area"], eps)),
                "persistent_state_digest_before": state_digest_before,
                "persistent_state_digest_after": state_digest_after,
            }
            for prefix, values in (
                    ("local", local_metrics),
                    ("collaborative", collaborative_metrics),
                    ("sender_0", sender_metrics[0]),
                    ("sender_1", sender_metrics[1])):
                for key in ("max_score", "apce", "entropy", "center_x",
                            "center_y", "width", "height", "area"):
                    counterfactual["{}_{}".format(prefix, key)] = values[key]
                for index, key in enumerate(("x", "y", "w", "h")):
                    counterfactual["{}_bbox_{}".format(prefix, key)] = values["bbox"][index]
        candidate = dict(local_candidate)
        candidate.update({
            "output": output,
            "target_bbox": target_bbox,
            "max_score": max_score,
            "apce": self.calAPCE(response),
            "response": response,
            "out_dict": head_output,
            "pred_boxes": pred_boxes,
            "used_remote": True,
            "plain_collaboration_diagnostics": diagnostics,
            "_plain_local_candidate": local_candidate,
            "_plain_state_digest_before": state_digest_before,
            "_plain_state_digest_after": state_digest_after,
        })
        if counterfactual is not None:
            candidate["plain_collaboration_counterfactual"] = counterfactual
        return candidate

    @staticmethod
    def _plain_bbox_motion_metrics(candidate):
        bbox = [float(value) for value in candidate["target_bbox"]]
        previous = candidate.get("prev_bbox")
        if previous is None:
            return 0.0, 0.0
        previous = [float(value) for value in previous]
        center_motion = math.hypot(
            bbox[0] + 0.5 * bbox[2] - previous[0] - 0.5 * previous[2],
            bbox[1] + 0.5 * bbox[3] - previous[1] - 0.5 * previous[3],
        )
        current_area = max(bbox[2] * bbox[3], 1e-12)
        previous_area = max(previous[2] * previous[3], 1e-12)
        return center_motion, math.log(current_area / previous_area)

    def plain_collaboration_target_prototype_row(
            self, local_candidate, receiver_view, frame_id, target_id):
        """Extract prediction-only local prototypes without changing runtime state."""
        if not getattr(
                self,
                "plain_collaboration_target_consistency_diagnostics",
                False):
            raise RuntimeError("target consistency diagnostics are disabled")
        receiver_view = str(receiver_view).upper()
        if receiver_view not in ("A", "B", "C"):
            raise ValueError("target prototype view must be A, B, or C")

        state_digest_before = self.fcvc_persistent_state_digest()
        out_dict = local_candidate.get("out_dict", {})
        feature = out_dict.get("backbone_feat")
        score_map = out_dict.get("score_map")
        if not torch.is_tensor(feature) or feature.dim() != 3:
            raise RuntimeError("target prototype requires [B,L,C] backbone_feat")
        if feature.shape[0] != 1:
            raise RuntimeError("target prototype requires batch size one")
        feat_len_s = int(self.network.feat_len_s)
        if feat_len_s != 256:
            raise RuntimeError("target prototype requires exactly 256 search tokens")
        template_len = int(feature.shape[1]) - feat_len_s
        if template_len != 64:
            raise RuntimeError("target prototype requires exactly 64 template tokens")
        expected_dim = int(self.network.plain_collaboration.token_dim)
        if int(feature.shape[2]) != expected_dim:
            raise RuntimeError("target prototype token dimension does not match adapter")
        if not torch.is_tensor(score_map):
            raise RuntimeError("target prototype requires CENTER raw score_map")

        search_tokens = feature[:, -feat_len_s:]
        template_tokens = feature[:, :-feat_len_s]
        raw_response = score_map.reshape(score_map.shape[0], -1)
        if tuple(raw_response.shape) != (1, feat_len_s):
            raise RuntimeError("target prototype score_map/token layout mismatch")
        if not bool(torch.isfinite(search_tokens).all().item()):
            raise RuntimeError("target prototype search tokens are non-finite")
        if not bool(torch.isfinite(template_tokens).all().item()):
            raise RuntimeError("target prototype template tokens are non-finite")
        if not bool(torch.isfinite(raw_response).all().item()):
            raise RuntimeError("target prototype raw response is non-finite")

        with torch.no_grad():
            weights = torch.softmax(raw_response.float(), dim=1).to(
                dtype=search_tokens.dtype)
            weighted = torch.sum(search_tokens * weights.unsqueeze(-1), dim=1)
            global_mean = search_tokens.mean(dim=1)
            template_conditioned = template_tokens.mean(dim=1)
        state_digest_after = self.fcvc_persistent_state_digest()
        if state_digest_after != state_digest_before:
            raise RuntimeError("target prototype extraction mutated persistent state")

        def vector(value):
            return value[0].detach().float().cpu().numpy()

        weighted_np = vector(weighted)
        mean_np = vector(global_mean)
        template_np = vector(template_conditioned)
        bbox = np.asarray(
            [float(value) for value in local_candidate["target_bbox"]],
            dtype=np.float32,
        )
        return {
            "target_id": str(target_id),
            "view_id": receiver_view,
            "frame_id": int(frame_id),
            "uses_gt": False,
            "source_local": True,
            "search_token_count": feat_len_s,
            "template_token_count": template_len,
            "token_dim": expected_dim,
            "temperature": 1.0,
            "target_bbox": bbox,
            "response_weighted": weighted_np,
            "global_mean": mean_np,
            "template_conditioned": template_np,
            "response_weighted_norm": float(np.linalg.norm(weighted_np)),
            "global_mean_norm": float(np.linalg.norm(mean_np)),
            "template_conditioned_norm": float(np.linalg.norm(template_np)),
            "persistent_state_digest_before": state_digest_before,
            "persistent_state_digest_after": state_digest_after,
        }

    def plain_collaboration_sender_counterfactual_row(
            self, local_candidate, branch_candidate, remote_candidates,
            receiver_view, sender_views, branch_name, frame_id, target_id,
            state_digest_before):
        """Build one prediction-only branch row; never reads annotations."""
        if not getattr(
                self,
                "plain_collaboration_sender_counterfactual_diagnostics",
                False):
            raise RuntimeError("sender counterfactual diagnostics are disabled")
        sender_views = tuple(sender_views)
        remote_candidates = tuple(remote_candidates)
        if len(sender_views) != len(remote_candidates):
            raise ValueError("sender view/candidate counts differ")

        def scalar_metrics(candidate):
            bbox = [float(value) for value in candidate["target_bbox"]]
            return {
                "bbox": bbox,
                "max_score": self._score_value(candidate["max_score"]),
                "apce": self._score_value(candidate["apce"]),
                "entropy": normalized_response_entropy(candidate.get("response")),
                "area": max(bbox[2] * bbox[3], 1e-12),
                "center": (bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]),
            }

        local = scalar_metrics(local_candidate)
        branch = scalar_metrics(branch_candidate)
        local_motion, local_scale = self._plain_bbox_motion_metrics(
            local_candidate)
        diagnostics = branch_candidate.get("plain_collaboration_diagnostics", {})
        row = {
            "target_id": str(target_id),
            "frame_id": int(frame_id),
            "receiver_view": str(receiver_view),
            "branch_name": str(branch_name),
            "sender_views": "|".join(sender_views),
            "uses_gt": False,
            "safe_commit": True,
            "search_token_count": int(self.network.feat_len_s),
            "remote_count": len(remote_candidates),
            "local_max_score": local["max_score"],
            "local_apce": local["apce"],
            "local_entropy": local["entropy"],
            "local_center_motion": local_motion,
            "local_scale_change": local_scale,
            "branch_max_score": branch["max_score"],
            "branch_apce": branch["apce"],
            "branch_entropy": branch["entropy"],
            "center_displacement": math.hypot(
                branch["center"][0] - local["center"][0],
                branch["center"][1] - local["center"][1]),
            "scale_difference": math.log(branch["area"] / local["area"]),
            "score_delta": branch["max_score"] - local["max_score"],
            "apce_delta": branch["apce"] - local["apce"],
            "residual_norm": float(diagnostics.get("residual_norm", 0.0)),
            "relative_residual_norm": float(
                diagnostics.get("relative_residual_norm", 0.0)),
            "residual_scale": float(diagnostics.get("residual_scale", 0.0)),
            "sender_weight_0": float(
                diagnostics.get("sender_weight_0", 0.0)),
            "sender_weight_1": float(
                diagnostics.get("sender_weight_1", 0.0)),
            "persistent_state_digest_before": state_digest_before,
            "persistent_state_digest_after": (
                branch_candidate.get("_plain_state_digest_after")
                if remote_candidates else state_digest_before),
        }
        for prefix, values in (("local", local), ("branch", branch)):
            for index, key in enumerate(("x", "y", "w", "h")):
                row["{}_bbox_{}".format(prefix, key)] = values["bbox"][index]
        for slot in range(2):
            prefix = "sender_{}".format(slot)
            if slot < len(remote_candidates):
                candidate = remote_candidates[slot]
                values = scalar_metrics(candidate)
                motion, scale = self._plain_bbox_motion_metrics(candidate)
                row.update({
                    prefix + "_view": sender_views[slot],
                    prefix + "_max_score": values["max_score"],
                    prefix + "_apce": values["apce"],
                    prefix + "_entropy": values["entropy"],
                    prefix + "_bbox_motion": motion,
                    prefix + "_scale_change": scale,
                })
            else:
                row.update({
                    prefix + "_view": "",
                    prefix + "_max_score": float("nan"),
                    prefix + "_apce": float("nan"),
                    prefix + "_entropy": float("nan"),
                    prefix + "_bbox_motion": float("nan"),
                    prefix + "_scale_change": float("nan"),
                })
        return row

    def plain_collaboration_finalize_frame(
            self, local_candidate, collaborative_candidate=None,
            info=None, debug_name=""):
        """Commit local or collaborative state and always report collaboration."""
        if not self.plain_collaboration_enabled:
            raise RuntimeError(
                "Plain Collaboration finalize requested while disabled")
        if collaborative_candidate is None:
            collaborative_candidate = local_candidate
            local_candidate = collaborative_candidate.get(
                "_plain_local_candidate", collaborative_candidate)
        self.frame_id += 1
        if getattr(self, "plain_collaboration_safe_commit", False):
            self._commit_state_from_candidate(
                local_candidate, info=info, debug_name=debug_name)
            output = copy.deepcopy(collaborative_candidate["output"])
        else:
            output = self._commit_candidate(
                collaborative_candidate, info=info, debug_name=debug_name)
        output["plain_collaboration_diagnostics"] = dict(
            collaborative_candidate["plain_collaboration_diagnostics"])
        if "plain_collaboration_counterfactual" in collaborative_candidate:
            row = dict(collaborative_candidate[
                "plain_collaboration_counterfactual"])
            row["state_output_bbox_x"] = float(self.state[0])
            row["state_output_bbox_y"] = float(self.state[1])
            row["state_output_bbox_w"] = float(self.state[2])
            row["state_output_bbox_h"] = float(self.state[3])
            row["reported_output_bbox_x"] = float(
                collaborative_candidate["target_bbox"][0])
            row["reported_output_bbox_y"] = float(
                collaborative_candidate["target_bbox"][1])
            row["reported_output_bbox_w"] = float(
                collaborative_candidate["target_bbox"][2])
            row["reported_output_bbox_h"] = float(
                collaborative_candidate["target_bbox"][3])
            row["next_crop_state_digest"] = self.fcvc_next_crop_digest()
            output["plain_collaboration_counterfactual"] = row
        return (
            output,
            collaborative_candidate["max_score"],
            collaborative_candidate["apce"],
        )

    def c3r_local_candidate(self, image):
        """Run and retain exactly one ordinary local EnTeR candidate."""
        if not self.c3r_enabled:
            raise RuntimeError("C3R local candidate requested while disabled")
        return self._run_candidate(
            image=image,
            search_factor=None,
            return_score=True,
        )

    def c3r_build_packet(self, candidate, target_id, sender_id,
                         frame_id, timestamp_ms):
        """Serialize one prediction-only local packet for orchestration."""
        image = candidate.get("image", None)
        if image is None:
            raise ValueError("C3R candidate is missing its local image shape")
        height, width = image.shape[:2]
        return build_packet_record(
            c3r=self.network.c3r,
            feat_len_s=self.network.feat_len_s,
            candidate=candidate,
            target_id=target_id,
            sender_id=sender_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            image_height=height,
            image_width=width,
        )

    def _c3r_candidate_from_head(self, local_candidate, head_output):
        """Decode a C3R head rerun using the unchanged local crop/state."""
        pred_box, pred_boxes, max_score, response = self._decode_prediction(
            head_output,
            local_candidate["resize_factor"],
            return_score=True,
        )
        image = local_candidate["image"]
        height, width = image.shape[:2]
        crop_bbox = local_candidate["crop_bbox"]
        target_bbox = clip_box(
            self.map_box_back(
                pred_box,
                local_candidate["resize_factor"],
                reference_bbox=crop_bbox,
            ),
            height,
            width,
            margin=10,
        )
        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size
                / local_candidate["resize_factor"],
                local_candidate["resize_factor"],
                reference_bbox=crop_bbox,
            )
            output = {
                "target_bbox": target_bbox,
                "all_boxes": all_boxes.view(-1).tolist(),
            }
        else:
            output = {"target_bbox": target_bbox}
        candidate = dict(local_candidate)
        candidate.update({
            "output": output,
            "target_bbox": target_bbox,
            "max_score": max_score,
            "apce": self.calAPCE(response),
            "response": response,
            "out_dict": head_output,
            "pred_boxes": pred_boxes,
            "used_remote": True,
        })
        return candidate

    def c3r_counterfactual_candidates(self, local_candidate, packets,
                                      context: C3RReceiverContext):
        """Head-only fixed-gate branches that never commit tracker state."""
        if not self.c3r_enabled:
            raise RuntimeError("C3R counterfactual requested while disabled")

        diagnostics_enabled = bool(
            self.temporal_gate_counterfactual_diagnostics)
        shared_pre_state_digest = self._counterfactual_state_digest()
        backbone_count_before = int(
            self._temporal_gate_backbone_forward_count)
        branch_traces = {
            "local": {
                "pre_state_digest": shared_pre_state_digest,
                "post_state_digest": shared_pre_state_digest,
                "executed_gates": [],
                "sender_provenance": [],
            }
        }

        def branch(selected_packets, branch_name):
            pre_state_digest = self._counterfactual_state_digest()
            if diagnostics_enabled and pre_state_digest != shared_pre_state_digest:
                raise AssertionError(
                    "counterfactual branches must share one pre-state digest")
            branch_context = C3RReceiverContext(
                target_id=context.target_id,
                receiver_id=context.receiver_id,
                sequence_hash=context.sequence_hash,
                frame_id=context.frame_id,
                timestamp_ms=context.timestamp_ms,
                frame_interval_ms=context.frame_interval_ms,
                last_frame_by_sender=dict(context.last_frame_by_sender),
            )
            provenance = [self._counterfactual_packet_provenance(packet)
                          for packet in selected_packets]
            executed_gates = []

            def fixed_gate(sender_id, vector):
                gate = vector.new_tensor(0.25)
                executed_gates.append({
                    "sender_id": int(sender_id),
                    "gate": float(gate.detach().cpu().item()),
                })
                return gate

            result = collaborate_local_candidate(
                c3r=self.network.c3r,
                forward_head=self.network.forward_head,
                feat_len_s=self.network.feat_len_s,
                candidate=local_candidate,
                packets=tuple(selected_packets),
                context=branch_context,
                gate_provider=fixed_gate,
            )
            candidate = (self._c3r_candidate_from_head(local_candidate, result.output)
                         if result.used_remote else local_candidate)
            sender_ids = tuple(int(value) for value in result.collaboration.get(
                "accepted_sender_ids", ()))
            post_state_digest = self._counterfactual_state_digest()
            trace = {
                "pre_state_digest": pre_state_digest,
                "post_state_digest": post_state_digest,
                "executed_gates": executed_gates,
                "sender_provenance": provenance,
            }
            if diagnostics_enabled:
                if post_state_digest != shared_pre_state_digest:
                    raise AssertionError(
                        "counterfactual branch mutated tracker state")
                if any(item["gate"] != 0.25 for item in executed_gates):
                    raise AssertionError(
                        "sender-only override gate must execute as exactly 0.25")
                provenance_senders = tuple(
                    item["sender_id"] for item in provenance)
                executed_senders = tuple(
                    item["sender_id"] for item in executed_gates)
                if sender_ids != provenance_senders or sender_ids != executed_senders:
                    raise AssertionError(
                        "counterfactual sender provenance mismatch")
            branch_traces[branch_name] = trace
            return candidate, sender_ids

        sender_only = {}
        for packet in packets:
            packet_trace = self._counterfactual_packet_provenance(packet)
            branch_name = "sender{}_only".format(packet_trace["sender_id"])
            candidate, sender_ids = branch((packet,), branch_name)
            if len(sender_ids) == 1:
                sender_only[int(sender_ids[0])] = candidate
        both_candidate, both_sender_ids = branch(tuple(packets), "both_sender")
        backbone_count_after = int(self._temporal_gate_backbone_forward_count)
        if diagnostics_enabled and backbone_count_after != backbone_count_before:
            raise AssertionError(
                "counterfactual branches must not add a backbone forward")
        return {
            "local": local_candidate,
            "sender_only": sender_only,
            "both": both_candidate,
            "both_sender_ids": both_sender_ids,
            "diagnostics": {
                "enabled": diagnostics_enabled,
                "shared_pre_state_digest": shared_pre_state_digest,
                "branches": branch_traces,
                "backbone_forward_count_before": backbone_count_before,
                "backbone_forward_count_after": backbone_count_after,
                "no_additional_backbone_forward": (
                    backbone_count_before == backbone_count_after),
            },
        }

    @staticmethod
    def _counterfactual_digest(value):
        digest = hashlib.sha256()

        def update(item):
            if torch.is_tensor(item):
                tensor = item.detach().cpu().contiguous()
                digest.update(b"tensor:")
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            elif isinstance(item, dict):
                digest.update(b"dict{")
                for key in sorted(item, key=lambda value: str(value)):
                    update(str(key))
                    update(item[key])
                digest.update(b"}")
            elif isinstance(item, (list, tuple)):
                digest.update(b"sequence[")
                for value_item in item:
                    update(value_item)
                digest.update(b"]")
            else:
                digest.update(json.dumps(
                    item, sort_keys=True, allow_nan=False,
                    separators=(",", ":")).encode("utf-8"))

        update(value)
        return digest.hexdigest()

    def _counterfactual_state_digest(self):
        return self._counterfactual_digest({
            "state": copy.deepcopy(self.state),
            "frame_id": int(self.frame_id),
            "last_frame_by_sender": dict(self.c3r_last_frame_by_sender),
        })

    def _counterfactual_packet_provenance(self, packet):
        if hasattr(packet, "sender_id"):
            payload = self.network.c3r.codec.serialize(packet)
        else:
            payload = bytes(packet)
        parsed = self.network.c3r.codec.parse(payload)
        return {
            "sender_id": int(parsed.sender_id),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "packet_bytes": len(payload),
        }

    def _counterfactual_candidate_digest(self, candidate):
        return self._counterfactual_digest({
            "target_bbox": candidate["target_bbox"],
            "pred_boxes": candidate.get("pred_boxes"),
            "response": candidate.get("response"),
        })

    def _audit_behavior_candidate_submission(self, counterfactual, candidate):
        if str(self.network.c3r.variant) != "c1":
            raise AssertionError(
                "only the frozen C1 behavior candidate may be submitted")
        counterfactual_candidates = (
            tuple(counterfactual["sender_only"].values())
            + (counterfactual["both"],))
        if any(candidate is value for value in counterfactual_candidates):
            raise AssertionError(
                "a counterfactual candidate was submitted for commit")
        behavior_digest = self._counterfactual_candidate_digest(candidate)
        counterfactual["diagnostics"].update({
            "behavior_variant": "c1",
            "behavior_candidate_digest": behavior_digest,
            "submitted_candidate_digest": behavior_digest,
            "only_frozen_c1_behavior_submitted": True,
        })
        return counterfactual["diagnostics"]

    @staticmethod
    def _counterfactual_source_diagnostic_fields(counterfactual, sender_id):
        diagnostic = counterfactual["diagnostics"]
        branch_name = "sender{}_only".format(int(sender_id))
        branch_trace = diagnostic["branches"][branch_name]
        provenance = branch_trace["sender_provenance"][0]
        executed = branch_trace["executed_gates"][0]
        return {
            "counterfactual_sender_only_executed_gate": executed["gate"],
            "counterfactual_sender_payload_sha256":
                provenance["payload_sha256"],
            "counterfactual_sender_provenance_id": provenance["sender_id"],
            "counterfactual_shared_pre_state_digest":
                diagnostic["shared_pre_state_digest"],
            "counterfactual_branch_pre_state_digests": {
                name: trace["pre_state_digest"]
                for name, trace in diagnostic["branches"].items()
            },
            "counterfactual_branch_post_state_digests": {
                name: trace["post_state_digest"]
                for name, trace in diagnostic["branches"].items()
            },
            "counterfactual_no_additional_backbone_forward":
                diagnostic["no_additional_backbone_forward"],
            "counterfactual_only_frozen_c1_behavior_submitted":
                diagnostic["only_frozen_c1_behavior_submitted"],
            "counterfactual_behavior_candidate_digest":
                diagnostic["behavior_candidate_digest"],
            "counterfactual_submitted_candidate_digest":
                diagnostic["submitted_candidate_digest"],
        }

    def c3r_track_with_packets(self, info, local_candidate, packets,
                               context: C3RReceiverContext,
                               sent_packets=1, debug_name=""):
        """Commit one explicit C3R receiver step without a second backbone."""
        if not self.c3r_enabled:
            raise RuntimeError("C3R packet tracking requested while disabled")
        self.frame_id += 1
        if int(context.frame_id) != int(self.frame_id):
            raise ValueError("C3R context frame does not match tracker frame")
        if context.last_frame_by_sender is not self.c3r_last_frame_by_sender:
            raise ValueError("C3R replay state must belong to this tracker")
        counterfactual = None
        if self.temporal_gate_rollout_capture:
            if self.temporal_gate_enabled:
                raise RuntimeError("behavior rollout must commit unchanged current C1")
            counterfactual = self.c3r_counterfactual_candidates(
                local_candidate, tuple(packets), context)
        gate_provider = None
        if self.temporal_gate_enabled:
            if self.temporal_gate_runtime is None:
                raise RuntimeError("Temporal Gate enabled without runtime")
            gate_provider = lambda sender_id, vector: self.temporal_gate_runtime.gate_for(
                context.target_id, context.receiver_id, sender_id,
                context.frame_id, vector)
        head_result = collaborate_local_candidate(
            c3r=self.network.c3r,
            forward_head=self.network.forward_head,
            feat_len_s=self.network.feat_len_s,
            candidate=local_candidate,
            packets=tuple(packets),
            context=context,
            instrumentation=self.c3r_instrumentation_enabled,
            remote_information_diagnostics=
                self.remote_information_diagnostics,
            gate_provider=gate_provider,
        )
        if head_result.used_remote:
            candidate = self._c3r_candidate_from_head(
                local_candidate, head_result.output)
        else:
            candidate = local_candidate
        if (counterfactual is not None
                and self.temporal_gate_counterfactual_diagnostics):
            self._audit_behavior_candidate_submission(
                counterfactual, candidate)
        dispositions = head_result.collaboration.get("dispositions", ())
        received_by_peer = {}
        for item in dispositions:
            if item.sender_id is not None:
                peer = int(item.sender_id)
                received_by_peer[peer] = received_by_peer.get(peer, 0) + 1
        self.c3r_message_accounting.record_frame(
            sent=int(sent_packets),
            received=len(tuple(packets)),
            accepted=int(head_result.collaboration.get("accepted_count", 0)),
            received_by_peer=received_by_peer,
        )
        output = self._commit_candidate(
            candidate, info=info, debug_name=debug_name)
        output["c3r_diagnostics"] = c3r_diagnostic_row(
            target_id=context.target_id,
            receiver_id=context.receiver_id,
            context=context,
            sent_packets=int(sent_packets),
            received_packets=len(tuple(packets)),
            collaboration=head_result.collaboration,
        )
        if self.c3r_instrumentation_enabled:
            local_quality = self.network.c3r.encoder.response_quality(
                local_candidate["out_dict"]["score_map"])[0].detach().float().cpu().tolist()
            final_quality = self.network.c3r.encoder.response_quality(
                candidate["out_dict"]["score_map"])[0].detach().float().cpu().tolist()
            common = {
                "fold_id": self.c3r_instrumentation_fold_id,
                "target_id": context.target_id,
                "sequence_id": "{}-{}".format(
                    context.target_id, int(context.receiver_id) + 1),
                "receiver_view": int(context.receiver_id),
                "frame_id": int(context.frame_id),
                "timestamp_ms": int(context.timestamp_ms),
                "local_bbox_xywh": [float(value) for value in local_candidate["target_bbox"]],
                "c1_bbox_xywh": [float(value) for value in candidate["target_bbox"]],
                "tracker_state_before_xywh": [
                    float(value) for value in local_candidate.get("prev_bbox", ())],
                "tracker_state_after_xywh": [float(value) for value in self.state],
                "local_score": self._score_value(local_candidate["max_score"]),
                "c1_score": self._score_value(candidate["max_score"]),
                "local_confidence": self._candidate_confidence(local_candidate),
                "c1_confidence": self._candidate_confidence(candidate),
                "local_apce": self._score_value(local_candidate["apce"]),
                "c1_apce": self._score_value(candidate["apce"]),
                "local_response_quality": local_quality,
                "c1_response_quality": final_quality,
                "uses_gt": False,
            }
            source_rows = []
            for source in head_result.collaboration.get(
                    "instrumentation_source_rows", ()):
                row = dict(common)
                row.update(copy.deepcopy(source))
                row["sender_view"] = int(row.pop("sender_id"))
                if counterfactual is not None:
                    sender_candidate = counterfactual["sender_only"].get(
                        row["sender_view"])
                    if sender_candidate is None:
                        raise RuntimeError(
                            "accepted sender missing counterfactual branch")
                    row["sender_only_gate_0.25_bbox_xywh"] = [
                        float(value) for value in sender_candidate["target_bbox"]]
                    row["both_senders_gate_0.25_bbox_xywh"] = [
                        float(value) for value in counterfactual["both"]["target_bbox"]]
                    row["behavior_c1_bbox_xywh"] = [
                        float(value) for value in candidate["target_bbox"]]
                    if self.remote_information_diagnostics:
                        sender_quality = self.network.c3r.encoder.response_quality(
                            sender_candidate["out_dict"]["score_map"]
                        )[0].detach().float().cpu().tolist()
                        both_quality = self.network.c3r.encoder.response_quality(
                            counterfactual["both"]["out_dict"]["score_map"]
                        )[0].detach().float().cpu().tolist()
                        row.update({
                            "sender_only_score": self._score_value(
                                sender_candidate["max_score"]),
                            "sender_only_confidence":
                                self._candidate_confidence(sender_candidate),
                            "sender_only_apce": self._score_value(
                                sender_candidate["apce"]),
                            "sender_only_response_quality": sender_quality,
                            "both_senders_score": self._score_value(
                                counterfactual["both"]["max_score"]),
                            "both_senders_confidence":
                                self._candidate_confidence(
                                    counterfactual["both"]),
                            "both_senders_apce": self._score_value(
                                counterfactual["both"]["apce"]),
                            "both_senders_response_quality": both_quality,
                        })
                    row["model_input_fields"] = ["reliability_input_normalized"]
                    row["uses_gt_for_features"] = False
                    if self.temporal_gate_counterfactual_diagnostics:
                        row.update(
                            self._counterfactual_source_diagnostic_fields(
                                counterfactual, row["sender_view"]))
                source_rows.append(row)
            aggregate = dict(common)
            aggregate.update(copy.deepcopy(head_result.collaboration.get(
                "instrumentation_aggregate", {})))
            aggregate["accepted_sender_views"] = [
                int(value) for value in head_result.collaboration.get(
                "accepted_sender_ids", ())]
            if counterfactual is not None:
                aggregate["counterfactual_diagnostics"] = copy.deepcopy(
                    counterfactual["diagnostics"])
            output["c3r_source_instrumentation"] = source_rows
            output["c3r_aggregate_instrumentation"] = aggregate
        return output, candidate["max_score"], candidate["apce"]

    def reset_temporal_gate(self, target_id=None, receiver_id=None):
        """Reset sidecar histories at initialization without touching C1 state."""
        if self.temporal_gate_runtime is not None:
            self.temporal_gate_runtime.reset()

    def c3r_accounting_report(self):
        return self.c3r_message_accounting.report(peers=2, broadcast=True)

    def pcum_track_with_remote(
        self,
        image,
        info=None,
        remote_prompts=None,
        remote_states=None,
        local_candidate=None,
        search_factor=None,
        debug_name=""
    ):
        """
        Commit one PCUM test-time tracking step.

        If remote prompts are unavailable, the supplied local candidate is
        committed. If configured, a remote candidate that sharply lowers the
        response score is rejected in favor of the local candidate.
        """
        self.frame_id += 1

        remote_prompts = [p for p in (remote_prompts or []) if p is not None]
        pcum_test_cfg = getattr(self.cfg.TEST, "PCUM", None)
        save_decision = bool(getattr(pcum_test_cfg, "SAVE_DECISION_LOG", False))
        selector_mode = validate_reliability_selector(getattr(
            pcum_test_cfg,
            "RELIABILITY_SELECTOR",
            "none",
        ))
        selector_diagnostics = bool(getattr(
            pcum_test_cfg,
            "SELECTOR_DIAGNOSTICS",
            True,
        ))
        redetect_triggered = search_factor is not None
        local_source = 0.0
        selected_source = 0.0
        fallback_reason = 0.0
        redetect_local_conf = -1.0
        remote_candidate_conf = -1.0
        raw_collaborative_candidate = None
        selector_decision = None

        if search_factor is not None and local_candidate is not None:
            use_redetect_local = bool(getattr(
                pcum_test_cfg,
                "MOTION_REDETECT_USE_LOCAL_CANDIDATE",
                True,
            ))
            if use_redetect_local:
                redetect_local_candidate = self._run_candidate(
                    image=image,
                    search_factor=search_factor,
                    return_score=True
                )
                redetect_local_conf = self._candidate_confidence(redetect_local_candidate)
                local_min_gain = float(getattr(
                    pcum_test_cfg,
                    "MOTION_REDETECT_LOCAL_MIN_GAIN",
                    0.0,
                ))
                if (
                    redetect_local_conf
                    > self._candidate_confidence(local_candidate) + local_min_gain
                ):
                    local_candidate = redetect_local_candidate
                    local_source = 1.0
                    selected_source = local_source

        if len(remote_prompts) == 0:
            candidate = local_candidate or self._run_candidate(
                image=image,
                search_factor=search_factor,
                return_score=True
            )
            raw_collaborative_candidate = candidate
            fallback_reason = 1.0
        else:
            candidate = self._run_candidate(
                image=image,
                search_factor=search_factor,
                remote_prompts=remote_prompts,
                remote_states=remote_states,
                return_score=True
            )
            raw_collaborative_candidate = candidate
            selected_source = 2.0
            remote_candidate_conf = self._candidate_confidence(candidate)

            if selector_mode == "deterministic" and local_candidate is not None:
                selector_decision = deterministic_reliability_selector_decision(
                    local_confidence=self._candidate_confidence(local_candidate),
                    collaborative_confidence=remote_candidate_conf,
                    collaborative_motion_reliability=self._candidate_motion_reliability(candidate),
                    margin=float(getattr(pcum_test_cfg, "SELECTOR_MARGIN", 0.0)),
                    motion_threshold=float(getattr(
                        pcum_test_cfg,
                        "SELECTOR_MOTION_THRESHOLD",
                        0.0,
                    )),
                )
                if not selector_decision["use_collaborative"]:
                    candidate = local_candidate
                    selected_source = local_source
                    fallback_reason = 4.0

            keep_local = (
                selector_mode == "none"
                and bool(getattr(pcum_test_cfg, "KEEP_LOCAL_IF_REMOTE_WORSE", True))
            )
            max_drop = float(getattr(pcum_test_cfg, "REMOTE_SCORE_MAX_DROP", 0.05))
            if keep_local and local_candidate is not None:
                local_score = self._score_value(local_candidate["max_score"])
                remote_score = self._score_value(candidate["max_score"])
                if remote_score + max_drop < local_score:
                    candidate = local_candidate
                    selected_source = local_source
                    fallback_reason = 2.0
                else:
                    keep_confidence = bool(getattr(
                        pcum_test_cfg,
                        "KEEP_LOCAL_IF_REMOTE_CONFIDENCE_WORSE",
                        False,
                    ))
                    confidence_drop = float(getattr(
                        pcum_test_cfg,
                        "REMOTE_CONFIDENCE_MAX_DROP",
                        0.02,
                    ))
                    if (
                        keep_confidence
                        and self._candidate_confidence(candidate) + confidence_drop
                        < self._candidate_confidence(local_candidate)
                    ):
                        candidate = local_candidate
                        selected_source = local_source
                        fallback_reason = 3.0

        candidate, mcr_record = self._apply_mcr(
            image, candidate, remote_prompts=remote_prompts, remote_states=remote_states)
        output = self._commit_candidate(candidate, info=info, debug_name=debug_name)
        if mcr_record is not None:
            output["mcr_diagnostics"] = mcr_record
        output = self._attach_motion_shadow_diagnostics(candidate, output)
        remote_weight_record = self._remote_aggregation_record(
            raw_collaborative_candidate
        )
        if remote_weight_record is not None:
            output["pcum_remote_weights"] = remote_weight_record
        remote_suppression_record = self._remote_suppression_record(
            raw_collaborative_candidate)
        if remote_suppression_record is not None:
            output["pcum_remote_suppression"] = remote_suppression_record
        if selector_diagnostics:
            output["pcum_selector"] = self._selector_record(
                selector_mode,
                selector_decision,
                num_remote_prompts=len(remote_prompts),
                selected_source=selected_source,
            )
        if self.pcum_diagnostics_enabled:
            source_names = {
                0.0: "local",
                1.0: "local_redetect",
                2.0: "raw_collaborative",
            }
            fallback_names = {
                0.0: "none",
                1.0: "no_remote_prompt",
                2.0: "remote_score_drop",
                3.0: "remote_confidence_drop",
                4.0: "reliability_selector",
            }
            self.last_pcum_diagnostic = {
                "local": self._diagnostic_candidate_snapshot(local_candidate),
                "raw_collaborative": self._diagnostic_candidate_snapshot(
                    raw_collaborative_candidate
                ),
                "final": self._diagnostic_candidate_snapshot(candidate),
                "final_source": source_names.get(selected_source, "unknown"),
                "fallback_triggered": fallback_reason in (2.0, 3.0, 4.0),
                "fallback_reason": fallback_names.get(fallback_reason, "unknown"),
            }
        if save_decision:
            local_conf = self._candidate_confidence(local_candidate)
            output["pcum_decision"] = [
                float(redetect_triggered),
                float(len(remote_prompts)),
                float(selected_source),
                float(fallback_reason),
                float(local_conf),
                float(redetect_local_conf),
                float(remote_candidate_conf),
                float(search_factor if search_factor is not None else -1.0),
            ]
        return output, candidate["max_score"], candidate["apce"]

    def _track_single(
        self,
        image,
        info=None,
        search_factor=None,
        return_score_apce=False,
        debug_name="",
        prompt_map=None,
        prompt_gate_input=None,
        remote_prompts=None,
        remote_states=None
    ):
        """
        单机跟踪核心函数。

        Args:
            image: 当前视角图像
            search_factor: 如果为 None，使用默认 self.params.search_factor；
                           如果指定，用于 general_redetect 这类大搜索区域。
            return_score_apce: 是否返回 max_score 和 APCE。
        """
        self.frame_id += 1

        candidate = self._run_candidate(
            image,
            search_factor=search_factor,
            prompt_map=prompt_map,
            prompt_gate_input=prompt_gate_input,
            remote_prompts=remote_prompts,
            remote_states=remote_states,
            return_score=return_score_apce
        )
        candidate, mcr_record = self._apply_mcr(
            image, candidate, remote_prompts=remote_prompts, remote_states=remote_states)
        output = self._commit_candidate(candidate, info=info, debug_name=debug_name)
        if mcr_record is not None:
            output["mcr_diagnostics"] = mcr_record
        output = self._attach_motion_shadow_diagnostics(candidate, output)

        if return_score_apce:
            return output, candidate["max_score"], candidate["apce"]

        return output

    # ------------------------------------------------------------
    # Standard single-view tracking API
    # ------------------------------------------------------------
    def track(self, image, info: dict = None):
        """
        标准单机 track。
        """
        if self.plain_collaboration_enabled:
            raise RuntimeError(
                "Plain Collaboration V1 requires the Three-MDOT "
                "synchronized three-view runner")
        return self._track_single(
            image=image,
            info=info,
            search_factor=self.params.search_factor,
            return_score_apce=False,
            debug_name=""
        )

    # ------------------------------------------------------------
    # ThreeMDOT-compatible tracking APIs
    # ------------------------------------------------------------
    def Fusetrack(self, image, info: dict = None):
        """
        兼容原 ThreeMDOT 接口。

        当前版本不做融合，只做单机 EnTeRTrack。
        """
        if self.plain_collaboration_enabled:
            raise RuntimeError(
                "Plain Collaboration V1 must not use independent Fusetrack")
        return self._track_single(
            image=image,
            info=info,
            search_factor=self.params.search_factor,
            return_score_apce=True,
            debug_name=""
        )

    def multi_Fusetrack(
        self,
        image_a,
        image_b,
        drone_id,
        info_a: dict = None,
        info_b: dict = None
    ):
        """
        兼容双机接口。

        当前版本忽略 image_b，只跟踪 image_a。
        """
        return self._track_single(
            image=image_a,
            info=info_a,
            search_factor=self.params.search_factor,
            return_score_apce=True,
            debug_name=str(drone_id)
        )

    def three_multi_Fusetrack(
        self,
        image_a,
        image_b,
        image_c,
        drone_id,
        info_a: dict = None,
        info_b: dict = None,
        info_c: dict = None
    ):
        """
        兼容三机接口。

        当前版本忽略 image_b / image_c，只跟踪 image_a。
        """
        return self._track_single(
            image=image_a,
            info=info_a,
            search_factor=self.params.search_factor,
            return_score_apce=True,
            debug_name=str(drone_id)
        )

    def three_nomulti_Fusetrack(
        self,
        image_a,
        image_b,
        image_c,
        drone_id,
        info_a: dict = None,
        info_b: dict = None,
        info_c: dict = None,
        prompt_input=None
    ):
        """
        兼容旧的 three_nomulti_Fusetrack 接口。

        当前版本：
            1. 不使用 prompt_input；
            2. 不生成 prompt；
            3. 只跟踪 image_a。

        返回第四个值 None，用来兼容旧代码中接收 generated_prompt 的写法。
        """
        out, max_score, response_APCE = self._track_single(
            image=image_a,
            info=info_a,
            search_factor=self.params.search_factor,
            return_score_apce=True,
            debug_name=str(drone_id)
        )

        generated_prompt = None

        return out, max_score, response_APCE, generated_prompt

    def multi_Fusetrack2(
        self,
        image_a,
        image_b,
        state2,
        info_a: dict = None,
        info_b: dict = None
    ):
        """
        兼容旧接口。

        当前版本忽略 image_b / state2，只跟踪 image_a。
        """
        out, max_score, _ = self._track_single(
            image=image_a,
            info=info_a,
            search_factor=self.params.search_factor,
            return_score_apce=True,
            debug_name=""
        )

        return out, max_score

    # ------------------------------------------------------------
    # Redetection-compatible APIs
    # ------------------------------------------------------------
    def general_redetect(
        self,
        image_a,
        image_b=None,
        drone_id="",
        tmp_s_factor=7.0,
        info_a: dict = None,
        info_b: dict = None
    ):
        """
        单机 general redetect。

        当前版本不使用 image_b。
        只是用更大的 search_factor 在 image_a 上重新搜索。
        """
        print(str(drone_id), "single-view general redetect")

        return self._track_single(
            image=image_a,
            info=info_a,
            search_factor=tmp_s_factor,
            return_score_apce=True,
            debug_name=str(drone_id)
        )

    def search_redetect(
        self,
        image_a,
        image_b,
        drone_id,
        state_b,
        tmp_factor=4.0,
        tmp_s_factor=12.0,
        info_a: dict = None,
        info_b: dict = None
    ):
        """
        兼容 cross-redetect 接口。

        当前没有多机和跨视角模板，因此退化为 image_a 上的大范围单机重检测。
        """
        print(str(drone_id), "single-view fallback redetect")

        return self.general_redetect(
            image_a=image_a,
            image_b=None,
            drone_id=drone_id,
            tmp_s_factor=tmp_s_factor,
            info_a=info_a,
            info_b=None
        )

    def three_search_redetect(
        self,
        image_a,
        image_b,
        drone_id,
        state_b,
        tmp_factor=4.0,
        tmp_s_factor=12.0,
        info_a: dict = None,
        info_b: dict = None
    ):
        """
        兼容 three_search_redetect 接口。

        当前没有多机和跨视角模板，因此退化为 image_a 上的大范围单机重检测。
        """
        print(str(drone_id), "single-view fallback three_search_redetect")

        return self.general_redetect(
            image_a=image_a,
            image_b=None,
            drone_id=drone_id,
            tmp_s_factor=tmp_s_factor,
            info_a=info_a,
            info_b=None
        )

    # ------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------
    def _debug_vis(
        self,
        image,
        info,
        x_patch_arr,
        pred_score_map,
        response,
        out_dict,
        debug_name=""
    ):
        if not self.use_visdom:
            x1, y1, w, h = self.state
            suffix = "" if debug_name == "" else "_" + str(debug_name)
            save_path = os.path.join(
                self.save_dir,
                "%04d%s.jpg" % (self.frame_id, suffix)
            )

            if cv2 is not None:
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(
                    image_BGR,
                    (int(x1), int(y1)),
                    (int(x1 + w), int(y1 + h)),
                    color=(0, 0, 255),
                    thickness=2
                )
                cv2.imwrite(save_path, image_BGR)
            else:
                pil_image = Image.fromarray(image.astype("uint8")).convert("RGB")
                draw = ImageDraw.Draw(pil_image)
                draw.rectangle(
                    (int(x1), int(y1), int(x1 + w), int(y1 + h)),
                    outline=(255, 0, 0),
                    width=2,
                )
                pil_image.save(save_path, quality=92)

        else:
            vis_name = "Tracking" if debug_name == "" else "Tracking" + str(debug_name)

            if info is not None and "gt_bbox" in info:
                gt_box = info["gt_bbox"]
                if hasattr(gt_box, "tolist"):
                    gt_box = gt_box.tolist()
                self.visdom.register(
                    (image, gt_box, self.state),
                    "Tracking",
                    1,
                    vis_name
                )
            else:
                self.visdom.register(
                    (image, self.state),
                    "Tracking",
                    1,
                    vis_name
                )

            suffix = "" if debug_name == "" else str(debug_name)

            self.visdom.register(
                torch.from_numpy(x_patch_arr).permute(2, 0, 1),
                "image",
                1,
                "search_region" + suffix
            )

            self.visdom.register(
                torch.from_numpy(self.z_patch_arr).permute(2, 0, 1),
                "image",
                1,
                "template" + suffix
            )

            self.visdom.register(
                pred_score_map.view(self.feat_sz, self.feat_sz),
                "heatmap",
                1,
                "score_map" + suffix
            )

            self.visdom.register(
                response.view(self.feat_sz, self.feat_sz),
                "heatmap",
                1,
                "score_map_hann" + suffix
            )

            if "removed_indexes_s" in out_dict and out_dict["removed_indexes_s"]:
                removed_indexes_s = out_dict["removed_indexes_s"]
                removed_indexes_s = [
                    removed_indexes_s_i.cpu().numpy()
                    for removed_indexes_s_i in removed_indexes_s
                    if removed_indexes_s_i is not None
                ]

                if len(removed_indexes_s) > 0:
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)

                    self.visdom.register(
                        torch.from_numpy(masked_search).permute(2, 0, 1),
                        "image",
                        1,
                        "masked_search" + suffix
                    )

            while self.pause_mode:
                if self.step:
                    self.step = False
                    break

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    def calAPCE(self, response):
        """
        Average Peak-to-Correlation Energy.

        response: [B, 1, H, W] or [B, H, W]
        """
        flattened_response = response.flatten(1)

        max_score = torch.max(flattened_response, dim=1, keepdim=True)[0]
        min_score = torch.min(flattened_response, dim=1, keepdim=True)[0]

        bottom = torch.mean(
            (flattened_response - min_score) ** 2,
            dim=1,
            keepdim=True
        )

        apce = ((max_score - min_score) ** 2) / (bottom + 1e-8)

        return apce

    # ------------------------------------------------------------
    # Box mapping
    # ------------------------------------------------------------
    def map_box_back(self, pred_box: list, resize_factor: float, reference_bbox=None):
        reference_bbox = self.state if reference_bbox is None else reference_bbox
        cx_prev = reference_bbox[0] + 0.5 * reference_bbox[2]
        cy_prev = reference_bbox[1] + 0.5 * reference_bbox[3]

        cx, cy, w, h = pred_box

        half_side = 0.5 * self.params.search_size / resize_factor

        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)

        return [
            cx_real - 0.5 * w,
            cy_real - 0.5 * h,
            w,
            h
        ]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float,
                           reference_bbox=None):
        reference_bbox = self.state if reference_bbox is None else reference_bbox
        cx_prev = reference_bbox[0] + 0.5 * reference_bbox[2]
        cy_prev = reference_bbox[1] + 0.5 * reference_bbox[3]

        cx, cy, w, h = pred_box.unbind(-1)

        half_side = 0.5 * self.params.search_size / resize_factor

        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)

        return torch.stack(
            [
                cx_real - 0.5 * w,
                cy_real - 0.5 * h,
                w,
                h
            ],
            dim=-1
        )


def get_tracker_class():
    return EnTeRTrack
