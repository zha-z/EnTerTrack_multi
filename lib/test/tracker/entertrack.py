import os
import torch
import cv2

from lib.models.entertrack import build_entertrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.tracker.data_utils import Preprocessor
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


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

        network = build_entertrack(params.cfg, training=False)
        self._load_network(network, self.params.checkpoint)

        self.network = network.cuda()
        self.network.eval()

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

        if self.save_all_boxes:
            all_boxes_save = info["init_bbox"] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

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
        return_score=True
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

        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            search_factor,
            output_sz=self.params.search_size
        )

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

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
            self.map_box_back(pred_box, resize_factor),
            H,
            W,
            margin=10
        )

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size / resize_factor,
                resize_factor
            )
            output = {
                "target_bbox": target_bbox,
                "all_boxes": all_boxes.view(-1).tolist()
            }
        else:
            output = {"target_bbox": target_bbox}

        return {
            "output": output,
            "target_bbox": target_bbox,
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
        }

    def _commit_candidate(self, candidate, info=None, debug_name=""):
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

        return candidate["output"]

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

    def pcum_local_candidate(self, image, search_factor=None):
        return self._run_candidate(
            image=image,
            search_factor=search_factor,
            return_score=True
        )

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
                local_min_gain = float(getattr(
                    pcum_test_cfg,
                    "MOTION_REDETECT_LOCAL_MIN_GAIN",
                    0.0,
                ))
                if (
                    self._candidate_confidence(redetect_local_candidate)
                    > self._candidate_confidence(local_candidate) + local_min_gain
                ):
                    local_candidate = redetect_local_candidate

        if len(remote_prompts) == 0:
            candidate = local_candidate or self._run_candidate(
                image=image,
                search_factor=search_factor,
                return_score=True
            )
        else:
            candidate = self._run_candidate(
                image=image,
                search_factor=search_factor,
                remote_prompts=remote_prompts,
                remote_states=remote_states,
                return_score=True
            )

            keep_local = bool(getattr(pcum_test_cfg, "KEEP_LOCAL_IF_REMOTE_WORSE", True))
            max_drop = float(getattr(pcum_test_cfg, "REMOTE_SCORE_MAX_DROP", 0.05))
            if keep_local and local_candidate is not None:
                local_score = self._score_value(local_candidate["max_score"])
                remote_score = self._score_value(candidate["max_score"])
                if remote_score + max_drop < local_score:
                    candidate = local_candidate

        output = self._commit_candidate(candidate, info=info, debug_name=debug_name)
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
        output = self._commit_candidate(candidate, info=info, debug_name=debug_name)

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
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.rectangle(
                image_BGR,
                (int(x1), int(y1)),
                (int(x1 + w), int(y1 + h)),
                color=(0, 0, 255),
                thickness=2
            )

            suffix = "" if debug_name == "" else "_" + str(debug_name)
            save_path = os.path.join(
                self.save_dir,
                "%04d%s.jpg" % (self.frame_id, suffix)
            )

            cv2.imwrite(save_path, image_BGR)

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
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]

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

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]

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
