import importlib
import os
import json
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
from lib.test.coop import CommunicationSimulator

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np

import copy
import torch


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence."""
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile."""

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                frame_disp = frame.copy()
                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


# 融合结果
    def Fuse_run_sequence(self, seq, debug=None):
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        init_info = seq.init_info()
        tracker = self.create_tracker(params)
        output = self.Fuse_track_sequence(tracker, seq, init_info)
        return output

# 融合结果
    def Fuse_track_sequence(self, tracker, seq, init_info):
        output = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)
            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out, max_score, response_APCE = tracker.Fusetrack(image, info)              # 保存了score
            prev_output = OrderedDict(out)
            
            # 💡【核心修复】：防止 tensor 传入列表，解决 .cpu() 报错与内存泄漏
            ms_val = max_score.item() if torch.is_tensor(max_score) else max_score
            apce_val = response_APCE.item() if torch.is_tensor(response_APCE) else response_APCE
            _store_outputs(out, {'time': time.time() - start_time, 'max_score': ms_val, 'APCE': apce_val})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output


# 融合结果
    def Fuse_multi_run_sequence(self, seq_a, seq_b, debug=None):
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        init_info_a = seq_a.init_info()
        init_info_b = seq_b.init_info()

        tracker = self.create_tracker(params)
        tracker2 = self.create_tracker(params)

        output_a, output_b = self.Fuse_multi_track_matching_sequence(tracker, tracker2, seq_a, seq_b, init_info_a, init_info_b)       # 多机匹配候选区域
        return output_a, output_b


# 融合结果, 多模板
    def Fuse_multi_track_sequence(self, tracker,tracker2, seq_a, seq_b, init_info_a, init_info_b):
        output_a = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}
        output_b = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}

        if tracker.params.save_all_boxes:
            output_a['all_boxes'] = []
            output_a['all_scores'] = []

        if tracker2.params.save_all_boxes:
            output_b['all_boxes'] = []
            output_b['all_scores'] = []

        def _store_outputs(output,tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        image_a = self._read_image(seq_a.frames[0])
        image_b = self._read_image(seq_b.frames[0])

        start_time = time.time()

        out_a = tracker.multi_initialize(image_a, image_b, init_info_a, init_info_b)
        out_b = tracker2.multi_initialize(image_b, image_a, init_info_b, init_info_a)

        if out_a is None: out_a = {}
        if out_b is None: out_b = {}

        prev_output_a = OrderedDict(out_a)
        init_default_a = {'target_bbox': init_info_a.get('init_bbox'), 'time': time.time() - start_time, 'max_score': 0, 'APCE':0}
        prev_output_b = OrderedDict(out_b)
        init_default_b = {'target_bbox': init_info_b.get('init_bbox'), 'time': time.time() - start_time, 'max_score': 0, 'APCE':0}
                        
        if tracker.params.save_all_boxes:
            init_default_a['all_boxes'] = out_a['all_boxes']
            init_default_a['all_scores'] = out_a['all_scores']

        if tracker2.params.save_all_boxes:
            init_default_b['all_boxes'] = out_b['all_boxes']
            init_default_b['all_scores'] = out_b['all_scores']

        _store_outputs(output_a, out_a, init_default_a)
        _store_outputs(output_b, out_b, init_default_b)

        for frame_num, frame_path in enumerate(seq_a.frames[1:], start=1):
            image_a = self._read_image(frame_path)
            image_b = self._read_image(seq_b.frames[frame_num])

            info_a = seq_a.frame_info(frame_num)
            info_a['previous_output'] = prev_output_a

            info_b = seq_b.frame_info(frame_num)
            info_b['previous_output'] = prev_output_b

            if len(seq_a.ground_truth_rect) > 1: info_a['gt_bbox'] = seq_a.ground_truth_rect[frame_num]
            if len(seq_b.ground_truth_rect) > 1: info_b['gt_bbox'] = seq_b.ground_truth_rect[frame_num]

            # 💡【核心修复】：将时间独立开来计算，防止时间累加导致FPS降低！
            start_time_a = time.time()
            out_a, max_score_a, response_APCE_a = tracker.multi_Fusetrack(image_a, image_b, "a", info_a, info_b)
            time_a = time.time() - start_time_a

            start_time_b = time.time()
            out_b, max_score_b, response_APCE_b = tracker2.multi_Fusetrack(image_b, image_a, "b",info_b, info_a)
            time_b = time.time() - start_time_b

            state_a = tracker.state
            state_b = tracker2.state

            # 💡【核心修复】：防止 tensor 传入列表
            ms_a_val = max_score_a.item() if torch.is_tensor(max_score_a) else max_score_a
            ap_a_val = response_APCE_a.item() if torch.is_tensor(response_APCE_a) else response_APCE_a
            ms_b_val = max_score_b.item() if torch.is_tensor(max_score_b) else max_score_b
            ap_b_val = response_APCE_b.item() if torch.is_tensor(response_APCE_b) else response_APCE_b

            prev_output_a = OrderedDict(out_a)
            _store_outputs(output_a, out_a, {'time': time_a, 'max_score': ms_a_val, 'APCE': ap_a_val})
            prev_output_b = OrderedDict(out_b)
            _store_outputs(output_b, out_b, {'time': time_b, 'max_score': ms_b_val, 'APCE': ap_b_val})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_a and len(output_a[key]) <= 1: output_a.pop(key)
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_b and len(output_b[key]) <= 1: output_b.pop(key)

        return output_a, output_b


# 多机融合结果，并且双机匹配，当目标丢失时把Search region映射过去
    def Fuse_multi_track_matching_sequence(self, tracker,tracker2, seq_a, seq_b, init_info_a, init_info_b):
        output_a = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}
        output_b = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}

        if tracker.params.save_all_boxes:
            output_a['all_boxes'] = []
            output_a['all_scores'] = []

        if tracker2.params.save_all_boxes:
            output_b['all_boxes'] = []
            output_b['all_scores'] = []

        def _store_outputs(output,tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        image_a = self._read_image(seq_a.frames[0])
        image_b = self._read_image(seq_b.frames[0])

        start_time = time.time()

        out_a = tracker.multi_initialize(image_a, image_b, init_info_a, init_info_b)
        out_b = tracker2.multi_initialize(image_b, image_a, init_info_b, init_info_a)

        if out_a is None: out_a = {}
        if out_b is None: out_b = {}

        prev_output_a = OrderedDict(out_a)
        init_default_a = {'target_bbox': init_info_a.get('init_bbox'), 'time': time.time() - start_time, 'max_score': 0, 'APCE':0}
        prev_output_b = OrderedDict(out_b)
        init_default_b = {'target_bbox': init_info_b.get('init_bbox'), 'time': time.time() - start_time, 'max_score': 0, 'APCE':0}
                        
        if tracker.params.save_all_boxes:
            init_default_a['all_boxes'] = out_a['all_boxes']
            init_default_a['all_scores'] = out_a['all_scores']

        if tracker2.params.save_all_boxes:
            init_default_b['all_boxes'] = out_b['all_boxes']
            init_default_b['all_scores'] = out_b['all_scores']

        _store_outputs(output_a, out_a, init_default_a)
        _store_outputs(output_b, out_b, init_default_b)

        for frame_num, frame_path in enumerate(seq_a.frames[1:], start=1):
            image_a = self._read_image(frame_path)
            image_b = self._read_image(seq_b.frames[frame_num])

            info_a = seq_a.frame_info(frame_num)
            info_a['previous_output'] = prev_output_a

            info_b = seq_b.frame_info(frame_num)
            info_b['previous_output'] = prev_output_b

            if len(seq_a.ground_truth_rect) > 1: info_a['gt_bbox'] = seq_a.ground_truth_rect[frame_num]
            if len(seq_b.ground_truth_rect) > 1: info_b['gt_bbox'] = seq_b.ground_truth_rect[frame_num]

            start_time_a = time.time()
            out_a, max_score_a, response_APCE_a = tracker.multi_Fusetrack(image_a, image_b, "a", info_a, info_b)
            time_a = time.time() - start_time_a

            start_time_b = time.time()
            out_b, max_score_b, response_APCE_b = tracker2.multi_Fusetrack(image_b, image_a, "b", info_b, info_a)
            time_b = time.time() - start_time_b

            state_a = copy.deepcopy(tracker.state)
            state_b = copy.deepcopy(tracker2.state)

            score_a_val = max_score_a.item() if torch.is_tensor(max_score_a) else max_score_a
            apce_a_val = response_APCE_a.item() if torch.is_tensor(response_APCE_a) else response_APCE_a
            score_b_val = max_score_b.item() if torch.is_tensor(max_score_b) else max_score_b
            apce_b_val = response_APCE_b.item() if torch.is_tensor(response_APCE_b) else response_APCE_b

            #######################################################  跨机重检测  ##########################################
            redet_factor_list = [[4,12], [3,9], [2,5]]

            if((score_a_val < 0.2 and apce_a_val < 100) and (score_b_val > 0.3) and (apce_b_val > apce_a_val)):
                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker.state = copy.deepcopy(state_a)
                    s_redetect = time.time()
                    out_a_tmp, max_score_a_tmp, response_APCE_a_tmp = tracker.search_redetect(image_a, image_b, "a", copy.deepcopy(state_b), factor[0], factor[1], info_a, info_b)
                    time_a += time.time() - s_redetect
                    
                    sc_tmp = max_score_a_tmp.item() if torch.is_tensor(max_score_a_tmp) else max_score_a_tmp
                    ap_tmp = response_APCE_a_tmp.item() if torch.is_tensor(response_APCE_a_tmp) else response_APCE_a_tmp

                    tmp_dict = {"out_a":out_a_tmp, "max_score_a":sc_tmp, "response_APCE_a":ap_tmp}
                    redet_results.append(tmp_dict)
                    
                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_a"] > ms:
                        ms = result_dict["max_score_a"]
                        label = i

                if redet_results[label]["max_score_a"] - score_a_val > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_a, score_a_val, apce_a_val = redet_results[label]["out_a"], redet_results[label]["max_score_a"], redet_results[label]["response_APCE_a"]
                    tracker.state = out_a["target_bbox"]
                else:
                    print("remain ori")
                    tracker.state = copy.deepcopy(state_a)

            elif((score_b_val < 0.2 and apce_b_val < 100) and (score_a_val > 0.3) and (apce_a_val > apce_b_val)):
                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker2.state = copy.deepcopy(state_b)
                    s_redetect = time.time()
                    out_b_tmp, max_score_b_tmp, response_APCE_b_tmp = tracker2.search_redetect(image_b, image_a, "b", copy.deepcopy(state_a), factor[0], factor[1], info_b, info_a)
                    time_b += time.time() - s_redetect

                    sc_tmp = max_score_b_tmp.item() if torch.is_tensor(max_score_b_tmp) else max_score_b_tmp
                    ap_tmp = response_APCE_b_tmp.item() if torch.is_tensor(response_APCE_b_tmp) else response_APCE_b_tmp

                    tmp_dict = {"out_b":out_b_tmp, "max_score_b":sc_tmp, "response_APCE_b":ap_tmp}
                    redet_results.append(tmp_dict)
                    
                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_b"] > ms:
                        ms = result_dict["max_score_b"]
                        label = i

                if redet_results[label]["max_score_b"] - score_b_val > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_b, score_b_val, apce_b_val = redet_results[label]["out_b"], redet_results[label]["max_score_b"], redet_results[label]["response_APCE_b"]
                    tracker2.state = out_b["target_bbox"]
                else:
                    print("remain ori")
                    tracker2.state = copy.deepcopy(state_b)

            prev_output_a = OrderedDict(out_a)
            _store_outputs(output_a, out_a, {'time': time_a, 'max_score': score_a_val,  'APCE': apce_a_val})
            prev_output_b = OrderedDict(out_b)
            _store_outputs(output_b, out_b, {'time': time_b, 'max_score': score_b_val, 'APCE': apce_b_val})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_a and len(output_a[key]) <= 1: output_a.pop(key)
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_b and len(output_b[key]) <= 1: output_b.pop(key)

        return output_a, output_b


# 三机融合结果
    def Fuse_three_multi_run_sequence(self, seq_a, seq_b, seq_c, debug=None):
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        init_info_a = seq_a.init_info()
        init_info_b = seq_b.init_info()
        init_info_c = seq_c.init_info()

        tracker = self.create_tracker(params)
        tracker2 = self.create_tracker(params)
        tracker3 = self.create_tracker(params)

        #output_a, output_b, output_c = self.Fuse_three_multi_track_matching_sequence(tracker, tracker2, tracker3, seq_a, seq_b,seq_c, init_info_a, init_info_b,init_info_c)       
        output_a, output_b, output_c = self.Fuse_three_multi_track(tracker, tracker2, tracker3, seq_a, seq_b,seq_c, init_info_a, init_info_b,init_info_c)  
        return output_a, output_b,output_c


# 三机融合结果，三模板无重检测
    def Fuse_three_multi_track(self, tracker, tracker2, tracker3,
                           seq_a, seq_b, seq_c,
                           init_info_a, init_info_b, init_info_c):
        """
        ThreeMDOT 三机测试入口，但当前版本不做多机融合。

        当前逻辑：
        1. tracker  只跟踪 seq_a；
        2. tracker2 只跟踪 seq_b；
        3. tracker3 只跟踪 seq_c；
        4. 不传 prompt；
        5. 不用多模板；
        6. 不用跨机重检测；
        7. 不用 B/C 图像辅助 A。
        """

        output_a = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE': []}
        output_b = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE': []}
        output_c = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE': []}

        if tracker.params.save_all_boxes:
            output_a['all_boxes'] = []
            output_a['all_scores'] = []
        if tracker2.params.save_all_boxes:
            output_b['all_boxes'] = []
            output_b['all_scores'] = []
        if tracker3.params.save_all_boxes:
            output_c['all_boxes'] = []
            output_c['all_scores'] = []

        def _store_outputs(output, tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        def _to_float(x):
            return x.item() if torch.is_tensor(x) else x

        def _payload(out, score, apce):
            return {
                'bbox': out.get('target_bbox', None),
                'score': float(score),
                'apce': float(apce),
            }

        def _get_cfg_value(obj, name, default=None):
            return getattr(obj, name, default) if obj is not None else default

        coop_cfg = getattr(tracker.cfg.TEST, "COOP", None)
        coop_enabled = bool(_get_cfg_value(coop_cfg, "ENABLED", False))
        coop_fusion = str(_get_cfg_value(coop_cfg, "FUSION", "none")).lower()
        coop_payload = str(_get_cfg_value(coop_cfg, "PAYLOAD", "bbox_score")).lower()
        comm_sim = None
        if coop_enabled:
            bandwidth_limit = _get_cfg_value(coop_cfg, "BANDWIDTH_LIMIT_BYTES_PER_FRAME", None)
            max_neighbors = _get_cfg_value(coop_cfg, "MAX_NEIGHBORS", None)
            comm_sim = CommunicationSimulator(
                num_agents=3,
                send_interval=int(_get_cfg_value(coop_cfg, "SEND_INTERVAL", 1)),
                bandwidth_limit_bytes_per_frame=bandwidth_limit,
                packet_loss=float(_get_cfg_value(coop_cfg, "PACKET_LOSS", 0.0)),
                delay_frames=int(_get_cfg_value(coop_cfg, "DELAY_FRAMES", 0)),
                max_neighbors=max_neighbors,
                seed=int(_get_cfg_value(coop_cfg, "SEED", 0)),
            )

        def _comm_payload(agent_id, payload):
            msg = {
                'agent_id': int(agent_id),
                'bbox': payload.get('bbox', None),
                'score': float(payload.get('score', 0.0)),
                'apce': float(payload.get('apce', 0.0)),
            }
            if coop_payload in ["prompt_4x128", "prompt"]:
                msg['prompt_tokens_int8'] = bytes(512)
            elif coop_payload in ["compressed_feature_128", "feature128"]:
                msg['feature128_int8'] = bytes(128)
            elif coop_payload in ["full_feature", "full_feature_upper_bound"]:
                msg['full_feature_stub'] = bytes(4096)
            return msg

        def _exchange_messages(frame_idx, payloads):
            if comm_sim is None:
                return {0: [], 1: [], 2: []}

            scores = {agent_id: payload.get('score', 0.0) for agent_id, payload in payloads.items()}
            for agent_id, payload in payloads.items():
                neighbors = comm_sim.select_neighbors(agent_id, [0, 1, 2], scores=scores)
                comm_sim.send(
                    frame_idx=frame_idx,
                    src=agent_id,
                    dsts=neighbors,
                    payload=_comm_payload(agent_id, payload),
                    priority=float(payload.get('score', 0.0)),
                )

            delivered = {0: [], 1: [], 2: []}
            for msg in comm_sim.deliver(frame_idx):
                delivered[msg.dst].append(msg.payload)
            return delivered

        def _save_comm_stats(num_frames):
            if comm_sim is None or not bool(_get_cfg_value(coop_cfg, "SAVE_STATS", True)):
                return

            os.makedirs(self.results_dir, exist_ok=True)
            stats = comm_sim.stats.as_dict(num_frames=num_frames)
            stats.update({
                "fusion": coop_fusion,
                "payload": coop_payload,
                "send_interval": int(_get_cfg_value(coop_cfg, "SEND_INTERVAL", 1)),
                "bandwidth_limit_bytes_per_frame": _get_cfg_value(coop_cfg, "BANDWIDTH_LIMIT_BYTES_PER_FRAME", None),
                "packet_loss": float(_get_cfg_value(coop_cfg, "PACKET_LOSS", 0.0)),
                "delay_frames": int(_get_cfg_value(coop_cfg, "DELAY_FRAMES", 0)),
                "sequence": getattr(seq_a, "name", "unknown"),
            })
            stats_file = os.path.join(self.results_dir, "{}_comm_stats.json".format(getattr(seq_a, "name", "unknown")))
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2, sort_keys=True)

        def _maybe_prompt_refine(trk, image, info, out, score, apce, peer_payloads):
            if not getattr(trk, "_should_use_prompt", None):
                return out, score, apce

            if not trk._should_use_prompt(score, apce, peer_payloads):
                return out, score, apce

            old_state = copy.deepcopy(trk.state)
            prompt_result = trk.track_with_peer_prompts(
                image=image,
                info=info,
                self_score=score,
                self_apce=apce,
                peer_states=peer_payloads
            )

            if prompt_result is None:
                trk.state = old_state
                return out, score, apce

            prompt_out, prompt_score, prompt_apce = prompt_result
            prompt_score = _to_float(prompt_score)
            prompt_apce = _to_float(prompt_apce)

            if prompt_score > float(score):
                return prompt_out, prompt_score, prompt_apce

            trk.state = old_state
            return out, score, apce

        pcum_test_cfg = getattr(tracker.cfg.TEST, "PCUM", None)
        pcum_remote_enabled = (
            bool(_get_cfg_value(pcum_test_cfg, "USE_REMOTE", False))
            and bool(getattr(getattr(tracker.cfg.MODEL, "PCUM", None), "ENABLED", False))
            and hasattr(tracker, "pcum_local_candidate")
            and hasattr(tracker, "pcum_track_with_remote")
        )

        def _candidate_score(candidate):
            return _to_float(candidate["max_score"])

        def _candidate_apce(candidate):
            return _to_float(candidate["apce"])

        def _candidate_is_local_low(candidate):
            score_thr = float(_get_cfg_value(pcum_test_cfg, "LOCAL_LOW_SCORE_THR", 0.25))
            apce_thr = float(_get_cfg_value(pcum_test_cfg, "LOCAL_LOW_APCE_THR", 100.0))
            return _candidate_score(candidate) < score_thr or _candidate_apce(candidate) < apce_thr

        def _valid_remote_candidate(candidate):
            if candidate is None or candidate.get("local_prompt", None) is None:
                return False
            score_thr = float(_get_cfg_value(pcum_test_cfg, "REMOTE_SCORE_THR", 0.0))
            apce_thr = float(_get_cfg_value(pcum_test_cfg, "REMOTE_APCE_THR", 0.0))
            return _candidate_score(candidate) >= score_thr and _candidate_apce(candidate) >= apce_thr

        def _remote_inputs_for(target_index, candidates, target_tracker):
            if bool(_get_cfg_value(pcum_test_cfg, "USE_REMOTE_ONLY_WHEN_LOCAL_LOW", False)):
                if not _candidate_is_local_low(candidates[target_index]):
                    return [], None

            peers = [
                candidate for i, candidate in enumerate(candidates)
                if i != target_index and _valid_remote_candidate(candidate)
            ]
            min_remote = int(_get_cfg_value(pcum_test_cfg, "MIN_REMOTE_PROMPTS", 1))
            if len(peers) < min_remote:
                return [], None

            target_device = target_tracker.output_window.device
            remote_prompts = [
                candidate["local_prompt"].detach().to(device=target_device)
                for candidate in peers
            ]
            remote_scores = [
                max(0.0, min(1.0, float(_candidate_score(candidate))))
                for candidate in peers
            ]
            remote_state = {
                "score": torch.tensor(
                    [sum(remote_scores) / max(1, len(remote_scores))],
                    device=target_device,
                    dtype=torch.float32
                )
            }
            return remote_prompts, remote_state

        # ------------------------------------------------------------
        # 初始化：三个 tracker 分别初始化自己的视角
        # ------------------------------------------------------------
        image_a = self._read_image(seq_a.frames[0])
        image_b = self._read_image(seq_b.frames[0])
        image_c = self._read_image(seq_c.frames[0])

        start_time_a = time.time()
        out_a = tracker.initialize(image_a, init_info_a)
        time_a = time.time() - start_time_a

        start_time_b = time.time()
        out_b = tracker2.initialize(image_b, init_info_b)
        time_b = time.time() - start_time_b

        start_time_c = time.time()
        out_c = tracker3.initialize(image_c, init_info_c)
        time_c = time.time() - start_time_c

        if out_a is None:
            out_a = {}
        if out_b is None:
            out_b = {}
        if out_c is None:
            out_c = {}

        prev_output_a = OrderedDict(out_a)
        prev_output_b = OrderedDict(out_b)
        prev_output_c = OrderedDict(out_c)

        init_default_a = {
            'target_bbox': init_info_a.get('init_bbox'),
            'time': time_a,
            'max_score': 0,
            'APCE': 0
        }
        init_default_b = {
            'target_bbox': init_info_b.get('init_bbox'),
            'time': time_b,
            'max_score': 0,
            'APCE': 0
        }
        init_default_c = {
            'target_bbox': init_info_c.get('init_bbox'),
            'time': time_c,
            'max_score': 0,
            'APCE': 0
        }

        if tracker.params.save_all_boxes:
            init_default_a['all_boxes'] = out_a['all_boxes']
            init_default_a['all_scores'] = out_a['all_scores']
        if tracker2.params.save_all_boxes:
            init_default_b['all_boxes'] = out_b['all_boxes']
            init_default_b['all_scores'] = out_b['all_scores']
        if tracker3.params.save_all_boxes:
            init_default_c['all_boxes'] = out_c['all_boxes']
            init_default_c['all_scores'] = out_c['all_scores']

        _store_outputs(output_a, out_a, init_default_a)
        _store_outputs(output_b, out_b, init_default_b)
        _store_outputs(output_c, out_c, init_default_c)

        # ------------------------------------------------------------
        # 逐帧跟踪：三个视角完全独立
        # ------------------------------------------------------------
        for frame_num, frame_path in enumerate(seq_a.frames[1:], start=1):
            image_a = self._read_image(frame_path)
            image_b = self._read_image(seq_b.frames[frame_num])
            image_c = self._read_image(seq_c.frames[frame_num])

            info_a = seq_a.frame_info(frame_num)
            info_b = seq_b.frame_info(frame_num)
            info_c = seq_c.frame_info(frame_num)

            info_a['previous_output'] = prev_output_a
            info_b['previous_output'] = prev_output_b
            info_c['previous_output'] = prev_output_c

            if len(seq_a.ground_truth_rect) > 1:
                info_a['gt_bbox'] = seq_a.ground_truth_rect[frame_num]
            if len(seq_b.ground_truth_rect) > 1:
                info_b['gt_bbox'] = seq_b.ground_truth_rect[frame_num]
            if len(seq_c.ground_truth_rect) > 1:
                info_c['gt_bbox'] = seq_c.ground_truth_rect[frame_num]

            if pcum_remote_enabled:
                start_time_a = time.time()
                candidate_a = tracker.pcum_local_candidate(image_a)
                local_time_a = time.time() - start_time_a

                start_time_b = time.time()
                candidate_b = tracker2.pcum_local_candidate(image_b)
                local_time_b = time.time() - start_time_b

                start_time_c = time.time()
                candidate_c = tracker3.pcum_local_candidate(image_c)
                local_time_c = time.time() - start_time_c

                candidates = [candidate_a, candidate_b, candidate_c]

                remote_prompts_a, remote_state_a = _remote_inputs_for(0, candidates, tracker)
                remote_prompts_b, remote_state_b = _remote_inputs_for(1, candidates, tracker2)
                remote_prompts_c, remote_state_c = _remote_inputs_for(2, candidates, tracker3)

                start_time_a = time.time()
                out_a, max_score_a, response_APCE_a = tracker.pcum_track_with_remote(
                    image_a,
                    info=info_a,
                    remote_prompts=remote_prompts_a,
                    remote_states=remote_state_a,
                    local_candidate=candidate_a,
                    debug_name="a"
                )
                time_a = local_time_a + (time.time() - start_time_a)

                start_time_b = time.time()
                out_b, max_score_b, response_APCE_b = tracker2.pcum_track_with_remote(
                    image_b,
                    info=info_b,
                    remote_prompts=remote_prompts_b,
                    remote_states=remote_state_b,
                    local_candidate=candidate_b,
                    debug_name="b"
                )
                time_b = local_time_b + (time.time() - start_time_b)

                start_time_c = time.time()
                out_c, max_score_c, response_APCE_c = tracker3.pcum_track_with_remote(
                    image_c,
                    info=info_c,
                    remote_prompts=remote_prompts_c,
                    remote_states=remote_state_c,
                    local_candidate=candidate_c,
                    debug_name="c"
                )
                time_c = local_time_c + (time.time() - start_time_c)

                score_a_val = _to_float(max_score_a)
                score_b_val = _to_float(max_score_b)
                score_c_val = _to_float(max_score_c)

                apce_a_val = _to_float(response_APCE_a)
                apce_b_val = _to_float(response_APCE_b)
                apce_c_val = _to_float(response_APCE_c)

                payload_a = _payload(out_a, score_a_val, apce_a_val)
                payload_b = _payload(out_b, score_b_val, apce_b_val)
                payload_c = _payload(out_c, score_c_val, apce_c_val)
                _exchange_messages(frame_num, {0: payload_a, 1: payload_b, 2: payload_c})

            else:
                # A 机单机跟踪
                start_time_a = time.time()
                out_a, max_score_a, response_APCE_a = tracker.Fusetrack(image_a, info_a)
                time_a = time.time() - start_time_a

                # B 机单机跟踪
                start_time_b = time.time()
                out_b, max_score_b, response_APCE_b = tracker2.Fusetrack(image_b, info_b)
                time_b = time.time() - start_time_b

                # C 机单机跟踪
                start_time_c = time.time()
                out_c, max_score_c, response_APCE_c = tracker3.Fusetrack(image_c, info_c)
                time_c = time.time() - start_time_c

                score_a_val = _to_float(max_score_a)
                score_b_val = _to_float(max_score_b)
                score_c_val = _to_float(max_score_c)

                apce_a_val = _to_float(response_APCE_a)
                apce_b_val = _to_float(response_APCE_b)
                apce_c_val = _to_float(response_APCE_c)

                payload_a = _payload(out_a, score_a_val, apce_a_val)
                payload_b = _payload(out_b, score_b_val, apce_b_val)
                payload_c = _payload(out_c, score_c_val, apce_c_val)

                delivered_payloads = _exchange_messages(
                    frame_num,
                    {
                        0: payload_a,
                        1: payload_b,
                        2: payload_c,
                    }
                )

                if coop_enabled:
                    peer_payloads_a = delivered_payloads[0] if coop_fusion == "prompt" else []
                    peer_payloads_b = delivered_payloads[1] if coop_fusion == "prompt" else []
                    peer_payloads_c = delivered_payloads[2] if coop_fusion == "prompt" else []
                else:
                    peer_payloads_a = [payload_b, payload_c]
                    peer_payloads_b = [payload_a, payload_c]
                    peer_payloads_c = [payload_a, payload_b]

                out_a, score_a_val, apce_a_val = _maybe_prompt_refine(
                    tracker, image_a, info_a, out_a, score_a_val, apce_a_val,
                    peer_payloads_a
                )
                out_b, score_b_val, apce_b_val = _maybe_prompt_refine(
                    tracker2, image_b, info_b, out_b, score_b_val, apce_b_val,
                    peer_payloads_b
                )
                out_c, score_c_val, apce_c_val = _maybe_prompt_refine(
                    tracker3, image_c, info_c, out_c, score_c_val, apce_c_val,
                    peer_payloads_c
                )

            prev_output_a = OrderedDict(out_a)
            prev_output_b = OrderedDict(out_b)
            prev_output_c = OrderedDict(out_c)

            _store_outputs(
                output_a,
                out_a,
                {
                    'time': time_a,
                    'max_score': score_a_val,
                    'APCE': apce_a_val
                }
            )

            _store_outputs(
                output_b,
                out_b,
                {
                    'time': time_b,
                    'max_score': score_b_val,
                    'APCE': apce_b_val
                }
            )

            _store_outputs(
                output_c,
                out_c,
                {
                    'time': time_c,
                    'max_score': score_c_val,
                    'APCE': apce_c_val
                }
            )

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_a and len(output_a[key]) <= 1:
                output_a.pop(key)

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_b and len(output_b[key]) <= 1:
                output_b.pop(key)

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_c and len(output_c[key]) <= 1:
                output_c.pop(key)

        _save_comm_stats(len(output_a.get('time', [])))

        return output_a, output_b, output_c



# 三机融合结果，并且三机匹配，当目标丢失时把Search region映射过去
    def Fuse_three_multi_track_matching_sequence(self, tracker, tracker2, tracker3, seq_a, seq_b, seq_c, init_info_a, init_info_b,init_info_c):
        output_a = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}
        output_b = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}
        output_c = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}

        if tracker.params.save_all_boxes:
            output_a['all_boxes'] = []
            output_a['all_scores'] = []

        if tracker2.params.save_all_boxes:
            output_b['all_boxes'] = []
            output_b['all_scores'] = []

        if tracker3.params.save_all_boxes:
            output_c['all_boxes'] = []
            output_c['all_scores'] = []

        def _store_outputs(output,tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image_a = self._read_image(seq_a.frames[0])
        image_b = self._read_image(seq_b.frames[0])
        image_c = self._read_image(seq_c.frames[0])

        start_time = time.time()

        out_a = tracker.three_multi_initialize(image_a, image_b, image_c, init_info_a, init_info_b, init_info_c)
        out_b = tracker2.three_multi_initialize(image_b, image_a, image_c, init_info_b, init_info_a, init_info_c)
        out_c = tracker3.three_multi_initialize(image_c, image_a, image_b, init_info_c, init_info_a, init_info_b)

        if out_a is None: out_a = {}
        if out_b is None: out_b = {}
        if out_c is None: out_c = {}

        prev_output_a = OrderedDict(out_a)
        init_default_a = {'target_bbox': init_info_a.get('init_bbox'), 'time': time.time() - start_time, 'max_score': 0, 'APCE':0}
        prev_output_b = OrderedDict(out_b)
        init_default_b = {'target_bbox': init_info_b.get('init_bbox'), 'time': time.time() - start_time, 'max_score': 0, 'APCE':0}
        prev_output_c = OrderedDict(out_c)
        init_default_c = {'target_bbox': init_info_c.get('init_bbox'), 'time': time.time() - start_time, 'max_score': 0, 'APCE':0}
                        
        if tracker.params.save_all_boxes:
            init_default_a['all_boxes'] = out_a['all_boxes']
            init_default_a['all_scores'] = out_a['all_scores']

        if tracker2.params.save_all_boxes:
            init_default_b['all_boxes'] = out_b['all_boxes']
            init_default_b['all_scores'] = out_b['all_scores']
        
        if tracker3.params.save_all_boxes:
            init_default_c['all_boxes'] = out_c['all_boxes']
            init_default_c['all_scores'] = out_c['all_scores']

        _store_outputs(output_a, out_a, init_default_a)
        _store_outputs(output_b, out_b, init_default_b)
        _store_outputs(output_c, out_c, init_default_c)

        # 💡【核心修改】：中央司令部初始化，存放“上一帧全场最佳特征”
        global_best_prompt = None

        for frame_num, frame_path in enumerate(seq_a.frames[1:], start=1):
            image_a = self._read_image(frame_path)
            image_b = self._read_image(seq_b.frames[frame_num])
            image_c = self._read_image(seq_c.frames[frame_num])

            info_a = seq_a.frame_info(frame_num)
            info_a['previous_output'] = prev_output_a
            info_b = seq_b.frame_info(frame_num)
            info_b['previous_output'] = prev_output_b
            info_c = seq_c.frame_info(frame_num)
            info_c['previous_output'] = prev_output_c

            if len(seq_a.ground_truth_rect) > 1: info_a['gt_bbox'] = seq_a.ground_truth_rect[frame_num]
            if len(seq_b.ground_truth_rect) > 1: info_b['gt_bbox'] = seq_b.ground_truth_rect[frame_num]
            if len(seq_c.ground_truth_rect) > 1: info_c['gt_bbox'] = seq_c.ground_truth_rect[frame_num]

            # 💡【核心修改】：时间独立开来计算，防止时间累加导致FPS降低！同时喂入 prompt_input
            start_time_a = time.time()
            out_a, max_score_a, response_APCE_a, prompt_a = tracker.three_nomulti_Fusetrack(image_a, image_b, image_c, "a", info_a, info_b, info_c, prompt_input=global_best_prompt)
            time_a = time.time() - start_time_a

            start_time_b = time.time()
            out_b, max_score_b, response_APCE_b, prompt_b = tracker2.three_nomulti_Fusetrack(image_b, image_a, image_c, "b", info_b, info_a, info_c, prompt_input=global_best_prompt)
            time_b = time.time() - start_time_b

            start_time_c = time.time()
            out_c, max_score_c, response_APCE_c, prompt_c = tracker3.three_nomulti_Fusetrack(image_c, image_a, image_b, "c", info_c, info_a, info_b, prompt_input=global_best_prompt)
            time_c = time.time() - start_time_c

            state_a = copy.deepcopy(tracker.state)
            state_b = copy.deepcopy(tracker2.state)
            state_c = copy.deepcopy(tracker3.state)

            # 安全剥离张量
            score_a_val = max_score_a.item() if torch.is_tensor(max_score_a) else max_score_a
            score_b_val = max_score_b.item() if torch.is_tensor(max_score_b) else max_score_b
            score_c_val = max_score_c.item() if torch.is_tensor(max_score_c) else max_score_c

            apce_a_val = response_APCE_a.item() if torch.is_tensor(response_APCE_a) else response_APCE_a
            apce_b_val = response_APCE_b.item() if torch.is_tensor(response_APCE_b) else response_APCE_b
            apce_c_val = response_APCE_c.item() if torch.is_tensor(response_APCE_c) else response_APCE_c

            # 💡【核心修改】：华山论剑，谁分数高，谁的 prompt 在下一帧掌权
            best_score = max(score_a_val, score_b_val, score_c_val)
            if best_score == score_a_val and prompt_a is not None:
                global_best_prompt = prompt_a.detach()
            elif best_score == score_b_val and prompt_b is not None:
                global_best_prompt = prompt_b.detach()
            elif prompt_c is not None:
                global_best_prompt = prompt_c.detach()

            #######################################################  跨机重检测  ##########################################
            redet_factor_list = [[4,12], [3,9], [2,5]]

            if score_a_val > max(score_b_val, score_c_val):
                tmp_max_score = copy.deepcopy(score_a_val)
                tmp_APEC = copy.deepcopy(apce_a_val)
                tmp_image = copy.deepcopy(image_a)
                tmp_state = copy.deepcopy(state_a)
                tmp_info = copy.deepcopy(info_a)
            elif score_b_val > max(score_a_val, score_c_val):
                tmp_max_score = copy.deepcopy(score_b_val)
                tmp_APEC = copy.deepcopy(apce_b_val)
                tmp_image = copy.deepcopy(image_b)
                tmp_state = copy.deepcopy(state_b)
                tmp_info = copy.deepcopy(info_b)
            else:
                tmp_max_score = copy.deepcopy(score_c_val)
                tmp_APEC = copy.deepcopy(apce_c_val)
                tmp_image = copy.deepcopy(image_c)
                tmp_state = copy.deepcopy(state_c)
                tmp_info = copy.deepcopy(info_c)

            # 💡 【核心重检测 A机】
            if((score_a_val < 0.2 and apce_a_val < 100) and (tmp_max_score > 0.3) and (tmp_APEC > apce_a_val)):
                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker.state = copy.deepcopy(state_a)
                    s_re = time.time()
                    out_a_tmp, max_score_a_tmp, response_APCE_a_tmp = tracker.three_search_redetect(image_a, tmp_image, "a", copy.deepcopy(tmp_state), factor[0], factor[1], info_a, tmp_info)
                    time_a += (time.time() - s_re) # 将重检测的时间累加回去

                    sc_tmp = max_score_a_tmp.item() if torch.is_tensor(max_score_a_tmp) else max_score_a_tmp
                    ap_tmp = response_APCE_a_tmp.item() if torch.is_tensor(response_APCE_a_tmp) else response_APCE_a_tmp
                    tmp_dict = {"out_a":out_a_tmp, "max_score_a":sc_tmp, "response_APCE_a":ap_tmp}
                    redet_results.append(tmp_dict)
                    
                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_a"] > ms:
                        ms = result_dict["max_score_a"]
                        label = i

                if redet_results[label]["max_score_a"] - score_a_val > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_a, score_a_val, apce_a_val = redet_results[label]["out_a"], redet_results[label]["max_score_a"], redet_results[label]["response_APCE_a"]
                    tracker.state = out_a["target_bbox"]
                else:
                    print("remain ori")
                    tracker.state = copy.deepcopy(state_a)


            # 💡 【核心重检测 B机】
            if((score_b_val < 0.2 and apce_b_val < 100) and (tmp_max_score > 0.3) and (tmp_APEC > apce_b_val)):
                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker2.state = copy.deepcopy(state_b)
                    s_re = time.time()
                    out_b_tmp, max_score_b_tmp, response_APCE_b_tmp = tracker2.three_search_redetect(image_b, tmp_image, "b", copy.deepcopy(tmp_state), factor[0], factor[1], info_b, tmp_info)
                    time_b += (time.time() - s_re)

                    sc_tmp = max_score_b_tmp.item() if torch.is_tensor(max_score_b_tmp) else max_score_b_tmp
                    ap_tmp = response_APCE_b_tmp.item() if torch.is_tensor(response_APCE_b_tmp) else response_APCE_b_tmp
                    tmp_dict = {"out_b":out_b_tmp, "max_score_b":sc_tmp, "response_APCE_b":ap_tmp}
                    redet_results.append(tmp_dict)
                    
                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_b"] > ms:
                        ms = result_dict["max_score_b"]
                        label = i

                if redet_results[label]["max_score_b"] - score_b_val > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_b, score_b_val, apce_b_val = redet_results[label]["out_b"], redet_results[label]["max_score_b"], redet_results[label]["response_APCE_b"]
                    tracker2.state = out_b["target_bbox"]
                else:
                    print("remain ori")
                    tracker2.state = copy.deepcopy(state_b)


            # 💡 【核心重检测 C机】
            if((score_c_val < 0.2 and apce_c_val < 100) and (tmp_max_score > 0.3) and (tmp_APEC > apce_c_val)):
                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker3.state = copy.deepcopy(state_c)
                    s_re = time.time()
                    out_c_tmp, max_score_c_tmp, response_APCE_c_tmp = tracker3.three_search_redetect(image_c, tmp_image, "c", copy.deepcopy(tmp_state), factor[0], factor[1], info_c, tmp_info)
                    time_c += (time.time() - s_re)

                    sc_tmp = max_score_c_tmp.item() if torch.is_tensor(max_score_c_tmp) else max_score_c_tmp
                    ap_tmp = response_APCE_c_tmp.item() if torch.is_tensor(response_APCE_c_tmp) else response_APCE_c_tmp
                    tmp_dict = {"out_c":out_c_tmp, "max_score_c":sc_tmp, "response_APCE_c":ap_tmp}
                    redet_results.append(tmp_dict)
                    
                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_c"] > ms:
                        ms = result_dict["max_score_c"]
                        label = i

                if redet_results[label]["max_score_c"] - score_c_val > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_c, score_c_val, apce_c_val = redet_results[label]["out_c"], redet_results[label]["max_score_c"], redet_results[label]["response_APCE_c"]
                    tracker3.state = out_c["target_bbox"]
                else:
                    print("remain ori")
                    tracker3.state = copy.deepcopy(state_c)


            prev_output_a = OrderedDict(out_a)
            _store_outputs(output_a, out_a, {'time': time_a, 'max_score': score_a_val,  'APCE': apce_a_val})
            prev_output_b = OrderedDict(out_b)
            _store_outputs(output_b, out_b, {'time': time_b, 'max_score': score_b_val, 'APCE': apce_b_val})
            prev_output_c = OrderedDict(out_c)
            _store_outputs(output_c, out_c, {'time': time_c, 'max_score': score_c_val, 'APCE': apce_c_val})


        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_a and len(output_a[key]) <= 1:
                output_a.pop(key)
        
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_b and len(output_b[key]) <= 1:
                output_b.pop(key)

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_c and len(output_c[key]) <= 1:
                output_c.pop(key)

        return output_a, output_b, output_c

    def moe_run_sequence(self, seq, debug=None):
        """Run tracker on sequence."""
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self.moe_track_sequence(tracker, seq, init_info)
        return output


    def moe_track_sequence(self, tracker, seq, init_info):

        output = {'target_bbox': [],
                'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output
