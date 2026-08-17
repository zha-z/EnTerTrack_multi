import numpy as np
import multiprocessing
import os
import sys
import csv
import json
import gzip
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
from lib.test.utils.pcum_diagnostics import DIAGNOSTIC_COLUMNS, diagnostic_filename
from lib.test.tracker.motion_state import (
    motion_diagnostics_file,
    save_motion_diagnostics,
)
from lib.test.tracker.mcr_redetection import save_mcr_diagnostics
import torch
from lib.test.utils.c3r_inference import C3R_DIAGNOSTIC_COLUMNS


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    if seq.dataset in ['trackingnet', 'got10k']:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    '''2021.1.5 create new folder for these two datasets'''
    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def save_float_matrix(file, data):
        values = np.asarray(data, dtype=float)
        np.savetxt(file, values, delimiter='\t', fmt='%.6f')

    def save_frame_diagnostics(file, data):
        with open(file, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            writer.writerows(data)

    def save_c3r_diagnostics(file, data):
        with open(file, 'w', newline='') as fh:
            writer = csv.DictWriter(
                fh, fieldnames=C3R_DIAGNOSTIC_COLUMNS, extrasaction='raise')
            writer.writeheader()
            writer.writerows(data)

    def save_c3r_instrumentation(file, data):
        with gzip.open(file, 'wt', encoding='utf-8') as fh:
            for item in data:
                rows = item if isinstance(item, list) else [item]
                for row in rows:
                    if row:
                        json.dump(row, fh, sort_keys=True, separators=(',', ':'))
                        fh.write('\n')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key == 'max_score':    # 保存score
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_max_score.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_max_score.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key == 'APCE':    # 保存APCE
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_APCE.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving APCE...")
                bbox_file = '{}_APCE.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key == 'pcum_decision':
            decision_file = '{}_pcum_decision.txt'.format(base_results_path)
            save_float_matrix(decision_file, data)

        if key == 'pcum_remote_weights':
            weight_file = '{}_pcum_remote_weights.txt'.format(base_results_path)
            save_float_matrix(weight_file, data)

        if key == 'pcum_remote_suppression':
            suppression_file = '{}_pcum_remote_suppression.txt'.format(
                base_results_path)
            save_float_matrix(suppression_file, data)

        if key == 'pcum_selector':
            selector_file = '{}_pcum_selector.txt'.format(base_results_path)
            save_float_matrix(selector_file, data)

        if key == 'pcum_frame_diagnostics':
            uav_id = data[0].get('current_uav', 'unknown')
            diagnostics_file = os.path.join(
                tracker.results_dir,
                diagnostic_filename(
                    tracker.name,
                    tracker.parameter_name,
                    tracker.run_id,
                    seq.name,
                    uav_id,
                ),
            )
            save_frame_diagnostics(diagnostics_file, data)

        if key == 'motion_state_diagnostics':
            save_motion_diagnostics(tracker.results_dir, seq.name, data)

        if key == 'mcr_diagnostics':
            save_mcr_diagnostics(tracker.results_dir, seq.name, data)

        if key == 'c3r_diagnostics':
            diagnostics_file = '{}_c3r_diagnostics.csv'.format(
                base_results_path)
            save_c3r_diagnostics(diagnostics_file, data)

        if key == 'c3r_comm_summary':
            summary_file = '{}_c3r_comm_summary.json'.format(base_results_path)
            with open(summary_file, 'w') as fh:
                json.dump(data, fh, indent=2, sort_keys=True)

        if key == 'c3r_source_instrumentation':
            instrumentation_file = '{}_c3r_source_instrumentation.jsonl.gz'.format(
                base_results_path)
            save_c3r_instrumentation(instrumentation_file, data)

        if key == 'c3r_aggregate_instrumentation':
            instrumentation_file = '{}_c3r_aggregate_instrumentation.jsonl.gz'.format(
                base_results_path)
            save_c3r_instrumentation(instrumentation_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)

# 输出max_score
def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        # output = tracker.run_sequence(seq, debug=debug)
        output = tracker.Fuse_run_sequence(seq, debug=debug)
    else:
        try:
            # output = tracker.run_sequence(seq, debug=debug)
            output = tracker.Fuse_run_sequence(seq, debug=debug)   # 保存
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')


# moe的LaSOT数据集运行
def run_moe_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_moe_sequence(seq, tracker_info, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_moe_sequence, param_list)
    print('Done')


# moe
def run_moe_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        # output = tracker.run_sequence(seq, debug=debug)
        output = tracker.moe_run_sequence(seq, debug=debug)
    else:
        try:
            # output = tracker.run_sequence(seq, debug=debug)
            output = tracker.moe_run_sequence(seq, debug=debug)   # 保存
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)





# mdot数据集运行
def run_mdot_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    len_data = len(dataset) // 2
    dataset_A = dataset[:len_data]
    dataset_B = dataset[len_data:]



    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq_a, seq_b in zip(dataset_A, dataset_B):
            for tracker_info in trackers:
                run_multi_sequence(seq_a, seq_b, tracker_info, debug=debug)    # 将双机对应的sequence传入
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')



# 在多机上追踪
def run_multi_sequence(seq_a: Sequence, seq_b: Sequence , tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist_a():
        if seq_a.object_ids is None:
            bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq_a.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq_a.name, obj_id) for obj_id in seq_a.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist_a() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq_a.name))

    if debug:
        output_a, output_b = tracker.Fuse_multi_run_sequence(seq_a, seq_b, debug=debug)
    else:
        try:
            output_a, output_b = tracker.Fuse_multi_run_sequence(seq_a, seq_b, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output_a['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output_a['time']])
        num_frames = len(output_a['time'])
    else:
        exec_time = sum(output_a['time'])
        num_frames = len(output_a['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq_a, tracker, output_a)
        _save_tracker_output(seq_b, tracker, output_b)


def three_view_triplets(dataset):
    """Bind Three-MDOT views by target identity, independent of list order."""
    groups = OrderedDict()
    for sequence in dataset:
        try:
            target, view_text = sequence.name.rsplit('-', 1)
            view = int(view_text)
        except (AttributeError, TypeError, ValueError):
            raise ValueError(
                "Three-MDOT sequence name must end in -1, -2, or -3: {}".format(
                    getattr(sequence, 'name', sequence)))
        if view not in (1, 2, 3):
            raise ValueError("Unsupported Three-MDOT view: {}".format(sequence.name))
        views = groups.setdefault(target, {})
        if view in views:
            raise ValueError("Duplicate Three-MDOT target/view: {}".format(sequence.name))
        views[view] = sequence

    incomplete = {
        target: sorted(views) for target, views in groups.items()
        if set(views) != {1, 2, 3}
    }
    if incomplete:
        raise ValueError("Incomplete Three-MDOT target groups: {}".format(incomplete))
    return [(views[1], views[2], views[3]) for views in groups.values()]


# Three mdot数据集运行
def run_mdot_dataset_three(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    seq_triplets = three_view_triplets(dataset)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq_a, seq_b, seq_c in seq_triplets:
            for tracker_info in trackers:
                run_three_multi_sequence(seq_a, seq_b, seq_c, tracker_info, debug=debug)    # 将双机对应的sequence传入
    elif mode == 'parallel':
        param_list = [(seq_a, seq_b, seq_c, tracker_info, debug, num_gpus) 
                      for (seq_a, seq_b, seq_c), tracker_info in product(seq_triplets, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_three_multi_sequence, param_list)
    print('Done')


# 在多机上追踪
def run_three_multi_sequence(seq_a: Sequence, seq_b: Sequence ,seq_c: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _pcum_decision_log_enabled():
        if not hasattr(tracker, "_save_pcum_decision_log"):
            try:
                params = tracker.get_parameters()
                pcum_cfg = getattr(params.cfg.TEST, "PCUM", None)
                tracker._save_pcum_decision_log = bool(
                    getattr(pcum_cfg, "SAVE_DECISION_LOG", False)
                )
            except Exception:
                tracker._save_pcum_decision_log = False
        return tracker._save_pcum_decision_log

    def _pcum_frame_diagnostics_enabled():
        if not hasattr(tracker, "_save_pcum_frame_diagnostics"):
            try:
                params = tracker.get_parameters()
                diagnostics_cfg = getattr(
                    getattr(params.cfg.TEST.PCUM, "FRAME_DIAGNOSTICS", None),
                    "ENABLED",
                    False,
                )
                tracker._save_pcum_frame_diagnostics = bool(diagnostics_cfg)
            except Exception:
                tracker._save_pcum_frame_diagnostics = False
        return tracker._save_pcum_frame_diagnostics

    def _motion_state_log_enabled():
        if not hasattr(tracker, "_save_motion_state_diagnostics"):
            try:
                params = tracker.get_parameters()
                motion_cfg = getattr(params.cfg.TEST, "MOTION_STATE", None)
                tracker._save_motion_state_diagnostics = bool(
                    getattr(motion_cfg, "ENABLED", False)
                    and getattr(motion_cfg, "LOG_ENABLED", False)
                )
            except Exception:
                tracker._save_motion_state_diagnostics = False
        return tracker._save_motion_state_diagnostics

    def _c3r_instrumentation_enabled():
        return bool(getattr(tracker, "c3r_instrumentation", False))

    def _sequence_results_exist(
        seq,
        need_decision_log=False,
        need_frame_diagnostics=False,
        need_motion_diagnostics=False,
        need_c3r_instrumentation=False,
        uav_id="unknown",
    ):
        if seq.object_ids is None:
            bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            if not os.path.isfile(bbox_file):
                return False
            if need_decision_log:
                decision_file = '{}/{}_pcum_decision.txt'.format(tracker.results_dir, seq.name)
                if not os.path.isfile(decision_file):
                    return False
            if need_frame_diagnostics:
                frame_file = os.path.join(
                    tracker.results_dir,
                    diagnostic_filename(
                        tracker.name,
                        tracker.parameter_name,
                        tracker.run_id,
                        seq.name,
                        uav_id,
                    ),
                )
                if not os.path.isfile(frame_file):
                    return False
            if need_motion_diagnostics and not os.path.isfile(
                motion_diagnostics_file(tracker.results_dir, seq.name)
            ):
                return False
            if need_c3r_instrumentation:
                for suffix in (
                        "_c3r_source_instrumentation.jsonl.gz",
                        "_c3r_aggregate_instrumentation.jsonl.gz"):
                    if not os.path.isfile('{}/{}{}'.format(
                            tracker.results_dir, seq.name, suffix)):
                        return False
            return True

        bbox_files = [
            '{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id)
            for obj_id in seq.object_ids
        ]
        missing = [not os.path.isfile(f) for f in bbox_files]
        if sum(missing) != 0:
            return False
        if need_decision_log:
            decision_file = '{}/{}_pcum_decision.txt'.format(tracker.results_dir, seq.name)
            if not os.path.isfile(decision_file):
                return False
        if need_frame_diagnostics:
            frame_file = os.path.join(
                tracker.results_dir,
                diagnostic_filename(
                    tracker.name,
                    tracker.parameter_name,
                    tracker.run_id,
                    seq.name,
                    uav_id,
                ),
            )
            if not os.path.isfile(frame_file):
                return False
        if need_motion_diagnostics and not os.path.isfile(
            motion_diagnostics_file(tracker.results_dir, seq.name)
        ):
            return False
        if need_c3r_instrumentation:
            for suffix in (
                    "_c3r_source_instrumentation.jsonl.gz",
                    "_c3r_aggregate_instrumentation.jsonl.gz"):
                if not os.path.isfile('{}/{}{}'.format(
                        tracker.results_dir, seq.name, suffix)):
                    return False
        return True

    def _results_exist_a():
        need_decision_log = _pcum_decision_log_enabled()
        need_frame_diagnostics = _pcum_frame_diagnostics_enabled()
        need_motion_diagnostics = _motion_state_log_enabled()
        need_c3r_instrumentation = _c3r_instrumentation_enabled()
        if (need_decision_log or need_frame_diagnostics
                or need_motion_diagnostics or need_c3r_instrumentation):
            return all(
                _sequence_results_exist(
                    seq,
                    need_decision_log=need_decision_log,
                    need_frame_diagnostics=need_frame_diagnostics,
                    need_motion_diagnostics=need_motion_diagnostics,
                    need_c3r_instrumentation=need_c3r_instrumentation,
                    uav_id=uav_id,
                )
                for seq, uav_id in ((seq_a, "A"), (seq_b, "B"), (seq_c, "C"))
            )
        return _sequence_results_exist(seq_a)

    if _results_exist_a() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq_a.name))

    if debug:
        output_a, output_b, output_c = tracker.Fuse_three_multi_run_sequence(seq_a, seq_b, seq_c, debug=debug)
    else:
        try:
            output_a, output_b, output_c = tracker.Fuse_three_multi_run_sequence(seq_a, seq_b, seq_c, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output_a['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output_a['time']])
        num_frames = len(output_a['time'])
    else:
        exec_time = sum(output_a['time'])
        num_frames = len(output_a['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq_a, tracker, output_a)
        _save_tracker_output(seq_b, tracker, output_b)
        _save_tracker_output(seq_c, tracker, output_c)
