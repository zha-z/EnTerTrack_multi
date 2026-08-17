import os
import sys
import argparse

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

if cv2 is not None:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

import torch
torch.set_num_threads(1)

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset,run_mdot_dataset_three
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.run_id import parse_run_id_argument


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb',
                sequence=None, debug=0, threads=0, num_gpus=8,
                checkpoint=None, fail_if_results_exist=False,
                no_gt_inference=False, c3r_instrumentation=False,
                instrumentation_fold_id=-1):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:#跟踪单个序列
        dataset = [dataset[sequence]]

    trackers = [Tracker(
        tracker_name,
        tracker_param,
        dataset_name,
        run_id,
        checkpoint_override=checkpoint,
        no_gt_inference=no_gt_inference,
        c3r_instrumentation=c3r_instrumentation,
        instrumentation_fold_id=instrumentation_fold_id,
    )]

    if fail_if_results_exist:
        trackers[0].reserve_results_dir()

    #run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)
    run_mdot_dataset_three(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name_pos', nargs='?', default=None)
    parser.add_argument('tracker_param_pos', nargs='?', default=None)
    parser.add_argument('--tracker_name', type=str, default=None, help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default=None, help='Name of config file.')
    parser.add_argument(
        '--runid', type=parse_run_id_argument, default=14,
        help='Numeric legacy runid or safe formal string runid.')
    parser.add_argument('--dataset_name', '--dataset', dest='dataset_name', type=str,
                        default='threemdot_test', help='Dataset name.')
    parser.add_argument('--sequence', type=str, default=None,help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=12, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=6)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fail_if_results_exist', type=int, choices=(0, 1), default=0)
    parser.add_argument('--no_gt_inference', type=int, choices=(0, 1), default=0)
    parser.add_argument('--c3r_instrumentation', type=int, choices=(0, 1), default=0)
    parser.add_argument('--instrumentation_fold_id', type=int, default=-1)

    args = parser.parse_args()
    tracker_name = args.tracker_name or args.tracker_name_pos or 'entertrack'
    tracker_param = args.tracker_param or args.tracker_param_pos or 'entertrack_threemdot_lasot_ft_cons'

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(
        tracker_name,
        tracker_param,
        args.runid,
        args.dataset_name,
        seq_name,
        args.debug,
        args.threads,
        num_gpus=args.num_gpus,
        checkpoint=args.checkpoint,
        fail_if_results_exist=bool(args.fail_if_results_exist),
        no_gt_inference=bool(args.no_gt_inference),
        c3r_instrumentation=bool(args.c3r_instrumentation),
        instrumentation_fold_id=args.instrumentation_fold_id,
    )


if __name__ == '__main__':
    main()
