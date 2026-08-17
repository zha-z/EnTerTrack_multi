import os
import argparse
import random
import sys

from pathlib import Path
import os
import random
import warnings

import numpy as np
import torch


warnings.filterwarnings(
    "ignore",
    message=r".*grid_sampler_2d_backward_cuda does not have a deterministic implementation.*",
    category=UserWarning,
)


def setup_reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 对没有确定性实现的算子警告后继续，而不是终止训练
    torch.use_deterministic_algorithms(True, warn_only=True)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str,default='ostrack', help='training script name')
    parser.add_argument('--config', type=str, default='vits_256_mae_32x4_ep300', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, default="../output/",help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="single",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int,default=1, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=1)  # whether to use wandb
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate the resolved config and training plan without training')

    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.script == "entertrack" and args.config == "fcvc_full":
        import torch.distributed as dist

        from lib.train.admin.settings import Settings
        from lib.train.train_script import run

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank >= 0 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
        root = Path(__file__).resolve().parents[1]
        settings = Settings()
        settings.script_name = args.script
        settings.config_name = args.config
        settings.cfg_file = str(root / "experiments" / args.script / (args.config + ".yaml"))
        settings.save_dir = args.save_dir
        settings.mode = args.mode
        settings.resume = args.resume
        settings.device_name = args.device
        settings.audit = bool(args.audit)
        settings.dry_run = bool(args.dry_run)
        settings.local_rank = local_rank
        try:
            run(settings)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return
    if args.mode == "single":
        train_cmd = "python lib/train/run_training.py --script %s --config %s --save_dir %s --use_lmdb %d " \
                    "--script_prv %s --config_prv %s --distill %d --script_teacher %s --config_teacher %s --use_wandb %d"\
                    % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                       args.distill, args.script_teacher, args.config_teacher, args.use_wandb)
    elif args.mode == "multiple":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_port %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
                    "--distill %d --script_teacher %s --config_teacher %s" \
                    % (args.nproc_per_node, random.randint(10000, 50000), args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    elif args.mode == "multi_node":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_addr %s --master_port %d --nnodes %d --node_rank %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
                    "--distill %d --script_teacher %s --config_teacher %s" \
                    % (args.nproc_per_node, args.ip, args.port, args.world_size, args.rank, args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
