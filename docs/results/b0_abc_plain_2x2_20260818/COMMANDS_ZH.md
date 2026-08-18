# 实际运行命令

工作目录均为 `/data/zjy/EnTeR-Track-main`，Python 环境为 `/home/user/.conda/envs/zjy`。

## E1 formal

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e1-formal-r002 PYTHONUNBUFFERED=1 /home/user/.conda/envs/zjy/bin/torchrun --standalone --nproc_per_node=4 lib/train/run_training.py --script entertrack --config b0_abc_plain_4gpu_ep50 --save_dir output/diagnostics/b0_abc_plain_long/e1_run_20260818_seed42_4gpu_r002 --seed 42 --use_wandb 0
```

## E2 formal

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e2-formal-r001 PYTHONUNBUFFERED=1 /home/user/.conda/envs/zjy/bin/torchrun --standalone --nproc_per_node=4 lib/train/run_training.py --script entertrack --config b0_abc_plain_ind_sampler_4gpu --save_dir output/diagnostics/b0_abc_plain_ind_sampler/e2_run_20260818_seed42_4gpu_r001 --seed 42 --use_wandb 0
```

E2 inner-val 使用 checkpoint override、runid `28225`、`--no_gt_inference 1`、`--fail_if_results_exist 1`，并由 `tracking/verify_pcum_test_run.py` 验证为 15/15。

## E3 formal

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-formal-r001 PYTHONUNBUFFERED=1 /home/user/.conda/envs/zjy/bin/torchrun --standalone --nproc_per_node=4 lib/train/run_training.py --script entertrack --config b0_abc_plain_ind_sampler_4gpu_ep50 --save_dir output/diagnostics/b0_abc_plain_ind_sampler_long/e3_run_20260818_seed42_4gpu_r001 --seed 42 --use_wandb 0
```

E3 固定 inner-val 列表由下列命令实际执行：

```bash
/bin/bash scripts/run_b0_abc_plain_e3_inner_val.sh
```

对应 runid 为 ep30/35/40/45/50 -> `28330/28335/28340/28345/28350`；ep25 使用网络权重 hash 完全相同的 E2 runid `28225`。

标准指标导出示例：

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py --tracker_name entertrack --tracker_param b0_abc_plain_ind_sampler_4gpu_ep50 --dataset_name threemdot_val --runid 28350 --output_dir docs/results/b0_abc_plain_2x2_20260818/E3/inner_val_ep0050
```
