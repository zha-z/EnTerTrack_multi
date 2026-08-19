# B1-ABC-ARP 执行命令

工作目录均为 `/data/zjy/EnTeR-Track-main`，Conda Python 为 `/home/user/.conda/envs/zjy/bin/python`。

## Stage A：已有 checkpoint 补测

```bash
/bin/bash scripts/run_b1_abc_arp_stage_a.sh
```

脚本对 E1/E3 的 ep26/27/28/29 分别执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack --tracker_param <E1或E3配置> \
  --dataset_name threemdot_val --runid <固定runid> --threads 12 --num_gpus 4 \
  --checkpoint <既有checkpoint> --fail_if_results_exist 1 --no_gt_inference 1

PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python tracking/verify_pcum_test_run.py \
  --config <配置> --dataset threemdot_val --runid <runid> --log <log> \
  --expected-source none --expected-checkpoint <checkpoint> \
  --expected-epoch <epoch> --require-no-pcum

PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py \
  --tracker_name entertrack --tracker_param <配置> \
  --dataset_name threemdot_val --runid <runid> --output_dir <目录>
```

新 runid：E1 为 29126--29129；E3 为 29326--29329。

## 四卡 smoke

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-b1-arp-smoke-r001 \
  /home/user/.conda/envs/zjy/bin/torchrun --standalone --nproc_per_node=4 \
  lib/train/run_training.py --script entertrack --config b1_abc_arp_4gpu_smoke \
  --save_dir output/diagnostics/b1_abc_arp/smoke_20260819_r001 \
  --seed 42 --use_wandb 0
```

## 四卡正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
  MPLCONFIGDIR=/tmp/mpl-b1-arp-formal-r001 PYTHONUNBUFFERED=1 \
  /home/user/.conda/envs/zjy/bin/torchrun --standalone --nproc_per_node=4 \
  lib/train/run_training.py --script entertrack --config b1_abc_arp_4gpu \
  --save_dir output/diagnostics/b1_abc_arp/formal_20260819_seed42_4gpu_r001 \
  --seed 42 --use_wandb 0
```

训练已正常结束，因此没有仍在运行的 PID。日志：

`output/diagnostics/b1_abc_arp/formal_20260819_seed42_4gpu_r001/logs/entertrack-b1_abc_arp_4gpu.log`

checkpoint：

`output/diagnostics/b1_abc_arp/formal_20260819_seed42_4gpu_r001/checkpoints/train/entertrack/b1_abc_arp_4gpu/`

## B1 inner-val sweep

```bash
/bin/bash scripts/run_b1_abc_arp_inner_val.sh
```

固定评测 ep15/20/25，对应 runid 29415/29420/29425；仅使用 `threemdot_val`。

## 推理路径 smoke 与结果导出

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python scripts/smoke_b1_arp_inference.py
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python scripts/export_b1_abc_arp_controlled.py
```

## 测试

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m unittest tests.test_pcum
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m py_compile \
  lib/models/layers/attn_blocks_arp.py lib/models/entertrack/vit_arp.py \
  lib/train/actors/entertrack_threemdot.py lib/train/trainers/ltr_trainer.py \
  scripts/export_b1_abc_arp_controlled.py scripts/smoke_b1_arp_inference.py
```

本轮没有执行任何 `threemdot_test`、official test 或 outer holdout 命令。
