# E3 后续命令（本次未执行长训练/完整验证）

以下均从仓库根目录 `/data/zjy/EnTeR-Track-main` 执行。

## 四卡正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/user/.conda/envs/zjy/bin/torchrun \
  --standalone --nproc_per_node=4 \
  lib/train/run_training.py \
  --script entertrack \
  --config target_prompt_collaboration_e3 \
  --save_dir output/diagnostics/target_prompt_collaboration_e3/run_20260828_seed42_4gpu_r001 \
  --seed 42 \
  --use_wandb 0
```

解析后的冻结训练量：per-GPU batch=2、global batch=8、epochs=25、train samples/epoch=6000、val samples/epoch=1500、adapter LR=8e-5、weight decay=1e-4。只有 E3 adapter 的 148,993 parameters 更新。

## 训练后 threemdot_val 评测

将 `<E3_CHECKPOINT>` 替换为训练产物的绝对或仓库相对路径：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/test.py \
  --tracker_name entertrack \
  --tracker_param target_prompt_collaboration_e3 \
  --dataset_name threemdot_val \
  --runid e3_target_prompt_val_seed42_ep25 \
  --threads 0 \
  --num_gpus 1 \
  --checkpoint <E3_CHECKPOINT> \
  --no_gt_inference 1 \
  --fail_if_results_exist 1
```

本文件不提供或授权 `threemdot_test`/outer holdout 命令。未来比较固定为同口径 E0 Local、E1 V1 full-256 Safe、E3 K8 Safe。
