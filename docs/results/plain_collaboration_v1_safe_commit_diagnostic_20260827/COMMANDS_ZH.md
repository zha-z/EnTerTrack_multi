# 可复现命令

工作目录均为：

```bash
cd /data/zjy/EnTeR-Track-main
```

Checkpoint：

```text
output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001/checkpoints/train/entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar
SHA256 0607e8dcd05d9732a0058bb7a3d9e356474dbeb219a65e0472ac83a83d58ad40
```

## 三路 inner-val 推理

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-plain-d0-local \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param plain_collaboration_v1_d0_local \
  --dataset_name threemdot_val --runid 28210 \
  --threads 5 --num_gpus 1 \
  --checkpoint output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001/checkpoints/train/entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 --no_gt_inference 1

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-plain-d0-closed \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param plain_collaboration_v1_d0_closed_loop \
  --dataset_name threemdot_val --runid 28211 \
  --threads 5 --num_gpus 1 \
  --checkpoint output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001/checkpoints/train/entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 --no_gt_inference 1

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-plain-d0-safe \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param plain_collaboration_v1_d0_safe \
  --dataset_name threemdot_val --runid 28212 \
  --threads 5 --num_gpus 1 \
  --checkpoint output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001/checkpoints/train/entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 --no_gt_inference 1
```

三条命令实际并行运行于 GPU 0/1/2，退出码均为 0。没有启动 official test。

## OSTrack 标准评测

三路分别执行：

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py \
  --tracker_name entertrack \
  --tracker_param plain_collaboration_v1_d0_safe \
  --dataset_name threemdot_val --runid 28212 \
  --output_dir output/analysis/plain_collaboration_v1_safe_commit_diagnostic_20260827/d0_safe
```

将 `tracker_param/runid/output_dir` 替换为 `d0_local/28210` 和 `d0_closed_loop/28211` 即可复现另外两路。

## 预测冻结与后验 GT join

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_plain_collaboration_safe_commit.py freeze \
  --results-dir output/test/tracking_results/entertrack/plain_collaboration_v1_d0_safe_28212 \
  --output-dir docs/results/plain_collaboration_v1_safe_commit_diagnostic_20260827

# 上一步产生并记录 prediction_only_features.csv 的 SHA256 后才执行：
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_plain_collaboration_safe_commit.py join \
  --output-dir docs/results/plain_collaboration_v1_safe_commit_diagnostic_20260827 \
  --dataset threemdot_val
```

脚本会拒绝 `threemdot_test`/包含 `test` 的 dataset，并在 join 前重新验证冻结 prediction SHA256。
