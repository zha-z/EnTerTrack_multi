# E2B 可复现命令

从仓库根目录执行：

```bash
cd /data/zjy/EnTeR-Track-main
```

## 1. 预注册与静态测试

预注册先单独提交：

```text
f72d24716e9344c2bd9a3ba1a8c82aaaa96100a2
```

```bash
/home/user/.conda/envs/zjy/bin/python -m py_compile \
  lib/config/entertrack/config.py \
  lib/test/tracker/entertrack.py \
  lib/test/evaluation/tracker.py \
  lib/test/evaluation/running.py \
  tracking/analyze_plain_collaboration_target_consistency.py

PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_plain_collaboration tests.test_pcum
```

## 2. 真实 checkpoint smoke

使用 runid `28420`，只取最短 target `md3048` 的 A/B/C 三序列，以 sequential mode 运行同一 `threemdot_val` tracker。smoke 检查 1,590 个 prototype row、6,360 个 branch row；bbox/state identity、shape、finite、no-GT 全部 PASS。smoke 输出位于：

```text
output/test/tracking_results/entertrack/plain_collaboration_v1_e2b_target_consistency_28420
```

## 3. 完整 prediction-only rollout

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
MPLCONFIGDIR=/tmp/mpl-plain-e2b-28421 \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param plain_collaboration_v1_e2b_target_consistency \
  --dataset_name threemdot_val \
  --runid 28421 \
  --threads 5 \
  --num_gpus 1 \
  --checkpoint output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001/checkpoints/train/entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 \
  --no_gt_inference 1
```

checkpoint SHA256：

```text
0607e8dcd05d9732a0058bb7a3d9e356474dbeb219a65e0472ac83a83d58ad40
```

结果目录：

```text
output/test/tracking_results/entertrack/plain_collaboration_v1_e2b_target_consistency_28421
```

## 4. prediction freeze（不加载 GT）

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_plain_collaboration_target_consistency.py freeze \
  --results-dir output/test/tracking_results/entertrack/plain_collaboration_v1_e2b_target_consistency_28421 \
  --e15-predictions docs/results/plain_collaboration_sender_counterfactual_20260827/prediction_only_sender_counterfactual.csv \
  --output-dir docs/results/plain_collaboration_target_consistency_e2b_20260827
```

## 5. 冻结后 post-hoc LOTO

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-plain-e2b-analysis \
/home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_plain_collaboration_target_consistency.py analyze \
  --output-dir docs/results/plain_collaboration_target_consistency_e2b_20260827 \
  --posthoc-labels docs/results/plain_collaboration_sender_counterfactual_20260827/posthoc_sender_labels.csv \
  --dataset threemdot_val \
  --e2a-summary docs/results/plain_collaboration_temporal_reliability_e2a_20260827/oof_tracking_summary.csv
```

没有训练命令，也没有运行 `threemdot_test`/official test。
