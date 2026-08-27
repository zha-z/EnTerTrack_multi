# E1.5 可复现命令

所有命令从仓库根目录运行：

```bash
cd /data/zjy/EnTeR-Track-main
```

冻结 checkpoint：

```text
output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001/checkpoints/train/entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar
SHA256 0607e8dcd05d9732a0058bb7a3d9e356474dbeb219a65e0472ac83a83d58ad40
```

## 1. 静态与单元测试

```bash
/home/user/.conda/envs/zjy/bin/python -m py_compile \
  lib/config/entertrack/config.py \
  lib/test/tracker/entertrack.py \
  lib/test/evaluation/tracker.py \
  lib/test/evaluation/running.py \
  tracking/analyze_plain_collaboration_sender_counterfactual.py

PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_plain_collaboration
```

## 2. 完整 threemdot_val prediction-only rollout

实际运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-plain-e15-28315 \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param plain_collaboration_v1_e15_sender_counterfactual \
  --dataset_name threemdot_val \
  --runid 28315 \
  --threads 5 \
  --num_gpus 1 \
  --checkpoint output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001/checkpoints/train/entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 \
  --no_gt_inference 1
```

退出码为 0。结果目录：

```text
output/test/tracking_results/entertrack/plain_collaboration_v1_e15_sender_counterfactual_28315
```

命令只运行 `threemdot_val`；没有运行 `threemdot_test` 或其他 official test。

## 3. 先冻结 prediction-only artifact

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_plain_collaboration_sender_counterfactual.py freeze \
  --results-dir output/test/tracking_results/entertrack/plain_collaboration_v1_e15_sender_counterfactual_28315 \
  --output-dir docs/results/plain_collaboration_sender_counterfactual_20260827
```

freeze 阶段不加载 dataset/GT，并写入 `prediction_manifest.json`。冻结结果：56,448 rows，14,112 groups，四 branch/group，SHA256：

```text
75ebb3cb292202346619e6f81cc46d050fa394a73e7c0d01ad9376545ba95f43
```

## 4. 冻结后才执行 post-hoc GT join

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-plain-e15-analysis \
/home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_plain_collaboration_sender_counterfactual.py join \
  --output-dir docs/results/plain_collaboration_sender_counterfactual_20260827 \
  --dataset threemdot_val
```

join 会先重新计算并验证冻结 prediction SHA。脚本拒绝任何 dataset 名含 `test` 的调用。Oracle 和 helpful/harmful label 仅在该 post-hoc 阶段使用 GT。

## 5. 最终回归测试

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_plain_collaboration tests.test_pcum

git diff --check
```

本任务不包含训练命令，因为明确禁止新训练。
