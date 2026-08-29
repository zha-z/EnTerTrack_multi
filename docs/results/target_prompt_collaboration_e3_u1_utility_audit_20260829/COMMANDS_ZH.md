# E3-U1 实际复现命令

均从 `/data/zjy/EnTeR-Track-main` 执行。

## Checkpoint 校验

```bash
sha256sum output/diagnostics/target_prompt_collaboration_e3/run_20260828_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3/EnTeRTrack_ep0025.pth.tar
```

预期：`d1c92bfe710e99d68b5f4a0eb2c151222204ca289bb1392f567fe76718d70c78`。

## 单元与回归测试

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-u1-tests \
/home/user/.conda/envs/zjy/bin/python -m unittest -v \
  tests.test_target_prompt_e3_u1 \
  tests.test_target_prompt_collaboration \
  tests.test_plain_collaboration
```

结果：47/47 PASS。

## 两帧 smoke

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-u1-smoke \
/home/user/.conda/envs/zjy/bin/python \
  tracking/run_target_prompt_e3_u1_counterfactual.py \
  --dataset threemdot_val \
  --checkpoint output/diagnostics/target_prompt_collaboration_e3/run_20260828_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3/EnTeRTrack_ep0025.pth.tar \
  --expected-sha256 d1c92bfe710e99d68b5f4a0eb2c151222204ca289bb1392f567fe76718d70c78 \
  --output-dir /tmp/e3-u1-smoke-parent-OXfqAe/artifacts \
  --runid e3_u1_smoke --gpu 0 --max-targets 1 --max-frames 2
```

## 完整 prediction-only val rollout

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-u1-val \
/home/user/.conda/envs/zjy/bin/python \
  tracking/run_target_prompt_e3_u1_counterfactual.py \
  --dataset threemdot_val \
  --checkpoint output/diagnostics/target_prompt_collaboration_e3/run_20260828_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3/EnTeRTrack_ep0025.pth.tar \
  --expected-sha256 d1c92bfe710e99d68b5f4a0eb2c151222204ca289bb1392f567fe76718d70c78 \
  --output-dir docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829 \
  --runid e3_u1_val_20260829 --gpu 0
```

## Freeze 复验

```bash
sha256sum \
  docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829/prediction_only_e3_sender_counterfactual.csv \
  docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829/prediction_only_e3_prompt_features.csv

wc -l \
  docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829/prediction_only_e3_sender_counterfactual.csv \
  docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829/prediction_only_e3_prompt_features.csv
```

## Post-hoc GT join 与 LOTO

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-u1-analysis \
/home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_target_prompt_e3_u1_utility.py \
  --output-dir docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829 \
  --dataset threemdot_val
```

本轮没有 training 命令，也没有 `threemdot_test`/official test 命令。

