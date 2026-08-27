# E2A 可复现命令

工作目录：

```bash
cd /data/zjy/EnTeR-Track-main
```

## 1. 验证预注册提交与输入

```bash
git show --stat 089b826

sha256sum \
  docs/results/plain_collaboration_sender_counterfactual_20260827/prediction_only_sender_counterfactual.csv
```

预期 E1.5 SHA：

```text
75ebb3cb292202346619e6f81cc46d050fa394a73e7c0d01ad9376545ba95f43
```

## 2. 静态与因果单元测试

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m py_compile \
  tracking/analyze_plain_collaboration_temporal_reliability.py \
  tests/test_plain_collaboration_temporal_reliability.py

PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e2a-tests \
/home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_plain_collaboration_temporal_reliability
```

## 3. Prediction-only temporal freeze

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_plain_collaboration_temporal_reliability.py freeze \
  --source-predictions \
    docs/results/plain_collaboration_sender_counterfactual_20260827/prediction_only_sender_counterfactual.csv \
  --source-manifest \
    docs/results/plain_collaboration_sender_counterfactual_20260827/prediction_manifest.json \
  --output-dir \
    docs/results/plain_collaboration_temporal_reliability_e2a_20260827
```

该阶段不加载 dataset/GT，不运行 tracker。预期 temporal SHA：

```text
717d41c95bd8276e254f2af0baabc8e9a17f2213fe65389e176b2be9993d9a24
```

## 4. SHA 验证后的 post-hoc LOTO/OOF

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e2a-analysis \
/home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_plain_collaboration_temporal_reliability.py analyze \
  --output-dir \
    docs/results/plain_collaboration_temporal_reliability_e2a_20260827 \
  --posthoc-labels \
    docs/results/plain_collaboration_sender_counterfactual_20260827/posthoc_sender_labels.csv \
  --dataset threemdot_val
```

脚本会拒绝任何 dataset 名包含 `test`，并在 label join 前重新计算 temporal SHA。

## 5. 回归测试

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e2a-final \
/home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_plain_collaboration_temporal_reliability \
  tests.test_plain_collaboration \
  tests.test_pcum

git diff --check
```

本轮没有 tracker inference/training 命令，因为 E1.5 冻结 artifact 已包含全部 E2A 主特征所需字段。
