# D2-P2 复现命令

以下命令从仓库根目录执行。writer 均拒绝覆盖已存在 artifact；完整重放应在独立 worktree/空结果目录进行，不要删除本目录的冻结结果。

冻结 executable commit：`53e20608ddc2a44a848a42b4c92a38b9e949314a`。Train selection 生成的 `selected_source.json` SHA256 为 `ca8587cd4d0eb8d7293c2e4c762da6f6c76a9612b43ccadf23411626ab912e60`，candidate 为 P50。

```bash
export D2P2_OUT=docs/results/target_prompt_d2_p2_partial_source_calibration_20260901
```

## 1. 静态检查与单元测试

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m py_compile \
  tracking/target_prompt_d2_p2_partial_degradation.py \
  tracking/run_target_prompt_d2_p2_representation.py \
  tracking/analyze_target_prompt_d2_p2_calibration.py

env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_target_prompt_d2_p2_calibration \
  tests.test_target_prompt_d2_p1_representation \
  tests.test_target_prompt_e3_u1
```

## 2. Train-only representation

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/run_target_prompt_d2_p2_representation.py \
  --phase train --output-dir "$D2P2_OUT" --gpu 0 --batch-size 64
```

## 3. Train-only selection 与 candidate freeze

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_target_prompt_d2_p2_calibration.py \
  --phase train-select --output-dir "$D2P2_OUT"
```

若 `selected_source.json.train_acceptance.pass=false`，立即 STOP，不执行后续命令。

## 4. 唯一 frozen candidate 的 VAL representation

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/run_target_prompt_d2_p2_representation.py \
  --phase val --output-dir "$D2P2_OUT" --gpu 0 --batch-size 64
```

VAL runner 只读取 `selected_source.json` 中 frozen candidate 与 P100；不会生成其余 partial candidate。

## 5. VAL holdout 与 robustness

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_target_prompt_d2_p2_calibration.py \
  --phase val-holdout --output-dir "$D2P2_OUT"
```

本任务不包含 training、tracking rollout、test、official test 或任何参数 sweep。

## 6. Identity 运行说明

首次 Train attempt 在正式 artifact 写入前被 P100 identity gate 拦截：末批不足 64 引起 CUDA batch-shape 数值漂移。没有降低 tolerance；commit `53e2060` 将末批 duplicate-pad 到 64，丢弃 padding output 后重跑，Train 2,341 与 VAL 599 个 P100 均达到 metric difference 0、FP16 prompt exact。正式 artifact 只来自通过 identity 的第二次运行。
