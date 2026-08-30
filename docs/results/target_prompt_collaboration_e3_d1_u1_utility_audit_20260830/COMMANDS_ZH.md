# E3-D1-U1 实际复现命令

所有仓库命令均从 `/data/zjy/EnTeR-Track-main` 执行。

## 1. 起始检查与 checkpoint 冻结

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short

sha256sum \
  output/diagnostics/target_prompt_collaboration_e3_d1/run_20260830_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3_d1/EnTeRTrack_ep0025.pth.tar
```

起始 branch 为 `feature/pcum-cross-layer-arp`，source HEAD 为 `8d3fa0db6c8ad0fe83f447033df497fd0303cd25`；checkpoint 预期且实测 SHA256 均为 `231e20df89c184dbe5411fa687b625b809d4c1580df64059ec9d50f3c62a1c2b`。

## 2. 语法与定向单元测试

```bash
git diff --check

/home/user/.conda/envs/zjy/bin/python -m py_compile \
  tracking/run_target_prompt_e3_u1_counterfactual.py \
  tracking/analyze_target_prompt_e3_u1_utility.py \
  tests/test_target_prompt_e3_u1.py

PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m unittest -v \
  tests.test_target_prompt_e3_u1 \
  tests.test_target_prompt_collaboration \
  tests.test_plain_collaboration \
  tests.test_target_prompt_asymmetric_degradation
```

结果：58 项 PASS，1 项真实模型 CUDA smoke 因测试进程不可见 CUDA 按装饰器跳过；随后用宿主 GPU 完成下述显式两帧 smoke。

## 3. 1 target / 2 frames smoke

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-d1-u1-smoke \
/home/user/.conda/envs/zjy/bin/python \
  tracking/run_target_prompt_e3_u1_counterfactual.py \
  --tracker-param target_prompt_collaboration_e3_d1 \
  --dataset threemdot_val \
  --checkpoint output/diagnostics/target_prompt_collaboration_e3_d1/run_20260830_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3_d1/EnTeRTrack_ep0025.pth.tar \
  --expected-sha256 231e20df89c184dbe5411fa687b625b809d4c1580df64059ec9d50f3c62a1c2b \
  --output-dir /tmp/e3-d1-u1-smoke-4UYniLnU/artifacts \
  --runid e3_d1_u1_smoke --gpu 0 --max-targets 1 --max-frames 2
```

结果：24 branch rows、12 feature rows；6 个 Local rows 与冻结 E3-U1 精确一致。`active_receiver_frames=3`，五个失败计数全部为 0。首次在受限沙箱内执行时 CUDA 不可见，失败发生在 `torch.cuda.set_device()`、任何模型推理之前；以上是随后实际成功的宿主 GPU 命令。

## 4. 唯一一次完整 prediction-only val rollout

正式结果目录已先写入协议，因此完整预测先原子写到独立 staging：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-d1-u1-full \
/home/user/.conda/envs/zjy/bin/python \
  tracking/run_target_prompt_e3_u1_counterfactual.py \
  --tracker-param target_prompt_collaboration_e3_d1 \
  --dataset threemdot_val \
  --checkpoint output/diagnostics/target_prompt_collaboration_e3_d1/run_20260830_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3_d1/EnTeRTrack_ep0025.pth.tar \
  --expected-sha256 231e20df89c184dbe5411fa687b625b809d4c1580df64059ec9d50f3c62a1c2b \
  --output-dir docs/results/.target_prompt_collaboration_e3_d1_u1_prediction_staging_20260830 \
  --runid e3_d1_u1_val_20260830 --gpu 0
```

此阶段不加载 GT。结果为 5 targets / 15 sequences、56,448 branch rows、28,224 feature rows；runtime audit 五个失败计数全部为 0。

## 5. Prediction freeze 独立复验

```bash
sha256sum \
  docs/results/.target_prompt_collaboration_e3_d1_u1_prediction_staging_20260830/prediction_only_d1_sender_counterfactual.csv \
  docs/results/.target_prompt_collaboration_e3_d1_u1_prediction_staging_20260830/prediction_only_d1_prompt_features.csv

wc -l \
  docs/results/.target_prompt_collaboration_e3_d1_u1_prediction_staging_20260830/prediction_only_d1_sender_counterfactual.csv \
  docs/results/.target_prompt_collaboration_e3_d1_u1_prediction_staging_20260830/prediction_only_d1_prompt_features.csv
```

独立 Python 审计同时验证：manifest bytes/hash/rows、禁止字段、`uses_gt=False`、四分支完整、runtime audit 全零，以及 14,112 个 Local bbox 对冻结 E3-U1 的全量逐字符串精确一致。得到：

```text
branch SHA256  72a4b3345fbb1f6807bac17bbeaf090e51f09cbe0bc20319492b1c28d04f9942
feature SHA256 c5702a26afd47f3ba03431ddfdf933b916eb9de6017a524e9fe95abc24d38087
```

只在复验 PASS 后，将三个冻结文件无改写搬入正式目录，并再次运行同一 `sha256sum`：

```bash
mv docs/results/.target_prompt_collaboration_e3_d1_u1_prediction_staging_20260830/prediction_only_d1_sender_counterfactual.csv \
  docs/results/target_prompt_collaboration_e3_d1_u1_utility_audit_20260830/
mv docs/results/.target_prompt_collaboration_e3_d1_u1_prediction_staging_20260830/prediction_only_d1_prompt_features.csv \
  docs/results/target_prompt_collaboration_e3_d1_u1_utility_audit_20260830/
mv docs/results/.target_prompt_collaboration_e3_d1_u1_prediction_staging_20260830/prediction_manifest.json \
  docs/results/target_prompt_collaboration_e3_d1_u1_utility_audit_20260830/
```

## 6. Prediction freeze 后的 post-hoc GT join

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-d1-u1-analysis \
/home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_target_prompt_e3_u1_utility.py \
  --profile d1_descriptive \
  --dataset threemdot_val \
  --output-dir docs/results/target_prompt_collaboration_e3_d1_u1_utility_audit_20260830 \
  --e3-reference-dir docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829
```

分析器先重新验证 D1 与 E3 prediction manifests/hashes，随后才读取 `threemdot_val` GT。该 profile 不运行 Logistic Regression、selector、gate 或任何 runtime policy。

## 7. 最终审计

```bash
git diff --check
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m unittest -v \
  tests.test_target_prompt_e3_u1 \
  tests.test_target_prompt_collaboration \
  tests.test_plain_collaboration \
  tests.test_target_prompt_asymmetric_degradation
```

本轮没有训练、完整模型/checkpoint backward 或 optimizer step 命令，没有 `threemdot_test`/official test 命令，没有 D2、sampler、model、checkpoint 或超参数修改命令。上述既有回归套件中 `test_backward_is_finite` 与 `test_backward_reaches_adapter_only` 会在隔离的 synthetic tensor 上调用 backward；它们不加载本轮 checkpoint、不更新参数、不写训练状态，也不属于本轮实验训练。
