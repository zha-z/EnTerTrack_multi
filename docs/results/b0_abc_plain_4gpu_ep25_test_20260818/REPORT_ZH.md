# B0-ABC-Plain 四卡训练与 Three-MDOT Test 报告

## 结论

本次使用 `B0-ABC-Plain-4GPU` 的 epoch 25 checkpoint，在 Three-MDOT test 上完成了 105/105 个独立视角序列的测试。Plain ViT-Tiny 不进行跨视角融合，因此下表的 Overall 是 105 个 A/B/C 视角序列的宏平均，不是融合轨迹指标。指标已于 2026-08-18 按仓库 OSTrack 标准评测路径重新计算，旧版 unified diagnostic AUC 54.3028 不再作为正式结果。

训练期没有出现“从 epoch 1 到 epoch 25，train loss 下降且 val loss 上升”的严格单调剪刀形：train loss 从 1.00480 降至 0.73315（-27.04%），val loss 从 0.95669 降至 0.90791（-5.10%）。但是 val loss 在 epoch 4 达到最低值 0.87986；从 epoch 4 到 25，train loss 又下降 17.96%，而 val loss 回升 3.19%。因此可描述为“早期最低点之后出现轻度 generalization gap”，不能描述成全程单调恶化。当前 validation 每个 epoch 有随机采样、center/scale jitter、5% 灰度和 50% 水平翻转，这会给 val loss 带来随机波动。

## Test 指标

| 视角 | AUC | Precision | Normalized Precision | Mean IoU |
| --- | ---: | ---: | ---: | ---: |
| Overall（105 sequences） | 48.9464 | 64.6659 | 77.6050 | 56.4467 |
| A（35 sequences） | 46.7075 | 63.3401 | 76.7755 | 53.3910 |
| B（35 sequences） | 48.8976 | 64.7136 | 76.9510 | 55.6166 |
| C（35 sequences） | 51.2340 | 65.9438 | 79.0886 | 60.4469 |

- Sequence-level bootstrap AUC 95% CI：43.3233–54.8360。
- OSTrack 口径：复用 `lib/test/analysis/extract_results.py::calc_seq_err_robust`；success thresholds 为 0.00–1.00、步长 0.05，比较符为严格 `IoU > threshold`；读取 `target_visible`；首帧强制为初始化 GT；`exclude_invalid_frames=false`；最终对 105 个序列做宏平均。
- Mean IoU 仅为附加诊断：先在每个序列的 visible-valid 帧上求均值，再对有 valid 帧的序列求 `nanmean`。它不参与 AUC 计算；`md3003-3` 没有 visible-valid 帧，因此该序列的 Mean IoU 为 NaN。
- 记录的平均推理时间为 0.004257 秒/帧，对应 234.93 FPS。该数值来自各 worker 的逐帧 `_time.txt`，不是四卡作业的端到端 wall-clock throughput。
- 测试使用 `no_gt_inference=1`，remote state source 为 `none`。
- 完整性检查：bbox 105/105、time 105/105、max_score 105/105、APCE 105/105。

### 同口径基线比较

| 模型 | OSTrack AUC | 相对 B0-ABC-Plain |
| --- | ---: | ---: |
| B0-ABC-Plain ep25 | 48.9464 | — |
| Current baseline ep25 | 48.1520 | B0 +0.7944 |
| Original ep21 | 47.2014 | B0 +1.7450 |

三者均由相同代码对已有 bbox 预测离线重算；没有重新推理，也没有使用 test 指标选择 checkpoint。

## 模型与 checkpoint

- 配置：`experiments/entertrack/b0_abc_plain_4gpu.yaml`
- Backbone：Plain ViT-Tiny，dim 192、depth 6、heads 3、patch 16。
- Search token：256；template token：64；所有 token 经过 6 个 Transformer blocks。
- ATP、ARP pruning、token compensation、PCUM、C3R、FCVC、remote state：全部关闭。
- Checkpoint epoch：25。
- Checkpoint SHA256：`363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3`。
- Checkpoint 大小：65,159,882 bytes；为避免 Git 仓库膨胀，checkpoint 本体不提交。
- Checkpoint 中 network tensor 数：170；ATP key 数：0；PCUM key 数：0。

## 训练数据与视角计数

- Train targets：22；val targets：5。
- 四卡配置每卡 group batch 2，全局 group batch 8；A/B/C flatten 后全局 local-view batch 24。
- rank 0 每个 train epoch 记录 A/B/C 各 1500；四个 rank 对应全局各 6000。
- rank 0 每个 val epoch实际记录 A/B/C 各 374；因 `drop_last=True`，四个 rank 对应全局各 1496。
- 25 个 epoch 中没有出现 A=100%、B=0%、C=0；所有 target 的 A/B/C 计数逐项相等。
- `multiview_epoch_counts.jsonl` 保存 25 条 train 和 25 条 val 的 `target_id × view_id` 计数。

## 实际测试命令

```bash
cd /data/zjy/EnTeR-Track-main

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=. \
PYTHONUNBUFFERED=1 \
MPLCONFIGDIR=/tmp/mpl-b0-abc-test \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param b0_abc_plain_4gpu \
  --dataset_name threemdot_test \
  --runid 26025 \
  --threads 8 \
  --num_gpus 4 \
  --checkpoint /data/zjy/EnTeR-Track-main/output/diagnostics/b0_abc_plain/run_20260818_seed42_4gpu_r001/checkpoints/train/entertrack/b0_abc_plain_4gpu/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 \
  --no_gt_inference 1
```

测试过程 exit code 为 0。随后执行 `tracking/verify_pcum_test_run.py`，得到：

```text
[VERIFIED] config=b0_abc_plain_4gpu dataset=threemdot_test runid=26025 bbox=105/105 source=none no_gt=true checkpoint=epoch=25,pcum_keys=0
```

## 文件说明

- `training_epoch_metrics.csv`：25 个 epoch 的 train/val total、GIoU、L1、Focal、IoU 和 LR。
- `multiview_epoch_counts.jsonl`：每个 epoch 的 A/B/C 与 `target_id × view_id` 计数。
- `test_summary.json`：OSTrack 口径的整体、逐视角和逐 target 指标，并内嵌 `evaluation_protocol`。
- `sequence_metrics.csv`：OSTrack 口径的 105 个序列视角独立指标。
- `test_run.log`：正式测试原始终端日志。
- `tracking_results_manifest.csv`：420 个本地原始结果文件的文件名、大小和 SHA256；逐帧 bbox 文件未提交。
- `provenance.json`：checkpoint、日志、runid、数据集和因果约束摘要。

指标由仓库统一分析入口 `tracking/analysis_results.py` 生成。该入口现在直接复用 OSTrack 的 `calc_seq_err_robust`，不再维护另一套 `IoU >= threshold` 且忽略 `target_visible` 的公式。
