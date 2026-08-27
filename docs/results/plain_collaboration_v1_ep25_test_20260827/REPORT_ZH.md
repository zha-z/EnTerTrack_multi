# Plain Collaboration V1 四卡训练与 Three-MDOT Test 报告

## 结论

`B0-ABC-Plain-Collaboration-V1` 的远端协作路径在训练和测试中均真实启用，但当前等权最终 search-feature 融合没有超过与它直接配对的 `B0-ABC-Plain-4GPU ep25`。

按仓库统一 OSTrack 口径，V1 overall AUC 为 **48.0028**，配对 B0（run `26025`）为 **48.9464**，差值为 **-0.9435 个百分点**。Precision、Normalized Precision 和 Mean IoU 也分别下降 0.9920、0.8524 和 1.0894 个百分点。因此 E1 当前应判为负结果，不能声称 collaboration 已提升整体 tracking AUC。

这个结论不能与旧的 `64.89` 直接比较：`64.89` 来自不同的 inner-val/历史口径；本报告只比较同一 Three-MDOT test、同一 OSTrack evaluator 下的冻结预测。仓库其他受控 B0 数字也可能来自不同 sampler/run，不能替代本次从同一初始化 checkpoint 得到的 E0 配对参考。

## 同口径 Test 指标

| 视角 | V1 AUC | B0 AUC | AUC 差值 | V1 Precision | V1 Norm Precision | V1 Mean IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall（105 sequences） | 48.0028 | 48.9464 | **-0.9435** | 63.6739 | 76.7527 | 55.3573 |
| A（35 sequences） | 43.9562 | 46.7075 | **-2.7513** | 60.0543 | 73.8407 | 49.5420 |
| B（35 sequences） | 47.6228 | 48.8976 | **-1.2748** | 62.9668 | 75.2790 | 54.2424 |
| C（35 sequences） | 52.4295 | 51.2340 | **+1.1955** | 68.0005 | 81.1383 | 62.4913 |

Overall 的完整配对差值如下：

| 指标 | V1 | B0 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| AUC | 48.0028 | 48.9464 | -0.9435 |
| Precision | 63.6739 | 64.6659 | -0.9920 |
| Normalized Precision | 76.7527 | 77.6050 | -0.8524 |
| Mean IoU | 55.3573 | 56.4467 | -1.0894 |

- V1 absolute sequence-level bootstrap AUC 95% CI：42.4376–53.8125。
- 105 个配对序列中，24 个上升、77 个下降、4 个完全持平。
- 按 `target_id` 聚合后，7/35 target 上升，28/35 target 下降。
- 对 35 个 target 做 100,000 次 grouped bootstrap，AUC 差值 95% CI 为 -2.1592–+0.2974 个百分点。区间包含 0，因此不能把当前下降写成已证明的普遍统计效应；但“E1 必须优于 E0”的实验成功条件没有通过。
- 最大负差值集中在 `md3002-1`（-40.7338）、`md3009-1`（-39.9619）和 `md3061-2`（-18.2911）；最大正差值为 `md3041-3`（+39.9730）。强烈的视角/序列异质性说明等权 remote 注入存在明显 negative transfer 风险。

评测严格复用 `lib.test.analysis.extract_results.calc_seq_err_robust`：success thresholds 为 0.00–1.00、步长 0.05，比较符为严格 `IoU > threshold`，读取 `target_visible`，首帧按 OSTrack 规则处理，最后对 105 个视角序列宏平均。

## 训练结果

V1 从 `B0-ABC-Plain-4GPU ep25` 初始化，冻结 local backbone 和 CENTER head，只训练 `plain_collaboration.*`。epoch 25 checkpoint 有 179 个 network tensors，其中 9 个 key 属于 collaboration adapter；ATP、PCUM 和 C3R key 均为 0。optimizer 只有一个参数组，LR 全程为 `8e-5`。

| Epoch | Train loss | Val loss | Train IoU | Val IoU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.71967 | 0.93552 | 0.83405 | 0.78970 |
| 18（最低 val loss） | 0.70827 | **0.90419** | 0.83783 | 0.79360 |
| 25 | 0.70374 | 0.92846 | 0.83730 | 0.78671 |

从 epoch 1 到 25，train loss 仅下降 2.21%，val loss 下降 0.75%；val loss 在 epoch 18 最低，epoch 25 比最低点回升 2.68%。由于 validation 保留随机 TrackingSampler、crop jitter 和随机增强，这条曲线只能作为训练诊断，不能单凭单个 epoch 的波动断言严格过拟合。本实验按预定 epoch 25 报告，没有用 test 指标选择 checkpoint。

每个 rank、每个 epoch 的记录均为 train A/B/C 各 1500，val A/B/C 各 374；四卡对应全局 train 各 6000、val 各 1496。25 个 epoch 共保存 50 条 multiview epoch summary，所有 `target_id × view_id` 计数在同一 target 内严格相等。25 份 validation sampling manifest 共 28,050 个 local-view rows，其文件行数、大小和 SHA256 已归档。

## 推理完整性与协作诊断

- 正式结果完整：bbox、time、max_score、APCE、collaboration diagnostics 均为 105/105。
- 总评测帧数 83,349；每个 bbox 文件与对应诊断 CSV 行数一致。
- 105 个初始化帧走 local-only；其余 83,244 帧全部 `used_remote=true`。
- A/B/C 各有 27,748 个 active remote 帧；每帧都有两个合法 sender。
- 两个 sender 固定等权 `0.5/0.5`；不存在 reliability weighting。
- 送入融合的 search token 始终为完整 256 个，没有 pruning/recovery。
- 测试期 learned residual scale 为 0.051145；relative residual norm 平均 0.11154，范围 0.06124–0.16279，未触及 0.25 cap。
- `.run_identity.json` 记录 `no_gt_inference=true`，PCUM/C3R/COOP 均未启用；协作只使用同 target、同 frame 的其他两个视角特征。
- 平均记录时间 0.008507 秒/帧，对应 117.54 FPS。该数值来自 worker 的逐帧 `_time.txt`，不是四卡作业的端到端 wall-clock throughput；B0 的记录 FPS 也只能作粗略参考。

这些证据说明结果下降不是因为 remote 路径未运行，也不是因为只测试了 A 视角。更合理的解释是：当前无可靠性判断的等权 feature residual 对 C 有平均帮助，但对 A/B 的污染更大，最终抵消并超过收益。

## 研究决策

当前 V1 可作为一个有价值的负对照：它证明“高带宽 remote feature 确实进入了闭环”，但没有证明“无条件 remote feature fusion 有效”。不建议继续增加 V1 的 residual 强度或依据 test 序列调权。

下一步如果执行 E2，应只在 inner-val/inner-dev 上设计 prediction-only reliability weighting，并保持 backbone、CENTER head、sampler、loss 和冻结的 E0/E1 checkpoint 不变。只有 inner-val 的预注册门槛通过后，才应申请新的正式 test；本次 test 结果不能用于逐 target 或逐视角手工设权。

## 代码与 checkpoint

- 训练主体实现：commit `4990ac8`；frame metadata 修复：`2381002`。
- 多视角闭环推理实现：commit `c6eaaa5`。
- 配置：`experiments/entertrack/plain_collaboration_v1.yaml`。
- 正式 checkpoint：epoch 25，23,706,506 bytes。
- Checkpoint SHA256：`0607e8dcd05d9732a0058bb7a3d9e356474dbeb219a65e0472ac83a83d58ad40`。
- checkpoint 本体和 11 MB 逐帧原始结果不提交 Git；仓库保存逐文件 SHA256 manifest，可核验本地冻结产物。
- checkpoint 没有嵌入精确 git HEAD，因此 provenance 只记录已确认的代码 lineage，不伪造无法从产物证明的 commit 身份。

## 可复现命令

四卡训练命令：

```bash
cd /data/zjy/EnTeR-Track-main

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=. \
MPLCONFIGDIR=/tmp/mpl-plain-collaboration-v1 \
/home/user/.conda/envs/zjy/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  lib/train/run_training.py \
  --script entertrack \
  --config plain_collaboration_v1 \
  --save_dir output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001 \
  --seed 42 \
  --use_wandb 0
```

四卡正式推理的可复现命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=. \
MPLCONFIGDIR=/tmp/mpl-plain-collaboration-v1-test \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param plain_collaboration_v1 \
  --dataset_name threemdot_test \
  --runid 27125 \
  --threads 8 \
  --num_gpus 4 \
  --checkpoint output/diagnostics/plain_collaboration_v1/e1_run_20260827_seed42_4gpu_r001/checkpoints/train/entertrack/plain_collaboration_v1/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 \
  --no_gt_inference 1
```

配对 OSTrack 离线重算命令：

```bash
PYTHONPATH=. \
MPLCONFIGDIR=/tmp/mpl-plain-collaboration-v1-analysis \
/home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py \
  --tracker_name entertrack \
  --tracker_param plain_collaboration_v1 \
  --dataset_name threemdot_test \
  --runid 27125 \
  --compare_param b0_abc_plain_4gpu_26025 \
  --output_dir output/analysis/plain_collaboration_v1_ep25_test_vs_b0
```

本地没有持久化正式 test 的 stdout 日志，因此上面的 test 命令是与 `.run_identity.json` 和结果目录一致的可复现命令，不冒充无法审计的原始 shell transcript。

## 归档文件

- `training_epoch_metrics.csv`：25 个 epoch 的 train/val total、GIoU、L1、Focal、IoU 和 LR。
- `multiview_epoch_counts.jsonl`：25 条 train 和 25 条 val 的 A/B/C、`target_id × view_id` 计数。
- `test_summary_vs_b0.json`：OSTrack overall/per-view/per-target 指标和 105 序列配对摘要。
- `sequence_metrics.csv`：V1 的 105 个独立序列指标。
- `paired_sequence_metrics_vs_b0.csv`：V1 与 B0 的逐序列四项指标和差值。
- `diagnostics_summary.json`：83,349 帧 collaboration 完整性、权重和 residual 统计。
- `tracking_results_manifest.csv`：526 个本地正式产物的大小、行数和 SHA256。
- `validation_sampling_manifest_inventory.csv`：25 份随机 val sampling manifest 的行数、大小和 SHA256。
- `provenance.json`：checkpoint、训练日志、run identity、代码 lineage 和因果约束。
