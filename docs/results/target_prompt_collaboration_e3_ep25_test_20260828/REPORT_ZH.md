# E3 Target Semantic Prompt ep25 正式测试与分析

## 结论

`B0-ABC-Plain-Target-Prompt-Collaboration-E3` 完成 25 epoch 四卡训练和一次冻结 checkpoint 的 Three-MDOT test。按与 E0/E1 相同的 OSTrack `calc_seq_err_robust`、105 个 A/B/C 独立序列宏平均口径：

- E0/B0 Plain：AUC **48.9464**；
- E1/V1 full-search collaboration：AUC **48.0028**；
- E3/K8 target prompt collaboration：AUC **48.7651**。

E3 相对 B0 为 **-0.1813 AUC point**，相对 E1 为 **+0.7622 AUC point**。因此 K=8 semantic prompt 明显修复了 V1 高带宽融合的大部分损失，但仍未超过 local baseline，预注册主成功门槛 `B0 +0.30 AUC point` 未通过。E3 当前应判为受控负结果，不应声称 collaboration 已提升 tracking AUC。

## 同口径指标

| 视角 | 方法 | AUC | Precision | Norm Precision | Mean IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| Overall（105 sequences） | B0 | 48.9464 | 64.6659 | 77.6050 | 56.4467 |
|  | E1 | 48.0028 | 63.6739 | 76.7527 | 55.3573 |
|  | **E3** | **48.7651** | **64.6272** | **77.4222** | **56.2229** |
| A（35） | B0 / E3 | 46.7075 / 46.5441 | 63.3401 / 63.3017 | 76.7755 / 76.6172 | 53.3910 / 53.1859 |
| B（35） | B0 / E3 | 48.8976 / 48.7740 | 64.7136 / 64.6972 | 76.9510 / 76.6249 | 55.6166 / 55.4679 |
| C（35） | B0 / E3 | 51.2340 / 50.9771 | 65.9438 / 65.8828 | 79.0886 / 79.0245 | 60.4469 / 60.1263 |

E3 相对 B0 的 Overall Precision、Norm Precision、Mean IoU 分别为 -0.0386、-0.1828、-0.2239 个百分点。三个视角的 AUC 均小幅下降：A -0.1634、B -0.1236、C -0.2569 个百分点，说明结果不是某一个 view 的孤立异常。

105 个配对序列中 30 个上升、70 个下降、5 个严格持平；按 target 聚合后 9/35 上升、26/35 下降。target-grouped 100,000 次 bootstrap 的 E3-B0 AUC 差值 95% CI 为 **[-0.2945, -0.0748] AUC point**。最差 target 为 `md3015`（-1.1278 point），没有 target 下降达到预注册的 5.00 point 红线。

E3-E1 的 target-grouped 95% CI 为 [-0.4785, +1.9980] point，区间包含 0。E3 的总体点估计明显优于 E1，但不能写成已证明的普遍统计优势。

## 原生 Three-MDOT Fused 指标说明

仓库原生 `three_fuse_extract_results_APCE` 还会用每帧 max-score/APCE 在 A/B/C 的输出中后处理选择一个 view，生成 `Fused` 曲线。原生 `eval_data.pkl` 的结果为：

| 方法 | Fused AUC | OP50 | OP75 | Precision | Norm Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 59.5908 | 73.4285 | 56.2971 | 75.7949 | 86.5107 |
| E1 | 59.3498 | 73.6887 | 55.0091 | 76.0617 | 86.1140 |
| E3 | 59.2296 | 73.3747 | 55.2452 | 75.6613 | 86.1330 |

该 `Fused` 值是 APCE/max-score 的 prediction-level view selector 指标，不是 E3 feature collaboration 本身的 105-view macro AUC，也不是 checkpoint 选择依据。本报告保留它作为仓库原生 evaluator 的 secondary 指标；预注册成败仍按 E0/E1/E3 的 A/B/C 独立序列同口径比较判断。原生曲线缓存 `eval_data.pkl` SHA256 为 `32a766d2249ad0ac89c134e24e704717a2f51d6404c621f739c2a9a5cf222341`，其可读数值和完整 success curves 已导出为 CSV。

## 训练诊断

固定使用 ep25，与 E0/E1 的 ep25 对照一致；没有在 test 上试多个 checkpoint。

| Epoch | Train loss | Val loss | Train IoU | Val IoU | Train relative residual norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.71859 | 0.95477 | 0.83509 | 0.78441 | 0.00860 |
| 14（最低 val loss） | 0.71082 | **0.94149** | 0.83563 | 0.78724 | 0.06587 |
| 25 | **0.69980** | 0.94780 | **0.83859** | 0.78619 | **0.10971** |

epoch 1 到 25，train loss 下降 2.61%，val loss 下降 0.73%；ep25 val loss 比最低点回升 0.67%。validation 仍采用随机 TrackingSampler、随机 crop jitter/augmentation，因此单个 epoch 的小幅波动不能独立证明严格过拟合。25 份 validation sampling manifest 共 28,050 rows，均已归档行数、大小和 SHA256。

每个 rank、每个 epoch 的 train A/B/C 均为 1500/1500/1500，val 为 374/374/374；四卡全局对应每个 view 的 train 6000、val 1496。50 条 train/val multiview summary 中，同一 target 的 A/B/C 计数始终对齐。

checkpoint 内 179 个 network tensors，其中 E3 adapter 9 keys、V1 0、PCUM 0、C3R 0；optimizer 只有一个参数组。checkpoint epoch=25，大小 23,707,338 bytes，SHA256 为 `d1c92bfe710e99d68b5f4a0eb2c151222204ca289bb1392f567fe76718d70c78`。

## 推理完整性

- 正式 run id：`28325`，新目录原子创建，没有覆盖历史结果；
- bbox/time/max-score/APCE/E3 diagnostics 均为 105/105；
- bbox 共 83,349 frames；首帧后 E3 diagnostics 共 83,244 rows；
- 83,244/83,244 行 `used_remote=true`，每行两个 valid sender；
- A/B/C active remote rows 均为 27,748；
- `uses_gt=False`，sender prompt source 和 persistent state source 均为 local；
- Safe Commit collaboration-state mismatch=0，跨帧 persistent-state continuity mismatch=0；
- inference residual relative norm 均值 0.11413，范围 0.06861–0.16591，0 次触及 0.25 cap；
- residual scale 固定为 0.0608939；不是 residual=0 或 bypass 导致的假阴性；
- K=8，每 sender/event 的理论 payload 为 FP32 6,144 bytes、FP16 3,072 bytes，相对 V1 恰好减少 32x；
- worker 逐帧记录的平均速度约 71.56 FPS，不代表四卡作业的端到端 wall-clock throughput。

GT 只在推理完成并冻结 526 个文件后用于离线指标计算。冻结 manifest SHA256 为 `d464dff241ca0a69867557d39854d7a7eedd10cc4d799d6264a4ea69c2656410`；归档脚本会在任何 GT join 前重新验证每个文件的 bytes、lines 和 SHA256。

## 预注册门槛

| Gate | 结果 | 判定 |
| --- | --- | --- |
| E3 AUC ≥ B0 +0.30 point | -0.1813 point | **FAIL** |
| E3 AUC > E1 | +0.7622 point | PASS（点估计） |
| 无 target 下降 ≥5.00 point | 最差 -1.1278 | PASS |
| payload 相对 V1 减少 32x | 32x | PASS |
| identity / Safe Commit / no-GT / remote active | 全部完整 | PASS |

按预注册规则，tracking 主门槛失败即判 E3 失败；通信压缩和完整性通过不能替代 AUC 收益。

## 结果解释与下一步

E3 相比 V1 的价值是把少数灾难性 negative transfer 压回到小扰动范围：V1 的 overall 降幅为 -0.9435 point，而 E3 仅为 -0.1813 point，且没有 target 跌幅超过 1.13 point。K=8 target prompt 确实比传输完整 256-token search feature 更安全。

但 E3 的小幅负增益非常广泛：70/105 sequence、26/35 target 下降，A/B/C 三视角均下降；同时 remote 路径全程 active、相对残差约 11.4%、未触及 cap。因此最符合证据的解释不是实现 bypass，而是“无可靠性判断、无几何对应的等权 remote prompt”持续注入了弱但系统性的噪声。

不应基于本次 test 继续调 K、residual scale、温度或逐 view 权重。下一步应回到 inner-val/inner-dev，优先做 prediction-only 的 event/remote utility 诊断，确认哪些帧或 sender 真正有正增益；只有冻结规则在 inner-val 明确优于 B0 后，才申请下一次正式 test。

## 实际命令

正式推理：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=. \
MPLCONFIGDIR=/tmp/mpl-target-prompt-e3-test \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param target_prompt_collaboration_e3 \
  --dataset_name threemdot_test \
  --runid 28325 \
  --threads 8 \
  --num_gpus 4 \
  --checkpoint output/diagnostics/target_prompt_collaboration_e3/run_20260828_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 \
  --no_gt_inference 1
```

105-view OSTrack 配对分析：

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-target-prompt-e3-analysis \
/home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py \
  --tracker_name entertrack \
  --tracker_param target_prompt_collaboration_e3 \
  --dataset_name threemdot_test \
  --runid 28325 \
  --compare_param b0_abc_plain_4gpu_26025 \
  --output_dir output/analysis/target_prompt_collaboration_e3_ep25_test
```

可复现归档：

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  scripts/archive_target_prompt_e3_results.py \
  --e3-result-dir output/test/tracking_results/entertrack/target_prompt_collaboration_e3_28325 \
  --b0-result-dir output/test/tracking_results/entertrack/b0_abc_plain_4gpu_26025 \
  --e1-result-dir output/test/tracking_results/entertrack/plain_collaboration_v1_27125 \
  --training-log output/diagnostics/target_prompt_collaboration_e3/run_20260828_seed42_4gpu_r001/logs/entertrack-target_prompt_collaboration_e3.log \
  --checkpoint output/diagnostics/target_prompt_collaboration_e3/run_20260828_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3/EnTeRTrack_ep0025.pth.tar \
  --eval-data output/test/result_plots/target_prompt_e3_b0_v1_ep25_20260828/eval_data.pkl \
  --output-dir docs/results/target_prompt_collaboration_e3_ep25_test_20260828
```

## Git 归档范围

提交代码和可读、可审计的精简结果：训练曲线、50 条 multiview counts、105 序列配对表、35 target 表、per-view 指标、原生 OSTrack 曲线、诊断汇总、prediction manifest、validation manifest inventory 和 provenance。

23.7 MB checkpoint、2.4 MB 原始训练日志、约 11 MB raw predictions 和二进制 `eval_data.pkl` 不提交 Git；它们的绝对路径、大小和 SHA256 已记录，所有关键曲线已导出为 CSV，ChatGPT 可直接读取仓库中的文字和表格结果。
