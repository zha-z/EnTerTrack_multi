# D2-P2 Partial Occlusion Source Calibration

## 冻结结论

本轮命中 aggregate **Case A**，同时存在一项预注册 **Case D robustness warning**。

- Train-only 从 P25/P50/P75 中冻结选择 **P50**；`D_source=0.6020`，7/7 primary feature 比 Clean 更接近 Natural。
- P50 明显优于历史 P100：Train `D_source 0.6020 vs 1.6472`，mean `|SMD(Candidate,Clean)| 1.0979 vs 2.2062`。
- 冻结 P50 在 VAL 达到 7/7 closer，`D_source=0.3806`，低于 P100 的 `0.8747`，因此 holdout PASS。
- severity curve 在 paired score/APCE/entropy、bbox displacement 与 prompt set similarity 上整体单调，支持 coverage 是主要强度轴。
- robustness 并不均匀：Train `md3020`（n=59）`D_source=2.0471`，触发冻结 warning；VAL view C 仅 2/7 closer、`D_source=1.5963`，虽未达到 extreme threshold，也必须保守报告。
- 下一步选择 **A. D2-S1 Source-only Collaboration Training**，但只能作为未来独立预注册任务；本轮没有训练，也没有授权直接启动 D2-S1。

## 范围、身份与泄漏门禁

| 项目 | 冻结事实 |
|---|---|
| branch / task-start HEAD | `feature/pcum-cross-layer-arp` / `8c978515803bdf3bfe69d0a83281688695acb3ea` |
| preregistered executable commit | `53e20608ddc2a44a848a42b4c92a38b9e949314a` |
| model | B0 ep25 Plain ViT-Tiny + CENTER；K=8 |
| checkpoint SHA256 | `363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3` |
| core identity | 170 keys，mismatch 0，strict full load PASS |
| adapter/collaboration | adapter forward 0；未使用 remote/cross-attention/fusion |
| D2-P1 reference | inventory/feature/prompt 三个 SHA 全量复验 PASS |
| P100 identity | Train 2,341 + VAL 599：metric max abs diff 0，FP16 prompt exact，mismatch 0 |
| split discipline | Train freeze 前未生成 VAL；VAL 只包含 Clean/Natural/P50/P100，无 P25/P75 |
| execution | `eval()` + `inference_mode()`；无 training/backward/optimizer/checkpoint update |
| test | 未访问 `threemdot_test`，未运行 official test 或 tracking rollout |
| GT | 仅复用 D2-P1 deterministic crop；未传入模型、未用于 candidate selection |

首次 P100 identity 尝试因末 batch 不足 64 导致少量 CUDA batch-shape 数值漂移，gate 在任何正式 artifact 写入前 STOP。没有放宽 `1e-6`/FP16 exact 门槛；后续以 duplicate-only padding 保持 D2-P1 的 batch shape，丢弃 padding output 后全量 identity 为 0 差异。该修复提交为 `53e2060`。

## Candidate 定义与样本

P25/P50/P75/P100 均是一个 edge-anchored contiguous block，orientation 只由 D2-P1 Clean sample ID 的 frozen SHA256 决定；同一样本各 severity 共用 orientation。fill 始终是 normalized `0.0`。实际平均 coverage 为 Train `25.80/50.80/75.81/100%`，VAL P50 `50.79%`。

| split | Clean | balanced Natural | generated candidates |
|---|---:|---:|---:|
| Train | 2,341 | 2,341 | P25/P50/P75/P100 各 2,341 |
| VAL | 599 | 599 | frozen P50/P100 各 599 |

Clean/Natural 没有重新抽样，均直接复用 D2-P1 reference。

## Q1：P25/P50/P75/P100 severity curve

严格 paired C→candidate 均值：

| split | candidate | Δ max score | Δ APCE | Δ entropy | bbox center displacement @256 | prompt symmetric best-match cosine |
|---|---|---:|---:|---:|---:|---:|
| Train | P25 | -0.1168 | -28.26 | +0.0659 | 4.80 px | 0.9372 |
| Train | P50 | -0.2341 | -71.27 | +0.1507 | 11.00 px | 0.8853 |
| Train | P75 | -0.3677 | -111.88 | +0.2876 | 29.91 px | 0.7823 |
| Train | P100 | -0.4824 | -134.36 | +0.3625 | 60.84 px | 0.7646 |
| VAL | P50 | -0.2357 | -93.75 | +0.1240 | 8.79 px | 0.8732 |
| VAL | P100 | -0.4525 | -152.27 | +0.2971 | 56.11 px | 0.7520 |

Train 的五项 paired mean 均随 coverage 呈合理单调趋势：score/APCE 持续下降、entropy/bbox displacement 持续上升、prompt similarity 持续下降。P75→P100 的 prompt cosine 降幅较小，但方向仍一致。

K8 prompt mean 也显示 P50 缓解 P100 collapse：

| split | group | top-k score mean | top1-top8 gap | prompt norm | pairwise cosine |
|---|---|---:|---:|---:|---:|
| Train | Clean | 0.1674 | 0.8190 | 12.5676 | 0.6652 |
| Train | P50 | 0.1993 | 0.5534 | 12.4814 | 0.6650 |
| Train | P100 | 0.1741 | 0.2707 | 12.3839 | 0.7717 |
| Train | Natural | 0.1925 | 0.5951 | 12.5152 | 0.6922 |
| VAL | Clean | 0.1772 | 0.7735 | 12.5966 | 0.6548 |
| VAL | P50 | 0.2269 | 0.4880 | 12.4852 | 0.6821 |
| VAL | P100 | 0.1895 | 0.2339 | 12.4000 | 0.7790 |
| VAL | Natural | 0.2209 | 0.4692 | 12.5031 | 0.7078 |

## Q2：Train-only 选择哪个 candidate？

选择 **P50**。正式 ranking 只使用 Train：

| candidate | D_source | closer / 7 | large shift / 7 | mean abs SMD vs Clean |
|---|---:|---:|---:|---:|
| P25 | 0.7057 | 6 | 2 | 0.4519 |
| **P50** | **0.6020** | **7** | **5** | **1.0979** |
| P75 | 1.0285 | 5 | 5 | 1.8058 |
| P100 reference | 1.6472 | 1 | 6 | 2.2062 |

P25 的 shift 最小，但正式目标不是“最弱扰动”，而是最小 Natural-normalized distance；P50 的 `D_source` 最低。因此没有根据 magnitude 或 VAL 改选 P25。

## Q3：P50 是否至少 5/7 比 Clean 更接近 Natural？

是，Train 为 **7/7**。七项 normalized distance 分别为：

| feature | R_j(P50) |
|---|---:|
| max_score | 0.5353 |
| APCE | 0.4809 |
| score_entropy | 0.3588 |
| prompt_topk_score_mean | 0.4081 |
| prompt_top1_top8_gap | 0.5118 |
| prompt_norm_mean | 0.9379 |
| prompt_pairwise_cos_mean | 0.9815 |

后两项仅略优于 Clean，但仍严格满足预注册 closer 条件；没有追加绝对“相似”阈值。

## Q4：相比 P100 是否显著减轻 representation shift？

是。Train large-shift count 从 `6/7` 降至 `5/7`，mean abs SMD 从 `2.2062` 降至 `1.0979`，只剩 P100 的 **49.76%**。VAL mean abs SMD 为 `1.7643 vs 2.8206`，是 P100 的 **62.55%**；VAL large-shift count 两者仍均为 6/7，因此“减轻”不等于“扰动很小”。

## Q5：冻结 P50 在 VAL 是否达到 >=4/7？

达到 **7/7**；VAL `D_source(P50)=0.3806 < D_source(P100)=0.8747`，两个 holdout gate 均 PASS。P50 的 score/APCE mean `0.5645/123.96` 几乎落在 Natural 的 `0.5642/122.74`，但这只是 pooled descriptive evidence，不是 tracking gain。

## Q6：Train→VAL 是否 generalize？

aggregate gate 上是：Train 7/7、VAL 7/7，且两边 P50 D_source 均低于 P100。

robustness 上不是完全均匀：

| split/view | D_source(P50) | D_source(P100) | closer / 7 |
|---|---:|---:|---:|
| Train A | 1.0803 | 2.9719 | 6 |
| Train B | 0.6126 | 1.8516 | 6 |
| Train C | 0.4630 | 0.8335 | 7 |
| VAL A | 0.3975 | 0.7039 | 7 |
| VAL B | 0.4309 | 0.7171 | 7 |
| VAL C | 1.5963 | 3.4639 | 2 |

Train `md3020`（n=59）`D_source=2.0471` 触发 frozen extreme warning。VAL `md3027` 的 `D_source=3.3762` 更高，但只有 20 rows，未满足预注册 n>=30 warning 条件；它仍是重要的描述性风险。禁止据此做 view/target-specific severity。

## Q7：下一步选择

按 aggregate Case A，选择 **A. D2-S1 Source-only Training**。未来 D2-S1 必须从 B0 ep25 + fresh E3 adapter initialization 重新训练，只把 P100 source 换为本轮 frozen P50；K=8、adapter、residual/cap、loss、optimizer/LR/epochs、common-visible sampler、Safe Commit、batch 与 samples/epoch 全部冻结。第一轮不得加入 Rescue/Preserve loss。

这个选择只是未来设计建议，不是本任务内的训练授权。由于 md3020 与 VAL-C 风险，D2-S1 preregistration 必须保留 per-view/per-target monitoring，但不得改成 view-specific P25/P50/P75。

## 结论边界

C-Paired 是 input transform 的因果比较；Candidate-Natural 仍来自不同 frame/target/time stage，只能支持 frozen representation distribution similarity。D2-P2 PASS 不能证明 P50 会提升 collaboration tracking、AUC、rescue capability 或降低 harmful rate。下一阶段若获授权，必须用独立 inner-dev gate 验证 source-only training，不能直接进入 official test。

## 主要 artifact

| artifact | rows / shape | SHA256 |
|---|---:|---|
| `train_representation_features.csv` | 14,046 | `0a8da1ca7cb45b257b2c04638bb64854931c36b65e9ccfb350a3956d747eb82d` |
| `train_candidate_prompts.npz` | `[9364,8,192]` FP16 | `213cc3b227f85a3fa4d9b2c86e8d8697b974926282af04b8b97386c529bb6131` |
| `selected_source.json` | P50 | `ca8587cd4d0eb8d7293c2e4c762da6f6c76a9612b43ccadf23411626ab912e60` |
| `val_representation_features.csv` | 2,396 | `b6b6e4fcbd77b2a7ebebe2368459b0297b471b599acfa0a075c6f998c531ce50` |
| `val_candidate_prompts.npz` | `[1198,8,192]` FP16 | `04be4e79ef0db30b8506caafdb7a82a3818b66211fc594f9ca90a05e88dc7361` |
| `train_distribution_distance.csv` | 35 | `39f3fc40b739a7b513bfe9e50de1b271f421898e4ab7526f50ca0324e592675b` |
| `val_distribution_distance.csv` | 21 | `26da395e7cc1358a83701ca903eed4d8a65eb9594915e65e92950c674d16931c` |
| `paired_severity_analysis.csv` | 6 | `31eb4a173161fb18936233bb46e30e239b5ae5509190cf5e7508ba43c6191048` |

完整命令见 `COMMANDS_ZH.md`，机器可读门禁与 hashes 见 `selected_source.json`、两个 representation manifest、`analysis_manifest.json` 和 `provenance.json`。
