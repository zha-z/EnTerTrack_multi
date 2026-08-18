# B0-ABC-Plain 2x2 诊断报告

## 1. 结论

本轮 2×2 已按 `E1 -> E2 -> E3` 顺序完成，所有正式训练均为四卡、seed 42；只运行了 `threemdot_val`，没有新增 Three-MDOT official test/outer-holdout 运行。

| Experiment | Epoch | Sampler | Train loss | Val loss | Inner-val AUC | P / NP |
|---|---:|---|---:|---:|---:|---:|
| E0 | 25 | common-visible | 0.733148 | 0.907914 | **64.885643** | 84.071170 / 82.892541 |
| E1 | 50 | common-visible | 0.737171 | 0.969760 | 61.575448 | 79.231100 / 78.789703 |
| E2 | 25 | independent-view | 0.756254 | 0.935480 | 61.130692 | 79.020919 / 79.177443 |
| E3 | 50 | independent-view | 0.740845 | 0.969760 | 61.738772 | 79.250560 / 79.780794 |

这里的 train/val loss 是 checkpoint 中 rank-0 epoch meter。E0 和 E1-ep25 的网络权重 hash 完全相同，因此 E0 inner-val 严格复用 E1-ep25；E2 和 E3-ep25 同理。完整 checkpoint 文件因 config/stats 元数据不同而 hash 不同，但网络张量级 hash 相同。

结论是：**当前 independent-view sampler 确实扩大了有效帧覆盖，但没有提升这套 Plain ViT-Tiny 的 inner-val 泛化；延长训练也没有超过 E0。** 本轮不应进入 official test，也没有依据继续扩大该方向的 test sweep。

## 2. Q1–Q5

- **Q1：25 -> 50 epoch 是否提升泛化？** common-visible 的 endpoint AUC 下降 `3.310194`（64.885643 -> 61.575448）。independent-view 的 endpoint AUC 上升 `0.608080`（61.130692 -> 61.738772），其 inner-val 最佳点是 ep30 的 `63.299755`，但仍低于 E0。
- **Q2：common-visible -> independent-view 是否提升？** ep25 下降 `3.754951`；ep50 只比 common-visible-long 高 `0.163324`，且两者都低于 E0。因此没有实际泛化收益证据。
- **Q3：是否有交互作用？** endpoint 差分中的差分为 `(E3-E2)-(E1-E0)=+3.918275`，说明更长训练会抵消一部分 independent-view 的早期劣势；这是相对交互，不是超过 E0 的绝对收益。
- **Q4：训练曲线与 online AUC 是否一致？** 大方向一致：E1/E2 的 endpoint val loss 和 online AUC 都差于 E0。细节并不单调：E3 online 最佳在 ep30，而随机 val-loss 最低点不在 ep30，所以不能只依赖单个 epoch 的随机 val loss 选模型。
- **Q5：实际增加多少 frame diversity？** train split 中独立视角可见帧总量为 `42,737`，共同可见等价量为 `12,064×3=36,192`，潜在覆盖增加 **18.0841%**；A/B/C 分别增加 `21.3362% / 12.2928% / 20.6233%`。E2/E3 每个 epoch 的 distinct-view-frame group ratio 都是 `1.0`，累计 causal/visibility violation 都是 `0`。

## 3. Inner-val checkpoint sweep

| Experiment | ep25 | ep30 | ep35 | ep40 | ep45 | ep50 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 common-visible | **64.885643** | 62.163147 | 61.355343 | 61.765681 | 61.550585 | 61.575448 | ep25 |
| E3 independent-view | 61.130692 | **63.299755** | 62.721740 | 62.856730 | 62.974406 | 61.738772 | ep30 |

所有指标由 `tracking/analysis_results.py --tracker_name ...` 生成；该 CLI 直接使用 `lib.test.analysis.extract_results.calc_seq_err_robust`：IoU 比较符 `>`、阈值 0–1/0.05、使用 `target_visible`、不排除 invalid frame、15 个序列 macro average。详表见 [`inner_val_sweep.csv`](inner_val_sweep.csv)。

## 4. 原代码行为与 Plain 路径

- 普通 `forward_pass()` 在未启用特殊路径时调用 `_select_first_view()`，确实只取第一个 view。
- B0 配置已有 `TRAIN.MULTIVIEW.FLAT_BASELINE=true`；actor 的 `_forward_flat_views()` 将 `[V,B,...]` 展平为 `[V*B,...]`，因此 E0–E3 的 A/B/C 全部参与同一个 Plain ViT 的反向传播。
- backbone 为 `vit_tiny_patch16_224_half`：dim 192、depth 6、patch/stride 16；template 128 -> 64 token，search 256 -> **256 token**，完整进入 CENTER head，score map 为 16×16。
- ARP/ATP/pruning/token compensation/PCUM/C3R/FCVC/search prompt/remote state 全部关闭。正式 checkpoint 均为 170 个网络 tensor，PCUM/ATP key count 为 0。
- 初始化使用 `pretrained_models/tiny_7_OSTrack_ep0300.pth.tar`，SHA256 为 `94b20f6f955d5be1f00fb5765a489377535e2be86e0f3289781ae936035c2c4c`。源 checkpoint 的 12 个 ATP-only key 被严格排除，Plain 模型其余 core key 严格加载。

## 5. Independent-view sampler 实现

新增默认关闭的 `TRAIN.MULTIVIEW.INDEPENDENT_VIEW_SAMPLING=false`。只有 flat local baseline 可启用；若同时要求 common visibility，或开启 PCUM/C3R/FCVC，构建 dataloader 会 fail fast。

启用后仍先绑定同一 `target_id` 的规范 A/B/C 三序列，再分别从各 view 自身 visibility 中采一个 causal `(template, search)` pair；保持 `search > template` 和原 `MAX_SAMPLE_INTERVAL` 的逐步扩展语义。actor 仍收到 `[V,B,...]` 并 flatten，不引入跨视角交互。

E2 25 个 epoch、E3 50 个 epoch均满足：

- train：每 epoch A/B/C 各 6000；val：每 epoch A/B/C 各 1496（DDP `drop_last` 后全局数）。
- target×view 计数、frame id、delta_t、visible flag 全部写入 manifest。
- 每 epoch distinct group ratio=1.0；causal violation=0；visibility violation=0。
- E3 50-epoch 平均 delta_t mean：A `82.5966`、B `77.9321`、C `79.7323`；平均 unique-pair ratio：A `0.97454`、B `0.93588`、C `0.95323`。
- `md3008` 的 ABC common-visible 为 0/652，但独立可见帧为 A/B/C=`182/156/25`，已实际进入训练；`md3036` common 仅 7/786，而 A/C 各有 786 个可见帧。

可读数据见 E2/E3 的 `frame_sampling_epoch_metrics.csv`、`train_visibility_coverage.csv` 和 manifest inventory。完整逐样本 manifest 保留在正式 run 目录；Git 中提交代表性样本、逐 epoch 汇总和全部 SHA256 inventory，避免提交数百 MB 重复 JSONL。

## 6. Validation 随机性

当前 val 仍由随机 `TrackingSamplerThreeMDOT` 逐 epoch 重采样；`joint_transform` 仍包含随机灰度与水平翻转，`STARKProcessing` 仍接收 center/scale jitter。因此 val loss 是随机估计，不是 fixed-val。

E0/E1 的 val loss 可直接比较；E2/E3 的 val loss可直接比较。由于 sampler 因子按现有 dataloader 机制同时作用于 train/val，跨 common/independent 的 val-loss 分布不完全相同；跨 sampler 的主要可比泛化证据是固定 `threemdot_val` online tracking 指标。

## 7. Smoke 与正式运行

Smoke 逐项结果见 [`SMOKE_TEST_ZH.md`](SMOKE_TEST_ZH.md)，三个新实验均 PASS。

| Exp | Config | PID | GPU | Formal run directory | Checkpoint count |
|---|---|---:|---|---|---:|
| E1 | `b0_abc_plain_4gpu_ep50` | 1104141 | 0,1,2,3 | `output/diagnostics/b0_abc_plain_long/e1_run_20260818_seed42_4gpu_r002` | 50 |
| E2 | `b0_abc_plain_ind_sampler_4gpu` | 1334760 | 0,1,2,3 | `output/diagnostics/b0_abc_plain_ind_sampler/e2_run_20260818_seed42_4gpu_r001` | 25 |
| E3 | `b0_abc_plain_ind_sampler_4gpu_ep50` | 1447915 | 0,1,2,3 | `output/diagnostics/b0_abc_plain_ind_sampler_long/e3_run_20260818_seed42_4gpu_r001` | 50 |

三次正式运行均为 batch size `2/GPU`、4 个 DDP rank、`SAMPLE_PER_EPOCH=6000`、seed 42。E1/E3 在 epoch 28 的日志 LR 为 `8e-6/2.4e-6`，epoch 29 为 `8e-7/2.4e-7`，保持原 `LR_DROP_EPOCH=28`。

实际命令见各实验目录的 `launch_command.txt` 和 [`COMMANDS_ZH.md`](COMMANDS_ZH.md)。关键 checkpoint SHA256 见 [`checkpoint_manifest.csv`](checkpoint_manifest.csv)；checkpoint 本身约 65 MB/个，保留在本地 output 中，不提交 Git。

## 8. 文件索引

- `comparison_metrics.csv`：最终 2×2 endpoint。
- `inner_val_sweep.csv`：E1/E3 固定 checkpoint sweep。
- `checkpoint_manifest.csv`：关键 checkpoint 路径与 SHA256。
- `E1/`、`E2/`、`E3/`：训练 CSV、原始训练日志、inner-val summary/sequence CSV、bbox/manifest inventory、provenance。
- `smoke/`：smoke 日志和小型采样 manifest。

本目录没有新增 official-test 预测或指标。历史 E0 official-test AUC `48.9464` 属于另一个 split，不能与本报告的 inner-val AUC `64.8856` 混为同一结果。
