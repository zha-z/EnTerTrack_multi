# B1-ABC-ARP 受控诊断报告

日期：2026-08-19

分支：`feature/pcum-cross-layer-arp`

结论状态：Stage A、B1 训练、inner-val 均已完成；未运行 `threemdot_test` 或 official test。

## 结论摘要

固定 ep25 的严格主比较中，B1-ABC-ARP 相对 E0 的 AUC 为 **-4.032381**，Precision 为 **-5.331166**，Norm Precision 为 **-5.058662**。因此当前证据不支持进入 PCUM；应保留 Plain ViT-Tiny 作为局部 baseline，并先诊断 ARP 的晚期退化与 compensation 的信息恢复质量。

Stage A 未给出 `LR_DROP_EPOCH=28` 是性能变化关键原因的可复现证据。E1 在降 LR 前已经下降，E3 在降 LR 前已经提升；E1 的 ep29 跳降没有在 E3 上复现。

## Stage A：ep25--30 closed-loop inner-val

所有评测均使用 `threemdot_val`，通过 `tracking/analysis_results.py` 最终调用 `calc_seq_err_robust`。success 使用 `IoU > threshold`，阈值 0.00--1.00、步长 0.05，使用 `target_visible`，不排除 invalid frame，按 15 条独立 sequence 做 macro average。

| Exp | Epoch | AUC | P | NP | Train loss | Val loss | Train IoU | Val IoU | Head LR | Backbone LR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 25 | 64.885643 | 84.071170 | 82.892541 | 0.733148 | 0.907914 | 0.832038 | 0.796248 | 8e-6 | 2.4e-6 |
| E1 | 26 | 62.836436 | 82.618914 | 80.125268 | 0.763609 | 0.932940 | 0.826288 | 0.787429 | 8e-6 | 2.4e-6 |
| E1 | 27 | 62.922760 | 82.635782 | 80.772988 | 0.726220 | 0.944610 | 0.833940 | 0.788854 | 8e-6 | 2.4e-6 |
| E1 | 28 | 64.714234 | 84.979279 | 82.833887 | 0.725094 | 0.988445 | 0.830468 | 0.778432 | 8e-6 | 2.4e-6 |
| E1 | 29 | 61.205622 | 78.692496 | 78.463749 | 0.724240 | 0.924404 | 0.833314 | 0.790029 | 8e-7 | 2.4e-7 |
| E1 | 30 | 62.163147 | 79.555616 | 79.503611 | 0.733681 | 0.938989 | 0.833374 | 0.785967 | 8e-7 | 2.4e-7 |
| E3 | 25 | 61.130692 | 79.020919 | 79.177443 | 0.756254 | 0.935480 | 0.825930 | 0.786579 | 8e-6 | 2.4e-6 |
| E3 | 26 | 61.465931 | 79.513312 | 79.603619 | 0.757712 | 0.889896 | 0.825797 | 0.792530 | 8e-6 | 2.4e-6 |
| E3 | 27 | 63.078353 | 83.393906 | 81.606710 | 0.774735 | 0.958595 | 0.824302 | 0.783737 | 8e-6 | 2.4e-6 |
| E3 | 28 | 62.995443 | 82.897013 | 80.800404 | 0.742226 | 0.976210 | 0.829731 | 0.778725 | 8e-6 | 2.4e-6 |
| E3 | 29 | 62.693499 | 82.942400 | 80.844298 | 0.750445 | 0.961676 | 0.828992 | 0.779586 | 8e-7 | 2.4e-7 |
| E3 | 30 | 63.299755 | 83.548768 | 81.660943 | 0.744276 | 0.939006 | 0.829987 | 0.784019 | 8e-7 | 2.4e-7 |

Q1：E1 在 LR drop 前已经下降。ep25→26 为 -2.049207 AUC，ep27 仍低；ep28 虽反弹至 64.714234，ep29 降 LR 后又跳降至 61.205622。LR drop 与 E1 的最大单次下降时间一致，但不是下降的起点。

Q2：E3 的提升在 LR drop 前开始。ep25→27 从 61.130692 增至 63.078353；ep29 小降，ep30 才在低 LR 下升至 63.299755。

Q3：没有足够证据认定 LR drop 是关键因素。两条训练轨迹对降 LR 的响应方向不一致，且 closed-loop AUC 与 train/val loss、IoU 也不呈稳定的同步关系。

完整 per-view、per-target、checkpoint SHA256、runid 和 prediction manifest 见 `stage_a_checkpoint_sweep.csv`、`checkpoint_manifest.csv`、`prediction_manifest.csv`。

## Stage B：受控配置与网络身份

E0 与 B1 保持相同的 split、common-visible sampler、seed 42、4 GPU DDP、batch size 2/GPU、`SAMPLE_PER_EPOCH=6000`、25 epoch、128/256 输入、patch16、ViT-Tiny dim192/depth6/heads3、预训练、optimizer、LR、weight decay、scheduler、augmentation、loss weights、ABC flatten 和 `REQUIRE_ALL_VIEWS_VISIBLE=true`。

| 配置 | E0 | B1-ABC-ARP |
|---|---|---|
| Backbone type | `vit_tiny_patch16_224` | `vit_tiny_patch16_224_arp` |
| `PRUNING_ENABLED` | false | true |
| `DYNAMIC_THRESHOLD_ENABLED` | false | true |
| `TOKEN_COMPENSATION_ENABLED` | false | true |
| `CE_LOC` | [] | [0] |
| `CE_KEEP_RATIO` | [] | [0.7] |
| ARP FLOPs loss weight | 0 | 0.03 |
| PCUM/C3R/FCVC | OFF | OFF |

`FLOPS_WEIGHT=0.03` 是仓库现有正式 ARP 包的设置，并非本轮新增调参。因此本报告测得的是整个既有 ARP/ATP/pruning/compensation 训练包相对 E0 的净效应，不能再细分为某一个子模块的纯因果效应。

普通非 flat actor 路径确实会通过 `_select_first_view()` 只取首视角；B1 复用 E0 的 `TRAIN.MULTIVIEW.FLAT_BASELINE=true` 路径，把 `[V,B,...]` 展为 `[V*B,...]`。全局聚合日志每个训练 epoch 为 A/B/C 各 6000 个样本、共 18000；每个 validation epoch 为 A/B/C 各 1496、共 4488。各 target×view 计数亦写入原始 ARP JSONL 日志。

## 主比较：固定 ep25

| Model | ARP | ATP | Compensation | Sampler | Epoch | Inner-val AUC | P | NP |
|---|---|---|---|---|---:|---:|---:|---:|
| E0 B0-ABC-Plain | OFF | OFF | OFF | common | 25 | 64.885643 | 84.071170 | 82.892541 |
| B1-ABC-ARP | ON | ON | ON | common | 25 | 60.853262 | 78.740005 | 77.833879 |
| B1 - E0 | | | | | | **-4.032381** | **-5.331166** | **-5.058662** |

B1 的预注册 sweep 为 ep15=64.146116、ep20=64.465713、ep25=60.853262。ep15/20 仅用于定位轨迹，不能替代固定 ep25 主比较；结果显示 ep20 后存在明显的 closed-loop 晚期退化。

### Per-view

| View | E0 AUC | B1 AUC | Delta |
|---|---:|---:|---:|
| A | 62.735955 | 61.839809 | -0.896146 |
| B | 61.312750 | 56.809809 | -4.502941 |
| C | 70.608222 | 63.910167 | -6.698055 |

Q3：退化主要发生在 C，其次是 B；A 只小幅下降。

### Per-target

| Target | E0 AUC | B1 AUC | Delta | 判定 | ep25 val keep ratio |
|---|---:|---:|---:|---|---:|
| md3016 | 64.329897 | 49.522173 | -14.807724 | harmful | 0.431541 |
| md3027 | 80.746982 | 80.534596 | -0.212385 | harmful | 0.493346 |
| md3034 | 56.982206 | 58.803499 | +1.821293 | helpful | 0.557251 |
| md3048 | 55.223121 | 45.118299 | -10.104822 | harmful | 0.650174 |
| md3055 | 67.146007 | 70.287741 | +3.141734 | helpful | 0.555918 |

helpful=2，harmful=3，tie=0。最严重的 sequence 是 md3016-B（-45.076092 AUC）和 md3048-C（-33.306379）；最大改善是 md3055-B（+12.486133）。详细 15 条 sequence 见 `b1_per_sequence_metrics.csv`。

## ARP 诊断

ep25 train：mean kept=161.3504，mean pruned=94.6496，keep ratio=0.630275，ATP threshold=0.464958±0.017372，compensation activation=0.369725。ep25 validation actor：mean kept=138.0809，mean pruned=117.9191，keep ratio=0.539378，threshold=0.464682±0.017044，compensation activation=0.460622。

ep25 validation 分视角 keep ratio 为 A=0.549567、B=0.533976、C=0.534592。B/C 的 pruning 略强于 A，与 view 退化方向相容，但 B/C keep ratio 几乎相同而 C 的 AUC 降幅更大，因此不能把 view 差异单独归因于 pruning 数量。

目标级相关性（仅 n=5）为：bbox area vs keep ratio `r=0.894748`，bbox area vs AUC delta `r=0.293775`，keep ratio vs AUC delta `r=0.206722`。样本级 ep25 validation actor 的 bbox area vs keep ratio 为 `r=-0.101380`。md3016 同时具有最低 keep ratio 和最坏 target delta，但 md3048 keep ratio 最高却仍严重退化。因此当前数据不支持“越小目标越被强剪枝，因而越差”的稳定单调解释；n=5 的 target 相关性只可视为诊断线索。

这些 bbox/keep 数据来自随机 validation actor 抽样，不是 closed-loop 推理每一帧的在线 pruning trace。推理 smoke 已确认真实物理 pruning、compensation、恢复 256 token 和 16×16 CENTER 输入均发生；这只证明空间尺寸恢复正确，不证明被删 token 的定位信息已充分恢复。

## 最终问答

- Q1：ep25→30 与 LR drop 没有明确、可复现的因果关系；E1 降低在 drop 前已出现，E3 提升也在 drop 前开始。
- Q2：相同 ABC/common-visible 条件、固定 ep25 下，既有 ARP 包的净效应为 **-4.032381 AUC**。
- Q3：主要伤害 C（-6.698055），其次 B（-4.502941）。
- Q4：明显伤害 md3016、md3048；帮助 md3034、md3055；md3027 近似持平但按严格符号归为 harmful。
- Q5：存在一些弱线索但无可靠单调关系；样本级相关很弱，target 级只有 5 点且出现明显反例。
- Q6：**保留 Plain backbone，暂不进入 PCUM；下一步先修/诊断 ARP。** 优先冻结当前预测并检查 ep20→25 的动态阈值/compensation 信息质量与 md3016-B、md3048-C 的 closed-loop failure，不在本轮擅自调 pruning ratio、threshold、loss 或 backbone。

## 边界与可复现性

- 未运行 `threemdot_test`、official test 或 outer holdout。
- 未按 test 选择 checkpoint；主结论固定使用 ep25。
- inference 为 no-GT，remote source 为 none。
- 25 个正式 checkpoint 全部保留；ep25 SHA256 为 `5de3d210f2a819eff9a4b94cd0aa97a3f81a7cf4682b79e80e86e700011c3932`。
- 详细命令见 `COMMANDS_ZH.md`，smoke 证据见 `SMOKE_TEST_ZH.md`，机器可读身份和来源见 `network_identity.json`、`provenance.json`。
