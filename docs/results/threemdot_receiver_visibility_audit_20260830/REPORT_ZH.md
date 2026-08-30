# D2-P0 Three-MDOT Natural Visibility 与 Receiver-visible Sampling Audit

## 冻结结论

本轮只读取 Three-MDOT train / val annotation，没有修改 dataset、sampler、actor、model 或 E3/E3-D1，没有训练、tracking rollout、`threemdot_test` 或 official test。

结论是：natural asymmetric visibility 在 train 中真实存在且常连续成段，但按冻结 receiver-frame 门槛不支持立即把 E3 sampler 重构为 receiver-visible：

- train：receiver-visible 相对 current common-visible capacity 增加 **18.0841%**，natural asymmetric 占 visible receiver-frame **15.3146%**；
- val：增加 **9.6745%**，natural asymmetric 占 **8.8211%**；
- Case A 的两项门槛没有在 train/val 同时达到；val 已落入 coverage `<10%` 的 Case B；
- 分布明显不均：val 的最大单 target 占 asymmetric frames **54.0268%**，top-3 占 **92.7852%**。

因此 Q8 选择 **B. 继续 common-visible + Rescue/Preserve Loss**。这里不是继续已经失败的 D1 synthetic occlusion，而是若后续另行授权，先在 inner-dev 预注册 receiver preserve/rescue objective。当前不实现 sampler。若未来重新考虑 receiver-visible，必须先解决 target balance 和 receiver-specific loss mask，不能简单设置 `REQUIRE_ALL_VIEWS_VISIBLE=false`。

## 1. Visibility schema 与当前 sampler

原始 sequence annotation 只有 `groundtruth.txt`、`occlusion.txt`、`out_of_view.txt`。训练 dataset 的真实定义为：

```text
valid_v[t] = bbox_w_v[t] > 0 AND bbox_h_v[t] > 0
visible_v[t] = (occlusion_v[t] == 0)
               AND (out_of_view_v[t] == 0)
               AND valid_v[t]
```

代码证据见 [`visibility_schema_audit.md`](visibility_schema_audit.md)。关键位置是 [`lib/train/dataset/threemdot.py`](../../../lib/train/dataset/threemdot.py) 第 119–152 行，以及 [`lib/train/data/sampler_threemdot.py`](../../../lib/train/data/sampler_threemdot.py) 第 124–134、268–275、296–344 行。

E3 YAML 固定 `REQUIRE_ALL_VIEWS_VISIBLE=true`；dataloader 将该值传给 train/val 的 `TrackingSamplerThreeMDOT`。最终逻辑是：

```text
common[t] = visible_A[t] AND visible_B[t] AND visible_C[t]
```

template 与 search 都从 common mask 采样且最终再次检查三视角全 true。当前 E3 actor 还在运行时硬性拒绝非 common-visible sampler。

原始标注复验：train 23 targets / 69 sequences，val 5 / 15；所有字段逐 view 等长，A/B/C 同步尾帧丢弃为 0；`occlusion` 值域 `{0,1}`，`out_of_view` 在这两个 split 中只有 `{0}`，同步范围内 bbox invalid 为 0。

`MISSING_DIAGNOSTIC`：annotation 没有 partial/full occlusion 分级、`absence` 或独立 raw `target_visible`。因此 `visible=1` 不能解释为完全无遮挡；本轮只能说不可见帧被二值 `occlusion` 字段标记。

## 2. Triplet visibility matrix

百分比分母是同 split 全部 synchronized A/B/C triplet。

| pattern A/B/C | train count | train % | val count | val % |
|---|---:|---:|---:|---:|
| 000 | 355 | 2.1998 | 0 | 0.0000 |
| 001 | 221 | 1.3694 | 2 | 0.0425 |
| 010 | 380 | 2.3547 | 0 | 0.0000 |
| 011 | 544 | 3.3709 | 466 | 9.9065 |
| 100 | 292 | 1.8094 | 0 | 0.0000 |
| 101 | 1,723 | 10.6767 | 39 | 0.8291 |
| 110 | 559 | 3.4639 | 90 | 1.9133 |
| 111 | 12,064 | 74.7552 | 4,107 | 87.3087 |
| **total** | **16,138** | **100** | **4,704** | **100** |

Visibility level：

| visible views | train count | train % | val count | val % |
|---:|---:|---:|---:|---:|
| 3 | 12,064 | 74.7552 | 4,107 | 87.3087 |
| exactly 2 | 2,826 | 17.5115 | 595 | 12.6488 |
| exactly 1 | 893 | 5.5335 | 2 | 0.0425 |
| 0 | 355 | 2.1998 | 0 | 0.0000 |

完整 CSV：[`train_triplet_visibility_patterns.csv`](train_triplet_visibility_patterns.csv)、[`val_triplet_visibility_patterns.csv`](val_triplet_visibility_patterns.csv)、[`train_visibility_levels.csv`](train_visibility_levels.csv)、[`val_visibility_levels.csv`](val_visibility_levels.csv)。

## 3. Common-visible 与 receiver-visible coverage

| split | synchronized triplets | N_common | current `3*N_common` | receiver A | receiver B | receiver C | receiver total | extra | increase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 16,138 | 12,064 | 36,192 | 14,638 | 13,547 | 14,552 | 42,737 | 6,545 | 18.0841% |
| val | 4,704 | 4,107 | 12,321 | 4,236 | 4,663 | 4,614 | 13,513 | 1,192 | 9.6745% |

相对于全部 synchronized triplet，当前 common-visible 筛掉 train `4,074 / 16,138 = 25.2448%`、val `597 / 4,704 = 12.6913%`。但不是每个被筛 triplet 都有可训练 receiver；换成 receiver-frame 口径后，额外可准入量分别是 +6,545 和 +1,192，即上表的 +18.0841% / +9.6745%。

每 target 的 N_common：

| split | min | max | mean | median |
|---|---:|---:|---:|---:|
| train | 0 | 1,028 | 524.5217 | 510 |
| val | 463 | 1,136 | 821.4000 | 805 |

train 的 `md3008` 为 N_common=0，因此当前 common-visible sampler 完全无法从该 target 返回训练 pair。逐 target 明细见 [`train_per_target_visibility.csv`](train_per_target_visibility.csv) 和 [`val_per_target_visibility.csv`](val_per_target_visibility.csv)；总体比较见 [`coverage_comparison.csv`](coverage_comparison.csv)。

## 4. Receiver-visible sender states

百分比分母为各自的 visible receiver-frame。

| split / receiver | receiver frames | R2: two senders visible | R1: one sender visible | R0: zero sender visible | natural asym R1+R0 |
|---|---:|---:|---:|---:|---:|
| train A | 14,638 | 12,064 (82.4156%) | 2,282 (15.5896%) | 292 (1.9948%) | 2,574 (17.5844%) |
| train B | 13,547 | 12,064 (89.0529%) | 1,103 (8.1420%) | 380 (2.8050%) | 1,483 (10.9471%) |
| train C | 14,552 | 12,064 (82.9027%) | 2,267 (15.5786%) | 221 (1.5187%) | 2,488 (17.0973%) |
| **train overall** | **42,737** | **36,192 (84.6854%)** | **5,652 (13.2251%)** | **893 (2.0895%)** | **6,545 (15.3146%)** |
| val A | 4,236 | 4,107 (96.9547%) | 129 (3.0453%) | 0 | 129 (3.0453%) |
| val B | 4,663 | 4,107 (88.0763%) | 556 (11.9237%) | 0 | 556 (11.9237%) |
| val C | 4,614 | 4,107 (89.0117%) | 505 (10.9450%) | 2 (0.0433%) | 507 (10.9883%) |
| **val overall** | **13,513** | **12,321 (91.1789%)** | **1,190 (8.8063%)** | **2 (0.0148%)** | **1,192 (8.8211%)** |

本地 train/val 的 `out_of_view==1` 和 invalid bbox 都是 0，因此所有 natural asymmetric receiver-frame 都可描述为“visible receiver + 至少一个 sender 的 `occlusion==1`”。这仍不能区分部分/完全遮挡。完整字段和可重叠 cause count 见 [`train_receiver_visibility_summary.csv`](train_receiver_visibility_summary.csv) 与 [`val_receiver_visibility_summary.csv`](val_receiver_visibility_summary.csv)。

## 5. Target concentration 与时间连续性

| split / receiver | runs | asymmetric frames | mean | median | P90 | max |
|---|---:|---:|---:|---:|---:|---:|
| train A | 66 | 2,574 | 39.0000 | 14 | 66.0 | 767 |
| train B | 56 | 1,483 | 26.4821 | 19 | 59.5 | 190 |
| train C | 62 | 2,488 | 40.1290 | 17 | 73.0 | 767 |
| **train overall** | **184** | **6,545** | **35.5707** | **17** | **63.4** | **767** |
| val A | 4 | 129 | 32.2500 | 33.5 | 38.1 | 39 |
| val B | 12 | 556 | 46.3333 | 36.5 | 89.4 | 148 |
| val C | 9 | 507 | 56.3333 | 40 | 104.8 | 148 |
| **val overall** | **25** | **1,192** | **47.6800** | **37** | **94.0** | **148** |

这些 case 显然不是以单帧随机噪声为主：train median 17、P90 63.4、max 767；val median 37、P90 94、max 148。它们更接近连续跟踪困难阶段。

但 target 分布不均：train 18/23 targets 有 natural asymmetry，最大 target `md3036` 占 23.8044%，top-3 占 49.8549%；val 5/5 targets 都有，但 `md3034` 单独占 54.0268%，top-3 占 92.7852%。这要求任何未来 receiver-visible 方案先考虑 target-balanced sampling，不能按 raw receiver-frame 全量灌入。

逐 `target + receiver` 统计见 [`train_asymmetric_run_lengths.csv`](train_asymmetric_run_lengths.csv) 和 [`val_asymmetric_run_lengths.csv`](val_asymmetric_run_lengths.csv)。

## 6. 与 E3-D1 的关系

两者不等价：

| 当前 E3-D1 | 候选 natural receiver-visible |
|---|---|
| 先从 111 common-visible pool 采样 | 从 receiver 自身 visible 的同步帧采样 |
| 50% triplet 随机选择恰好一个 view | sender 的自然状态来自 annotation 对应帧 |
| 用 GT target crop bbox 把该 view 的 search target-region 写 0 | 不人工覆盖图像 |
| 三个 receiver 原 annotation 与常规 loss 仍保留 | 只有 visible receiver 的 localization loss 有效 |
| synthetic weak view 原因明确为 augmentation | natural sender invisible 的细因受 schema 限制 |

E3-D1 ep25 已是冻结负结果；本统计不能把自然 asymmetry 等同为“换真实遮挡后 D1 必然有效”。

## 7. Q1–Q8

**Q1：当前实际筛掉多少潜在数据？**

Triplet diversity：train 4,074（25.2448%），val 597（12.6913%）不是 111。Receiver-frame capacity：相对 receiver-visible pool，当前少 6,545 train frames 和 1,192 val frames；相对当前 capacity 是 +18.0841% / +9.6745%。

**Q2：减少 optimizer step 还是 candidate frame diversity？**

主要减少 candidate frame diversity 和 target support，不减少预设 optimizer steps。配置固定 `SAMPLE_PER_EPOCH=6000/1500`，sampler `__len__` 仍返回固定长度；成功取样时每 epoch batch/step 数不变。不可用 target/frame 会触发重采样，`md3008` 因 N_common=0 完全退出当前训练候选，而不是让 epoch 自动少走相同步数。

**Q3：receiver visible 后 train / val 增加多少？**

train `36,192 -> 42,737`，+6,545，+18.0841%；val `12,321 -> 13,513`，+1,192，+9.6745%。

**Q4：两个 sender 状态占比？**

train overall R2/R1/R0 = 84.6854% / 13.2251% / 2.0895%；val = 91.1789% / 8.8063% / 0.0148%。分 receiver 见第 4 节。

**Q5：natural asymmetric collaboration 是否足够多？**

train 有实质覆盖（15.3146%），val 边缘不足（8.8211%），且没有跨 split 达到冻结的 `coverage >=20% AND asymmetry >=10%` 双门槛。因此“存在并值得保留为后续候选”成立，“足以优先重构训练”不成立。

**Q6：零散还是连续？**

连续成段。train/val pooled median 为 17/37 帧，P90 63.4/94 帧；但长段集中在少数 target。

**Q7：common-visible 是否构成重要数据分布瓶颈？**

它是 train 的中等分布瓶颈，并造成 `md3008` 完全缺席；但没有证据把它定为跨 split 的主要瓶颈：val expansion 小于 10%，asymmetry 小于 10%，且 target concentration 很强。结论是“有偏但尚非首要重构项”。

**Q8：下一步选什么？**

选择 **B. 继续 common-visible + Rescue/Preserve Loss**；不重复当前 D1 augmentation，不实现 receiver-visible sampler。未来只有在独立授权与预注册下，才设计新的 preserve/rescue objective。若再评估 receiver-visible，应使用 target-balanced sampling 与 receiver-specific loss mask。

## 8. 未来候选设计边界（未实现）

若未来证据与授权允许 Natural Asymmetric Receiver-visible Training，最小逻辑应是：

```text
for synchronized frame t:
    for receiver r in {A,B,C}:
        receiver_loss_valid[r,t] = GT_visible[r,t]   # train only
        if receiver_loss_valid[r,t]:
            compute receiver localization loss
        else:
            mask all normal receiver localization losses
```

sender 可 visible 或 invisible；GT sender visibility 只允许用于 training sample construction 或 auxiliary supervision。validation/inference 禁止把 receiver/sender GT visibility 用作 `remote_valid`、sender weight、gate、selector 或 runtime fallback，否则是 GT leakage。

## 9. 完整性

- train/val target overlap：0；A/B/C group 均完整；同步尾帧丢弃：0。
- pattern、visibility level、receiver count、R2/R1/R0、run-frame 五类守恒检查全部 PASS。
- annotation aggregate SHA256 与所有 CSV SHA256 见 [`provenance.json`](provenance.json)。
- 实际命令见 [`COMMANDS_ZH.md`](COMMANDS_ZH.md)。
- official test accessed=false；training=false；validation tracking rollout=false。
