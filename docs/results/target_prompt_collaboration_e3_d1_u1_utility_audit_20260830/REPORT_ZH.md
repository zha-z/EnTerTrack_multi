# E3-D1-U1 Asymmetric-Degradation Per-Sender Counterfactual Utility Audit

## 结论

本轮严格只使用冻结的 E3-D1 ep25 checkpoint，在 `threemdot_val` 的 5 targets / 15 sequences 上完成 Local、sender0-only、sender1-only、Both 四分支 prediction-only rollout。没有训练、没有访问 `threemdot_test`、没有重跑 official test、没有 selector/classifier、没有参数 sweep，也没有实现 D2。

冻结结论是 **Case C**：D1 没有增强 single-sender rescue capability。GT Oracle-single 从原 E3 的 **66.3173** 降至 **66.1924**，即从相对 Local 的 **+1.4317** 降至 **+1.3067 AUC point**；绝对下降 **0.12495 point**，超过预注册的 `0.10 point` 明显下降门槛。同时，remote-help 与 helpful rate 均略降，harmful rate 从 **8.17%** 大幅升至 **13.74%**。因此当前 synthetic target-box occlusion 更符合“破坏了 rescue representation/candidate”，而不只是“更会救但也更会伤”。下一步选择 **B. 重新设计 degradation（training objective / degradation source）**，停止继续 D1；禁止直接 sweep `TRIPLET_RATIO` 或 `OCCLUSION_BOX_SCALE`。

## 1. Q1–Q8 直接回答

| 问题 | 冻结回答 |
|---|---|
| Q1：D1 是否真正增强 single-sender rescue？ | **否**。Oracle-single、remote-help 与 helpful rate 都没有增加；固定 S0/S1 AUC 还分别低于 Local `1.0668/0.6164 point`。 |
| Q2：Oracle-single 相比原 E3 +1.4317 如何变化？ | D1 Oracle-single 为 **66.1924**，headroom 为 **+1.3067 point**；较 E3 Oracle-single **下降 0.12495 point**，按预注册门槛属于明显下降。 |
| Q3：remote-help 相比 12.2327% 如何变化？ | 降至 **11.2928%**，下降 **0.9398 percentage point**（1,526 / 13,513 visible valid receiver frames）。 |
| Q4：helpful/harmful/tie 如何变化？ | `7.23/8.17/84.60% -> 7.12/13.74/79.14%`：helpful `-0.1110` point，harmful `+5.5687` points，tie `-5.4577` points。 |
| Q5：更大的 residual 主要在 helpful 还是 harmful frame？ | **更偏 harmful**。pooled single 的 Q4 为 `9.64% helpful / 19.59% harmful`，Both Q4 为 `8.36% / 17.24%`；两者 mean delta IoU 都更负。 |
| Q6：cap-hit 更有帮助还是更危险？ | **更危险**。single cap-hit 为 `9.43% helpful / 17.64% harmful`、mean delta IoU `-0.01371`；Both cap-hit 为 `9.17% / 16.86%`、mean `-0.02439`。 |
| Q7：official AUC 下降符合 A/B/C/D 哪种解释？ | **C. 破坏了 rescue representation**。不是 A，因为 rescue 指标下降；不是 B，因为 Oracle 也明显下降；不是 D，因为 inner-val Oracle 与 Both 都未上升。 |
| Q8：下一步 A/B/C/D？ | **B. 重新设计 degradation**。按 Case C 停止继续 D1，重新设计 objective / degradation source；本轮不实现。 |

## 2. 冻结、恒等与无泄漏

- source HEAD：`8d3fa0db6c8ad0fe83f447033df497fd0303cd25`；branch：`feature/pcum-cross-layer-arp`；
- checkpoint SHA256：`231e20df89c184dbe5411fa687b625b809d4c1580df64059ec9d50f3c62a1c2b`；
- prediction branches：56,448 rows，SHA256 `72a4b3345fbb1f6807bac17bbeaf090e51f09cbe0bc20319492b1c28d04f9942`；
- prompt features：28,224 rows，SHA256 `c5702a26afd47f3ba03431ddfdf933b916eb9de6017a524e9fe95abc24d38087`；
- receiver/frame groups：14,112；active groups：14,097；GT-visible valid groups：13,513；
- 14,112 个 D1 Local bbox 与冻结 E3-U1 Local 逐 CSV 字符串精确一致；
- `local_forward_mismatch=0`、`state_mutation=0`、`both_report_mismatch=0`、`local_state_mismatch=0`、`uses_gt_rows=0`；
- 每个 active receiver/frame 的 Local backbone forward 增量严格为 1；Both 复用标准 candidate/finalize 路径，报告 Both、提交 Local；
- prediction-only schema 不含 GT bbox、IoU、visibility、helpful/harmful 或 oracle；`uses_gt` 全为 false；
- 两个 prediction CSV 原子写入 staging 并记录 manifest，独立进程复验 bytes/rows/SHA256 与全量 Local 恒等后才搬入正式目录；搬运后 SHA256 再次一致，随后才加载 val GT；
- post-hoc labels 每行绑定冻结 prediction SHA256；分析完成后 prediction SHA256 保持不变。

## 3. Tracking 与 Oracle 对照

15 个 sequence 按 E3-U1 相同 OSTrack curve 口径宏平均：

| Variant | E3 AUC | E3-D1 AUC | D1 vs Local（point） | D1 vs E3（point） |
|---|---:|---:|---:|---:|
| Local | 64.8856 | 64.8856 | 0 | 0 |
| sender0-only | 64.7323 | 63.8188 | -1.0668 | -0.9135 |
| sender1-only | 64.9073 | 64.2692 | -0.6164 | -0.6381 |
| Both | 64.8705 | 64.0251 | -0.8605 | -0.8453 |
| **GT Oracle-single** | **66.3173** | **66.1924** | **+1.3067** | **-0.12495** |
| GT Oracle-4 | 66.3704 | 66.2495 | +1.3639 | -0.12086 |

D1 仍保留部分可用 candidate（Oracle-single 相对 Local 仍有 +1.3067 point），但核心问题是“是否增强”：答案为否。Oracle-4 仅比 Oracle-single 多 `0.0571 point`，Both 没有提供足以改变结论的独有 headroom。

## 4. Sender utility 与 view/target 集中性

27,026 个 visible valid directed sender rows 中：

| 指标 | E3 | E3-D1 | D1 - E3 |
|---|---:|---:|---:|
| Helpful | 7.2301%（1,954） | 7.1191%（1,924） | -0.1110 point |
| Harmful | 8.1736%（2,209） | 13.7423%（3,714） | +5.5687 points |
| Tie | 84.5963%（22,863） | 79.1386%（21,388） | -5.4577 points |
| mean delta IoU | -0.001189 | -0.006663 | -0.005474 |

最差有向 pair 是 `B <- A`：harmful **23.33%**、helpful **5.58%**、AUC 相对 B-Local **-2.1623 points**。按 receiver view，B 的 Both 相对 Local 为 **-1.8502 points**，明显比 A 的 `-0.3573` 和 C 的 `-0.3740` 更差。

target 维度上，D1 Both 相对 E3 Both 在 5 个 target 中 4 个下降；`md3048` 最大下降 **2.9380 points**。Oracle-single 相对 E3 在 5 个 target 中 4 个下降，仅 `md3034` 微增 `0.0646 point`，不构成普遍 rescue 增强。

## 5. Both interaction

- 两个 single 都 helpful：398 frames；Both 有 394/398（98.99%）仍 helpful；
- 两个 single 都 harmful：981 frames；Both 有 966/981（98.47%）仍 harmful；
- sender0 helpful / sender1 tie：448 frames；Both 只保留 161（35.94%）为 helpful；
- sender0 tie / sender1 helpful：564 frames；Both 只保留 174（30.85%）为 helpful；
- 两个 single 都 tie：9,370 frames（69.34%）；Both 仍有 28 helpful、17 harmful。

因此 mean aggregation 能保留“双 helpful”，但同样几乎完整保留“双 harmful”；面对单个 rescue + tie 时，多数被稀释为 tie。D1 后更突出的主问题是 harmful candidate 数量大增，而不是 Both 能可靠 preserve sparse rescue。

## 6. Residual quartile、cap 与相关性

D1 相对 E3 的 active post-init mean relative residual norm：S0 `0.11517 -> 0.18247`，S1 `0.11530 -> 0.17098`，Both `0.10835 -> 0.15565`，确认 residual 明显增强。

| 分支 | 低 residual Q1 helpful/harmful | 高 residual Q4 helpful/harmful | Q1 mean delta IoU | Q4 mean delta IoU |
|---|---:|---:|---:|---:|
| pooled single | 5.66% / 7.78% | 9.64% / 19.59% | -0.00197 | -0.01208 |
| Both | 3.82% / 5.69% | 8.36% / 17.24% | -0.00131 | -0.01242 |

较大 residual 会同时增加 helpful 与 harmful 的非 tie 事件，但 harmful 增长更大，净效用更负。Spearman `relative_residual_norm` vs `delta_iou` 为 pooled single **-0.0641**、Both **-0.0380**：相关性弱，但方向不是“residual 越大越有用”。

实际模型 cap 为 `0.25`，命中规则固定为 `relative_residual_norm >= cap - 1e-6`：

| 分支 | cap-hit 比例 | Helpful | Harmful | Tie | mean delta IoU |
|---|---:|---:|---:|---:|---:|
| pooled single | 12.9982%（3,509/26,996） | 9.43% | 17.64% | 72.93% | -0.01371 |
| Both | 5.0081%（676/13,498） | 9.17% | 16.86% | 73.96% | -0.02439 |

cap-hit 不是纯 harmful，也会包含少量强 rescue；但 harmful 比 helpful 多、mean delta IoU 显著更负，所以在本轮冻结数据上应解释为更危险，不能据此改 cap 或生成 runtime threshold。

## 7. 与冻结 official TEST 的关系和停止条件

本轮没有访问或重跑 TEST，只引用既有冻结报告：D1 official AUC `48.6930`，低于 E3 `48.7651`（-0.0720 point）；D1 official active rows 中约 9.85% 触及 cap。当前 val 同时出现 Oracle-single 下降、Both 下降、remote-help 下降与 harmful 大增，所以不满足 Case D 的“Oracle 与 Both inner-val 上升”。更一致的解释是 Case C：当前 synthetic target-box occlusion 损害 useful rescue representation/candidate。

最终动作：**STOP E3-D1**。下一步仅建议重新设计 degradation objective/source；不授权训练、D2、prompt redesign、sampler 修改、参数 sweep 或再次 official test。若未来研究 natural receiver-visible，必须作为独立实验处理 receiver-specific loss mask 与 target balance。
