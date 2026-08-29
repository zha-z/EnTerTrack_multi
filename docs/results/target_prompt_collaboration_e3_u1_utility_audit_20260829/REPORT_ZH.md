# E3-U1 Target Prompt Per-Sender Counterfactual and Utility Audit

## 结论

本轮严格只在 `threemdot_val` 的 5 targets / 15 sequences 上使用冻结 E3 ep25 checkpoint，没有训练、没有运行 `threemdot_test`，也没有修改 E3 adapter、K、residual scale 或 sender weight。

冻结结论是 **Case B**：E3 K=8 representation 确实能产生可用的 single-sender rescue candidate，GT Oracle-single 相对 Local 有 **+1.4317 AUC point** headroom；但 prediction-only prompt feature 对 sender-specific helpful-vs-harmful 的 target-LOTO ROC 最高只有 **0.4436**，P4 OOF Safe Report 只兑现 **+0.0781 AUC point**，未达到预注册的 `+0.30`。因此不实现 Selective Target Prompt Report，不继续堆 selector feature；下一步优先 **B. E3-D1 Asymmetric Degradation Training**。

## 1. 冻结与完整性

- 预注册 commit：`63646b5eee0775377b0c2232c7f2e916dc653a1d`，早于本轮 GT join；
- 实现 commit：`b6e572077fbd1fce8c041e5ca4ee9694489ade97`；
- checkpoint SHA256：`d1c92bfe710e99d68b5f4a0eb2c151222204ca289bb1392f567fe76718d70c78`；
- prediction branch：56,448 rows，SHA256 `09d74a9f4060654b5843e8d3dd4355b0b8e669d71556bb49ea7c80b08f77d9c3`；
- prompt feature：28,224 rows，SHA256 `99ba5e643eceee40eb8e48769df66c914830f6104d5a4388ddb43a876a6048c3`；
- receiver/frame groups：14,112；active groups：14,097；GT-visible valid groups：13,513；
- 每 active view/frame local backbone forward 增量严格为 1；
- state mutation、Both report mismatch、Local state mismatch、`uses_gt=true` rows 均为 0；
- `both` 直接调用当前标准 E3 candidate 与 Safe Commit finalize 路径，因此不是重写的近似分支；标准 reported bbox mismatch=0；
- Local AUC 精确复现历史冻结 inner-val Local `64.8856`。

prediction-only 两个 CSV 先完成写入和 SHA256 manifest，独立复验 hash 后才加载 val GT。运行时 feature schema 不含 GT、IoU、visibility 或 label。

## 2. Oracle headroom（Q1）

统一 OSTrack `calc_seq_err_robust` 等价口径，15 个 A/B/C sequence 宏平均：

| Variant | AUC | Δ vs Local（point） | Precision | Norm Precision | Mean IoU |
|---|---:|---:|---:|---:|---:|
| Local | 64.8856 | 0 | 84.0712 | 82.8925 | 69.1493 |
| sender0-only | 64.7323 | -0.1533 | 84.2602 | 83.0440 | 68.9655 |
| sender1-only | 64.9073 | +0.0217 | 84.2336 | 82.7999 | 69.1785 |
| Both E3 | 64.8705 | -0.0152 | 84.2750 | 82.9609 | 69.1267 |
| P4 OOF selector | 64.9637 | +0.0781 | 84.2771 | 83.1105 | 69.2381 |
| **GT Oracle-single** | **66.3173** | **+1.4317** | **84.5531** | **84.0996** | **70.7311** |
| GT Oracle-4 | 66.3704 | +1.4847 | 84.5657 | 84.1090 | 70.7870 |

至少一个 single sender 满足 `delta_iou > 0.02` 的有效 receiver/frame 比例为 **12.2327%**。Oracle-4 只比 Oracle-single 多 `0.0530` point，说明主要 headroom 来自在 Local 与两个 single sender 之间正确选择，而不是 Both 的独有输出。

## 3. Single sender 与 Both（Q2）

固定 sender0 整体下降 `0.1533` point，固定 sender1 仅提升 `0.0217` point；Both 下降 `0.0152` point。sender1 比 Both 高 `0.0368` point，但幅度很小，而且 slot 混合了不同有向 view pair，不能解释为“sender1 永远更可靠”。

| 有向 pair | Helpful | Harmful | Tie | mean ΔIoU | AUC Δ（point） |
|---|---:|---:|---:|---:|---:|
| A <- B | 5.15% | 5.03% | 89.83% | -0.00036 | -0.1253 |
| A <- C | 5.55% | 6.68% | 87.77% | -0.00047 | -0.0681 |
| B <- A | 7.68% | 10.23% | 82.09% | -0.00320 | -0.2021 |
| B <- C | 9.61% | 8.17% | 82.22% | -0.00291 | -0.0508 |
| C <- A | 7.63% | 12.09% | 80.28% | -0.00128 | -0.1325 |
| C <- B | 7.43% | 6.44% | 86.13% | +0.00125 | +0.1839 |

所有有向 sender rows 中 helpful/harmful/tie 为 `7.23% / 8.17% / 84.60%`。Single candidate 偶尔有用，但固定使用任一 slot 都不能稳定提升。

## 4. 负迁移来源与 aggregation interaction（Q3）

good/bad 混合并不常见：

- sender0 helpful / sender1 harmful：70 frames（0.52%）；Both 14 helpful、0 harmful、56 tie；
- sender0 harmful / sender1 helpful：102 frames（0.75%）；Both 6 helpful、1 harmful、95 tie；
- 两 sender 都 helpful：301 frames；Both 297/301 helpful；
- 两 sender 都 harmful：370 frames；Both 361/370 harmful；
- 两者都 tie：10,193 frames，占 75.43%。

因此当前小幅负迁移不能主要归因于“一个好 sender 被一个坏 sender 平均掉”。更直接的问题是 rescue 稀疏，且当两个 single branch 同时 harmful 时，mean aggregation 有 **97.57%** 仍然 harmful。E3 representation 不是完全无信息（Oracle +1.43），但 common-visible 训练下 remote 的使用方式没有学会在 receiver 需要帮助时稳定产生正修正。

## 5. Prompt utility 可预测性（Q4/Q5）

固定 target-LOTO Logistic Regression：

| Group | Task-A ROC / PR | Task-B ROC / PR |
|---|---:|---:|
| P0 score/APCE | 0.3371 / 0.3614 | 0.4736 / 0.0717 |
| P1 + top-k concentration | 0.3438 / 0.3670 | 0.5118 / 0.0770 |
| P2 + prompt diversity | 0.3638 / 0.3860 | 0.5838 / 0.1014 |
| P3 + set compatibility | 0.4195 / 0.4470 | 0.5745 / 0.0969 |
| P4 + post-fusion disagreement | **0.4436 / 0.4346** | **0.6695 / 0.1836** |

Task-A positive prior 为 0.4694，P4 PR 仍低于 prior，ROC 远低于预注册 0.65。相对历史 E2A T4 ROC 0.4014 和 E2B pooled S5 ROC 0.3778，P3/P4 有有限改善，但没有形成可泛化的 sender-specific helpful-vs-harmful predictor。

Task-B 的 P4 ROC 0.6695 看起来更强，且高于 E2B Task-B 0.5172；但固定 0.5 threshold 的 precision/recall 只有 `0.4879 / 0.0619`，只在 213/14,112 receiver groups 选择 remote。最终 AUC 只增加 `0.0781` point，且 md3034 下降 `0.1659` point。这说明模型只找到极稀疏的高置信事件，没有达到跨 target 的预注册 tracking gate。

结论：prompt top-k/diversity/set feature 对 Task-B 比旧 scalar/pooled feature更有信息，但不能可靠解决核心 Task-A，也不能把 Oracle headroom兑现为足够的 OOF tracking 增益。

## 6. 预注册决策

| Gate | 结果 | 判定 |
|---|---:|---|
| Oracle-single 明显高于 Local | +1.4317 point | PASS |
| P4 Task-A LOTO ROC >= 0.65 | 0.4436 | FAIL |
| P4 OOF AUC >= Local +0.30 point | +0.0781 point | FAIL |
| Safe Commit / no-GT / prediction freeze | 全部通过 | PASS |

严格对应 **Case B**：representation 中存在信息，但“何时、使用哪一个 sender”仍难以从 prediction-only 观测跨 target 泛化。

Q6 下一步排序：

1. **B. Asymmetric Degradation Training**：让训练显式覆盖 receiver weak/remote strong 与 receiver strong/remote weak；
2. C. Sender aggregation redesign：仅作为 D1 后仍显示 Both 系统性损害 single 的后续方向；当前 mixed good/bad 覆盖不足；
3. D. 重新设计 prompt representation：当前已有 +1.43 Oracle headroom，不是第一优先；
4. A. Selective Target Prompt Report：本轮两项 selector gate 均失败，当前禁止实现。

本任务到此 STOP。不启动 E3-D1、不训练、不运行 official test。

