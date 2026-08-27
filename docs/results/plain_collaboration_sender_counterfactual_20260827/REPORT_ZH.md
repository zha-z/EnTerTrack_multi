# Plain Collaboration V1 E1.5：逐 sender 反事实诊断

## 1. 结论先行

本轮只在 `threemdot_val` 上复用冻结的 V1 epoch 25 checkpoint，没有训练、没有修改 adapter、没有访问 official test。

结论是：当前 full remote search feature 中存在可利用的信息，但还没有可靠的 prediction-only 选择机制将其转化为稳定收益。

- Local AUC：`64.8856`
- 固定 sender0-only AUC：`64.6706`（相对 Local `-0.2150` AUC 点）
- 固定 sender1-only AUC：`65.0461`（相对 Local `+0.1604` AUC 点）
- Both 0.5/0.5 Safe AUC：`64.8943`（相对 Local `+0.0086` AUC 点）
- GT Oracle-single AUC：`66.2076`（相对 Local `+1.3220` AUC 点）
- GT Oracle-4 AUC：`66.2573`（相对 Local `+1.3717` AUC 点）
- 至少一个 remote branch 达到 `delta_iou > 0.02` 的有效帧：`11.8849%`

Oracle 是使用 GT 的 post-hoc upper bound，不能部署，也不能作为 runtime selector。`+1.37` AUC 点说明 representation 仍有中等、值得继续诊断的 headroom；它不代表当前 collaboration 已获得同等提升。

本轮最符合预设决策规则的 **Case B**：信息存在，但现有 sender score/APCE/entropy 的可预测性弱。Case A 只得到部分支持；Case C 不成立；Case D（主要故障是好坏 sender 被强制平均）不成立。

## 2. 实验边界与因果约束

- 分支：`feature/pcum-cross-layer-arp`
- 源 HEAD：`affcd40c67e31ca976164fd92f35197ffaa61cf2`
- 数据：`threemdot_val`，5 targets、15 sequences；未读取 `threemdot_test`。
- checkpoint：`EnTeRTrack_ep0025.pth.tar`
- checkpoint SHA256：`0607e8dcd05d9732a0058bb7a3d9e356474dbeb219a65e0472ac83a83d58ad40`
- 没有训练，没有 optimizer/backward，没有调 residual scale、adapter、attention head 或 feature representation。
- runtime 所有 branch 都只读取 prediction-only 数据，`uses_gt=false`。
- 每个 receiver/frame 只执行一次 local backbone，四个 branch 共用同一个 local candidate 和同一个 pre-state。
- `local`、`sender0_only`、`sender1_only`、`both` 都不写 persistent tracker state；最终 state 始终提交 local，`both` 仅作为默认 reported result。
- 正常 V1 路径仍严格要求两个 canonical sender；允许 R=1 只存在于默认关闭的 sender-counterfactual diagnostic 模式。

sender0/sender1 是 canonical 顺序，不是训练得到的优先级：

| Receiver | sender0 | sender1 |
| --- | --- | --- |
| A | B | C |
| B | A | C |
| C | A | B |

因此“sender1-only 更好”不能解释为 slot 1 本身更优，必须查看有物理含义的有向 view pair。

## 3. 运行完整性

完整 rollout 产生 14,112 个 receiver/frame group、56,448 行冻结预测，每组严格四个 branch。后验分析中有 13,513 个 target-visible 有效 receiver frame。

审计结果：

- 15 个 sequence 的 sender-counterfactual CSV 全部存在。
- active prediction rows：56,388；初始化 rows：60。
- local forward counter 对每帧三路 receiver 的增量严格为 `(1, 1, 1)`。
- persistent digest mutation：0。
- `uses_gt=true`：0。
- E1.5 `local` bbox 与既有 D0-local：0 个不一致。
- E1.5 `both` bbox 与既有 D0-safe：0 个不一致。
- E1.5 `both` bbox 与本轮标准输出 txt：0 个不一致。
- search token count：256。
- `sender0_only`/`sender1_only` remote count 为 1，`both` 为 2；both 权重保持 0.5/0.5。

预测先冻结为 `prediction_only_sender_counterfactual.csv`，再进行 GT join。冻结文件 SHA256：

```text
75ebb3cb292202346619e6f81cc46d050fa394a73e7c0d01ad9376545ba95f43
```

join 阶段重新验证该 SHA；冻结文件 schema 中没有 GT、IoU 或 helpful/harmful label。

## 4. Q1：逐 sender 对 receiver 的影响

helpful 定义为 `delta_iou > 0.02`，harmful 定义为 `delta_iou < -0.02`，其余为 tie。

| 有向 pair | 有效行 | Helpful | Harmful | Tie | mean delta IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| A <- B | 4,236 | 4.30% | 14.92% | 80.78% | -0.00265 |
| A <- C | 4,236 | 6.87% | 8.62% | 84.51% | +0.00053 |
| B <- A | 4,663 | 4.42% | 6.67% | 88.91% | -0.00147 |
| B <- C | 4,663 | 10.79% | 5.40% | 83.81% | +0.00444 |
| C <- A | 4,614 | 7.11% | 13.68% | 79.22% | -0.00093 |
| C <- B | 4,614 | 9.75% | 5.27% | 84.98% | +0.00271 |

最清楚的正向关系是 `B <- C` 和 `C <- B`；最差的是 `A <- B`。sender C 的总体 helpful/harmful 为 `8.92% / 6.93%`，优于 sender A 的 `5.76% / 10.15%`。这表明 sender 质量具有明确的有向 pair 异质性，不能只用一个全局固定 sender 权重描述。

## 5. Q2：单 sender 与 0.5/0.5 双 sender

| Branch | Helpful | Harmful | Tie | AUC |
| --- | ---: | ---: | ---: | ---: |
| sender0-only | 5.30% | 11.65% | 83.05% | 64.6706 |
| sender1-only | 9.21% | 6.36% | 84.43% | 65.0461 |
| both 0.5/0.5 | 5.70% | 6.93% | 87.37% | 64.8943 |

答案不是“任一 single sender 都更稳定”。canonical sender1-only 优于 both：AUC 高 `0.1518` 点、helpful 更高且 harmful 更低；但 sender0-only 明显比 both 差。真实结论是 **sender/pair-specific selection 可能有价值**，而不是固定退化成单 sender。

分 receiver 的 AUC 也一致显示这种非对称性：

| Receiver | Local | sender0-only | sender1-only | Both | Oracle-4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 62.7360 | 62.3705 | 62.5583 | 62.5149 | 63.7593 |
| B | 61.3128 | 61.1385 | 61.7586 | 61.4393 | 62.9541 |
| C | 70.6082 | 70.5027 | 70.8213 | 70.7286 | 72.0585 |

## 6. Q3：negative transfer 是否主要来自好坏 sender 混合

不是。13,513 个有效 frame 中，仅 37 帧属于“一 sender helpful、另一 sender harmful”：

- sender0 helpful / sender1 harmful：14 帧；both 为 helpful/harmful/tie = `6/1/7`。
- sender0 harmful / sender1 helpful：23 帧；both 为 helpful/harmful/tie = `1/0/22`。

相反：

- 两个 sender 都 helpful：368 帧；both 有 366 帧 helpful。
- 两个 sender 都 harmful：484 帧；both 有 480 帧 harmful。

所以 good/bad 强制平均是可观察问题，但发生太少，不能解释主要 negative transfer。更主要的现象是两个 single branch 的效果同号，或者一个 branch 处于 tie。尤其两者都 harmful 时，0.5/0.5 几乎总是保留 harmful。

## 7. Q4：prediction-only reliability

单变量 helpful-vs-harmful 诊断：

| Feature | ROC-AUC | PR-AUC | 解释 |
| --- | ---: | ---: | --- |
| sender low entropy | 0.5236 | 0.4769 | 最好的 sender 自身单变量，但接近随机 |
| sender score | 0.5159 | 0.4749 | 弱 |
| sender APCE | 0.5091 | 0.4742 | 弱 |
| receiver low entropy | 0.5470 | 0.5556 | 所列 pre-fusion 单变量中最高，仍弱 |
| receiver APCE | 0.5400 | 0.5544 | 弱 |
| receiver score | 0.5348 | 0.5418 | 弱 |
| sender-receiver score | 0.4867 | 0.4530 | 无正向判别力 |
| sender-receiver APCE | 0.4822 | 0.4539 | 无正向判别力 |

因此，对“sender 自己的哪项最能预测 helpful”的直接回答是 **低 entropy**，ROC-AUC `0.5236`；但其强度不足以支持部署 sender gate。

固定三组 feature 的 leave-one-target-out Logistic Regression 也支持这一结论：

| Task | Feature set | ROC-AUC | PR-AUC | 备注 |
| --- | --- | ---: | ---: | --- |
| helpful vs harmful | sender-only | 0.4226 | 0.4239 | 非 tie coverage 16.28% |
| helpful vs harmful | receiver + sender | 0.3738 | 0.3817 | 更差 |
| helpful vs harmful | + post-fusion disagreement | 0.5625 | 0.5210 | 最好但仍弱，且 remote 后才可得 |
| replace local | sender-only | 0.5447 | 0.0806 | 正类率 7.26%，precision@0.5 8.86% |
| replace local | receiver + sender | 0.4847 | 0.0714 | 不可用 |
| replace local | + post-fusion disagreement | 0.4547 | 0.1124 | PR 上升但 ROC 低于随机 |

没有进行 feature search 或阈值调参。这些分类器只用于诊断，不是新增 runtime gate。

## 8. Communication Trigger 与 Report Selector

两者不能混为一谈：

1. **Communication Trigger** 必须在 remote 到达前决定，只能使用 local prediction/history。当前 local score/APCE/entropy 的判别力弱，本轮没有证据支持可靠的 pre-communication trigger。
2. **Collaboration Commit/Report Selector** 在 remote 已到达、各 branch 已计算后决定是否报告 collaboration。它可以使用 local-vs-collab displacement、branch disagreement 和 sender reliability。post-fusion feature set 达到 ROC-AUC `0.5625`，但仍不足以成为稳健 selector。

`local_collab_center_displacement` 或任何 post-fusion disagreement 都不能宣称为 communication trigger。

## 9. Q5：Oracle 与现实上限

| Variant | AUC | 相对 Local | GT runtime |
| --- | ---: | ---: | --- |
| Local | 64.8856 | 0 | 否 |
| Both Safe | 64.8943 | +0.0086 | 否 |
| Oracle-single | 66.2076 | +1.3220 | **是，仅 upper bound** |
| Oracle-4 | 66.2573 | +1.3717 | **是，仅 upper bound** |

按 receiver 的 `remote_help_available`：A `8.59%`、B `13.25%`、C `13.52%`。总体 `11.88%` 明显高于 both 当前 helpful `5.70%`，说明混合后的单一输出确实掩盖了一部分 remote 候选信息；但 Oracle 的绝对 headroom 只有约 1.37 AUC 点，应评价为中等而不是巨大。

Oracle-4 只比 Oracle-single 多 `0.0497` AUC 点，说明 `both` branch 对 Oracle 的独有补充很小；主要可用空间来自在 local 与两个 single sender branch 之间做正确选择。

## 10. Q6：下一步排序

按本轮数据排序：

1. **C. richer temporal reliability**：Oracle 有 headroom，但当前帧 score/APCE/entropy 与固定 Logistic Regression 都弱；应优先增加 prediction-only 的时间一致性、轨迹稳定性、target consistency 诊断。
2. **A. sender selection / reliability weighting**：这是最可能承接 Oracle-single headroom 的机制，但应建立在第 1 项找到更可靠特征之后；直接用现有特征实现 gate 风险很高。
3. **B. receiver gating**：可作为 selective report 的一部分，但当前 local-only 信号不足，不能先实现成 communication trigger。
4. **D. target semantic prompt**：当前 representation 已有约 +1.37 点 Oracle 空间，因此暂不需要立即放弃它；若 temporal reliability 仍失败，再转向更 target-aware 的表示。
5. **E. redesign fusion**：本轮禁止且证据优先指向选择/可靠性问题；在 selector 可预测性尚未厘清前重做 fusion 不可归因。

最终建议：下一轮仍先做只读/影子诊断，构造严格 prediction-only 的 temporal sender reliability；在 target-grouped OOF 上证明它能预测 sender-specific helpful 后，再实现默认关闭的 selective report。不要直接训练神经 gate，也不要用 GT oracle 做决策。

## 11. 产物索引

- 冻结预测与清单：`prediction_only_sender_counterfactual.csv`、`prediction_manifest.json`
- 后验标签：`posthoc_sender_labels.csv`
- 有向 sender 统计：`sender_helpfulness_summary.csv`
- 单变量 reliability：`sender_reliability_analysis.csv`
- 聚合交互：`aggregation_interaction_analysis.csv`
- Oracle：`oracle_headroom_summary.csv`、`oracle_per_view.csv`、`oracle_per_target.csv`
- target-grouped CV：`selector_group_cv.csv`
- 复现命令：`COMMANDS_ZH.md`
- 测试审计：`SMOKE_TEST_ZH.md`
- provenance：`provenance.json`
