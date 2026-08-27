# Plain Collaboration V1 Safe Commit 受控诊断报告

## 结论

本轮只使用冻结的 `threemdot_val`（5 个 target、15 个视角序列）和同一个 V1 epoch-25 checkpoint，没有训练、没有修改 adapter、没有访问 Three-MDOT official test。

结论是：**V1 在当前 inner-val 上的主要损失来自 collaborative bbox 被写入 persistent state 后造成的 closed-loop contamination，而不是当前帧 feature fusion 本身。**

| 路径 | AUC | Precision | Normalized Precision | Mean IoU |
| --- | ---: | ---: | ---: | ---: |
| D0-local | 64.8856 | 84.0712 | 82.8925 | 69.1493 |
| D0-closed-loop | 60.2624 | 77.7143 | 77.5403 | 64.1556 |
| D0-safe | **64.8943** | **84.1664** | **83.0164** | **69.1745** |

D0-safe 相比 closed-loop 恢复 **+4.6318 AUC 点**，相比 local 仅为 **+0.0086 点**，可视为基本持平。15 个序列的 paired bootstrap 中，safe-local AUC 差值 95% CI 为 `[-0.2865, +0.2662]` 点；不能把名义上的微小正差写成已证明的提升。closed-local 为 `-4.6232` 点，但其区间 `[-11.4807, +0.1273]` 受 15 个序列和两个严重漂移序列影响，仍需保守表述。

因此本轮名义上满足 Case C 的 `Safe > B0`，但差值太小，不应声称 collaboration 已提升；具有实际解释力的是 Case A：Safe Commit 基本恢复 B0，而 closed-loop 明显下降。

## D0 因果对比

- D0-local：同一个 V1 checkpoint，只加载 local backbone/CENTER 权重并忽略 9 个 adapter tensor；输出和 state 都是 local。
- D0-closed-loop：`report=collaborative`、`state=collaborative`，等价于修改前的 V1。
- D0-safe：`report=collaborative`、`state=local`；下一帧 crop 由 local state 产生，sender 仍来自同帧 local candidate。
- 三路使用相同 checkpoint、相同 15 个序列、相同初始化帧和 OSTrack evaluator。safe 与 local 每个后续 crop 的起点一致；safe 的 current-frame report 仍是真实 collaborative prediction。

视角结果显示 closed-loop 的损失主要集中在 B/C：

| View | local AUC | closed AUC | safe AUC | safe-local |
| --- | ---: | ---: | ---: | ---: |
| A | 62.7360 | 62.2378 | 62.5149 | -0.2210 |
| B | 61.3128 | 55.2374 | 61.4393 | +0.1265 |
| C | 70.6082 | 63.3122 | 70.7286 | +0.1204 |

按 target 看，closed-loop 在 `md3034` 和 `md3048` 分别比 local 低约 10.48、12.81 AUC 点；Safe Commit 将二者恢复到 local 附近。完整 per-view/per-target 数据见对应 CSV。

## Q1–Q7

### Q1：损失来自当前帧 fusion，还是 closed-loop contamination？

主要来自 **closed-loop contamination**。仅改变 state commit，AUC 从 60.2624 恢复到 64.8943；不改变当前帧 collaborative report。14,097 个 active frame 中，协作 forward 的 persistent-state digest 变化数为 0，Safe Commit 后 `state_output != local` 和 `reported_output != collaborative` 的计数也均为 0。

### Q2：Safe Commit 是否明显恢复性能？

是。相对 closed-loop 恢复 +4.6318 AUC 点，并回到 local 的 ±0.01 点范围。它不是一个已证明的新 AUC 增益，而是明确的漂移隔离结果。

### Q3：remote information 在什么条件下 helpful？

在 13,513 个可分析帧中，770 帧 helpful（5.70%）、937 帧 harmful（6.93%）、11,806 帧 tie（87.37%）。B 视角的 mean delta IoU 为 +0.00178，C 为 +0.00122；A 为 -0.00109。

单变量中，`local_collab_center_displacement` 对 helpful-vs-harmful 的 ROC-AUC 为 0.726、PR-AUC 为 0.608（正类基准约 0.451）。其最高四分位 helpful 比例为 13.23%，说明“大幅 collaborative 修正”有时对应真实收益；但该量是 fusion 后的 prediction-only disagreement，不代表某个 sender 已被证明可靠，也不能据此在本轮调阈值。

### Q4：remote information 在什么条件下 harmful？

A 视角 harmful 比例最高（9.66%），C 为 7.85%，B 为 3.56%。`relative_residual_norm` 最高四分位 harmful 比例为 9.41%，但其 helpful-vs-harmful ROC-AUC 仅 0.493，不能形成稳定判别规则。现有特征没有给出一个可审计、可直接部署的 harmful 条件；最重要的风险证据仍是错误 collaborative state 的时间累积。

### Q5：local score/APCE/entropy 能否判断本机需要协同？

当前不能。以 helpful 为正类，local score、APCE、entropy 的 ROC-AUC 分别为 0.533、0.539、0.463；反向构造的 receiver-need 指标也低于 0.5。local score 最低四分位 helpful/harmful 比例为 7.26%/8.24%，没有显示“本机低置信就更容易受益”的稳定关系。

### Q6：remote score/APCE/entropy 能否判断哪个 sender 可信？

当前不能。sender mean/max score 的 ROC-AUC 为 0.486/0.548，mean/max APCE 为 0.464/0.539，entropy reliability 为 0.478，均只是弱信号。本轮日志只有两 sender 的质量和最终等权融合结果，没有逐 sender counterfactual，因此不能把 aggregate 标签归因给某一个 sender。

### Q7：下一步优先方向

优先选择 **A. receiver-need gating**，但要限定为“prediction-only selective report/commit 的受控验证”，不是立刻训练复杂 gate。原因不是 local score/APCE 已经可用，而是 Safe Commit 已证明无条件 commit 是主要故障点，且 fusion 后 local–collaborative disagreement 是目前唯一有明显单变量区分力的信号。

排序为：A > B > C > D。

- B（sender reliability weighting）只有很弱的 aggregate 信号，且缺少 per-sender counterfactual，暂不能训练或调权。
- C（target semantic prompt）和 D（redesign fusion）不应优先，因为 Safe Commit 下当前 256-token fusion 已与 B0 持平，没有证据要求先更换 representation。
- 下一步应先冻结一个新的 inner-dev 协议，验证预注册的简单 prediction-only gate；official test 仍需单独授权。

## 完整性与泄漏审计

- prediction-only 文件先冻结为 14,112 行并计算 SHA256，之后才运行 GT join。
- 冻结 SHA256：`23f3ed852fe96a7cd3c05ab494f367a5d89512a84d4dbbab7d84d94e7ca9b7f9`。
- 14,097 个 active frame 全部为 256 search tokens、2 个 sender、`uses_gt=false`。
- 协作 forward persistent mutation：0。
- state/local mismatch：0；report/collaborative mismatch：0。
- D0-local 保存结果与 safe 日志中的 local candidate（整数保存口径）不一致数：0。
- Safe bbox 结果与日志中的 collaborative report 不一致数：0。
- 没有运行训练，没有读取或评测 official test，没有依据 view/target 调阈值。

详细数据见 `safe_commit_comparison.csv`、`counterfactual_frame_metrics.csv`、`reliability_feature_analysis.csv` 和 `safe_commit_integrity.json`。
