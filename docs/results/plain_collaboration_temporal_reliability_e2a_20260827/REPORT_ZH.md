# E2A Causal Temporal Sender Reliability Audit 报告

## 1. 结论先行

E2A 判定为 **Case C**：严格 causal 的 score/APCE/entropy、bbox motion 与 scale temporal feature 没有形成可泛化的 sender reliability。预注册 T4 OOF selector 不但未达到 `Local +0.30` AUC point，反而比 Local 低 `0.0705` AUC point。

本轮没有重跑 tracker、没有训练 neural gate、没有修改 V1 representation 或 Safe Commit，也没有访问 official test。唯一拟合的是 5-target Leave-One-Target-Out 中每折独立的 Logistic Regression diagnostic。

| 方法 | AUC | 相对 Local（AUC point） | 角色 |
| --- | ---: | ---: | --- |
| Local | 64.8856 | 0 | 冻结基线 |
| Both Safe | 64.8943 | +0.0086 | 冻结 0.5/0.5 report |
| fixed sender0 | 64.6706 | -0.2150 | 描述性对照 |
| fixed sender1 | 65.0461 | +0.1604 | 描述性对照，不可解释为通用 slot policy |
| OOF policy T0 | 64.8707 | -0.0149 | current sender only |
| OOF policy T1 | 64.8167 | -0.0689 | + sender temporal |
| OOF policy T2 | 64.8011 | -0.0846 | + receiver temporal |
| OOF policy T3 | 64.8221 | -0.0635 | + sender-receiver difference |
| **OOF policy T4** | **64.8152** | **-0.0705** | 预注册 primary |
| GT Oracle-single | 66.2076 | +1.3220 | GT upper bound only |

预注册 primary、secondary ROC、secondary PR 三项均 FAIL；Safety gate PASS。不得据此实现 runtime temporal selector，也不继续阈值或 feature search。

## 2. 数据、冻结与因果完整性

- 分支：`feature/pcum-cross-layer-arp`。
- E2A 开始 HEAD：`4d59a18cec1ce10c427d8d040d7082fe056748ab`。
- 预注册 plan 在结果产生前单独提交：`089b826`。
- 输入：E1.5 prediction-only 56,448 rows，SHA256 `75ebb3cb292202346619e6f81cc46d050fa394a73e7c0d01ad9376545ba95f43`。
- E2A prediction-only temporal artifact：28,224 sender rows、14,112 receiver/frame groups、30 isolated histories。
- E2A temporal SHA256：`717d41c95bd8276e254f2af0baabc8e9a17f2213fe65389e176b2be9993d9a24`。
- history key：`(target_id, receiver_view, sender_view)`；5 targets × 3 receivers × 2 senders = 30。
- 主窗口 W=8；仅生成预注册的 nested W=4 对照；没有 window search。
- prediction-only schema 中 GT/IoU/label/visibility 字段数量为 0，`uses_gt=true` 行数为 0。
- freeze 后才读取 `posthoc_sender_labels.csv`；join 前重新校验 E1.5 与 E2A 两层 SHA。
- 所有 imputation/scaling/model fitting 位于 fold pipeline，只读取 4 个训练 targets。
- 14,112 条 OOF policy decision 的 `uses_gt_at_decision=true` 为 0。
- 没有重新 rollout；candidate 均来自冻结 E1.5 Safe Commit，selector 只改变 reported bbox。

初始帧不参与 classifier fitting，固定报告 Local。active sender rows 为 26,996；helpful/harmful 非 tie rows 为 4,394，Task A coverage `16.2765%`。

## 3. 冻结 feature 可用性

E1.5 已足以离线重建：

- sender/receiver current score、APCE、entropy、bbox size、motion、scale；
- mean/std/slope/delta causal prefix；
- bbox-derived velocity、acceleration、constant-velocity residual、scale trend/variance；
- sender-receiver causal difference。

E1.5 没有 response map、image size/search-grid 元数据或 compact/dense target feature，因此以下字段没有伪造：

- response top1/top2 gap；
- response peak sharpness；
- bbox/search-region border proximity；
- target consistency / feature prototype similarity。

前三项在 `temporal_feature_definition.csv` 标记 unavailable；target consistency 固定留给 future E2B，没有实现 T5。

## 4. Q1–Q3：sender helpful vs harmful

### Q1：时序信息是否改善单帧 reliability 不足

没有形成稳定改善。

| Ablation | Feature count | ROC-AUC | PR-AUC | Precision@0.5 | Recall@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 current sender | 7 | 0.4281 | 0.4379 | 0.3690 | 0.4878 |
| T1 + sender temporal | 28 | 0.4359 | 0.4201 | 0.3948 | 0.4480 |
| T2 + receiver temporal | 56 | 0.4008 | 0.4233 | 0.3471 | 0.4577 |
| T3 + temporal difference | 69 | 0.4006 | 0.4236 | 0.3489 | 0.4571 |
| T4 + motion temporal | 93 | 0.4014 | 0.4290 | 0.3488 | 0.4551 |

Task A positive rate 为 `0.4461`。T1 相对 T0 只有 `+0.0078` ROC，且 PR 下降；加入 receiver、difference 和 motion 后 ROC 约 0.40。不能把 T1 的极小变化写成有效 temporal improvement，也不能在看到结果后反转 probability 或重新选 feature。

### Q2：哪个 temporal group 带来最稳定的 target-group OOF 增益

**没有任何一组。** T1 的 sender temporal 是唯一让 overall ROC 略升的增量，但它在 receiver A 上从 `0.3790` 降至 `0.3333`，不具视角稳定性；T2–T4 整体继续下降。

单变量中相对较高的是 receiver score std-8（ROC `0.5812`）、receiver APCE std-4（`0.5808`）和 std-8（`0.5795`），仍未到 0.65，且合并后没有跨 target 泛化。它们只说明 receiver response 波动值得未来观察，不足以成为 gate。

### Q3：能否预测某个 sender 是 helpful 还是 harmful

不能。Primary T4 ROC `0.4014`，PR `0.4290 < 0.4461` positive baseline。分视角 ROC 为 A `0.2190`、B `0.4233`、C `0.5135`，异质性强且没有全局可靠方向。

## 5. Q4：Remote Help Available

不能可靠预测当前 frame 是否至少存在一个 helpful remote。

| Ablation | ROC-AUC | PR-AUC | Positive rate |
| --- | ---: | ---: | ---: |
| T0 | 0.4441 | 0.1054 | 0.1190 |
| T1 | 0.4290 | 0.1027 | 0.1190 |
| T2 | 0.4194 | 0.1068 | 0.1190 |
| T3 | 0.4187 | 0.1089 | 0.1190 |
| T4 | 0.4196 | 0.1128 | 0.1190 |

T4 PR 仍低于正类率。该任务本身已经读取 remote sender feature，所以即使成功也只能属于 post-communication selection，不是 communication trigger。

## 6. Task C：sender ranking

对于两 sender delta IoU 差异 `>0.02` 的 1,874 帧：

| Variant | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| T0 no identity | 0.5979 | 0.3192 |
| T1 no identity | 0.5834 | 0.3089 |
| T2 no identity | 0.5834 | 0.3089 |
| T3 no identity | 0.5712 | 0.3014 |
| T4 no identity | 0.5625 | 0.2892 |
| T4 with pair identity | 0.5354 | 0.2833 |

current-frame pairwise difference 有弱 ranking signal，但 temporal feature 没有增强它；pair identity 反而下降。分视角差异很大（T4 no-identity：A `0.4602`、B `0.6797`、C `0.4743`），不能训练 view-specific policy，也不能把 B 的局部结果泛化。

## 7. Q5–Q7：OOF Safe Report policy

### Q5：能否可靠选择 Local / sender0 / sender1

不能。预注册 T4 policy 在 14,112 帧中选择 Local 7,969、sender0 3,386、sender1 2,757；remote report coverage 约 `43.53%`，但这些选择没有带来 AUC 收益。

policy 不使用 Both、不使用 pair identity；每折 branch-level helpful-vs-(harmful+tie) Logistic Regression 只从 4 个训练 targets 学习。两 sender 概率都低于 0.5、相等、非 finite 或初始化帧时均 fail closed 到 Local。

### Q6：OOF Selective AUC

T4 OOF Selective AUC 为 **64.8152**。相对 Local `64.8856` 是 **-0.0705 AUC point**。三视角均未提升：

| Receiver | Local | T4 selector | Delta（AUC point） |
| --- | ---: | ---: | ---: |
| A | 62.7360 | 62.6038 | -0.1321 |
| B | 61.3128 | 61.2960 | -0.0168 |
| C | 70.6082 | 70.5458 | -0.0625 |

T4 Precision 为 `84.2315`、Normalized Precision 为 `83.0055`，略高于 Local，但 AUC 和 Mean IoU 均下降，不能据此判 primary 成功。

### Q7：Oracle headroom utilization

```text
Local             = 64.8856
OOF selector      = 64.8152
GT Oracle-single  = 66.2076
selector gain     = -0.0705 AUC point
Oracle headroom   = +1.3220 AUC point
Oracle utilization = -5.33%
```

利用率为负，表示 selector 消耗而不是兑现 Oracle headroom。Oracle 使用 GT，只是 upper bound。

## 8. Q8：target safety

没有达到预注册 catastrophic threshold `<= -5.00 AUC point`，Safety gate PASS；但存在一个明确的 target 回退：

| Target | T4 - Local（AUC point） |
| --- | ---: |
| md3016 | **-0.4746** |
| md3027 | +0.0517 |
| md3034 | +0.0279 |
| md3048 | +0.0180 |
| md3055 | +0.0247 |

最大负差为 `-0.4746`，不是 catastrophic，但也说明总体负结果并非纯数值舍入。

## 9. 预注册 gates 与 Q9 决策

| Gate | 标准 | 结果 | 判定 |
| --- | --- | --- | --- |
| Primary | T4 selector >= Local +0.30 point | -0.0705 point | FAIL |
| Secondary ROC | T4 Task A ROC >=0.65 | 0.4014 | FAIL |
| Secondary PR | PR > positive baseline | 0.4290 < 0.4461 | FAIL |
| Safety | 无 target <= -5.00 point | 最差 -0.4746 | PASS |

Q9 下一步应选：**C. Target Consistency Reliability（E2B）**。

- 不选 A runtime temporal sender selector：primary/secondary 均失败。
- 不选 B high-confidence sparse selection：本轮不是“ROC 明显改善但效用不足”，而是 ROC 仍弱；继续阈值调参会违反冻结规则。
- 选择 C：需要 sender current target representation 与 initial template、previous reliable representation、causal prototype 的一致性，而不是继续堆 score/APCE/motion 统计。
- 暂不选 D Target Semantic Prompt：按协议，应先完成独立 E2B；只有 E2B 仍不可预测才转 representation redesign。

本任务到此停止，不自行实现 E2B。

## 10. 实现修正记录

最终结果前修复了两个可审计实现问题，均未改变预注册 feature/model/policy：

1. 首次 post-hoc join 没有把冻结 feature row 带入内部训练记录，sklearn 因全 NaN 输入报错；修正为内部 join feature，输出 label CSV 仍保持精简。
2. 首次 Oracle 重建用浮点 post-hoc delta 选 branch、再用整数 bbox 评测，与 E1.5 口径不一致；修正为直接在整数 candidate bbox 上按 GT IoU 选。修正后 Local/Both/fixed/Oracle 与 E1.5 全部精确一致。
3. per-view coverage 最初错误使用全局 denominator；只修正 descriptive coverage 分母，OOF probability 和 tracking metric 不变。

没有因观察到 held-out 结果修改 feature、window、threshold、classifier、policy 或 success criteria。
