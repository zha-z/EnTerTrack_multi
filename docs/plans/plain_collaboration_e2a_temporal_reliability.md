# E2A Causal Temporal Sender Reliability Audit 预注册方案

状态：**FROZEN BEFORE E2A OOF EVALUATION**  
冻结日期：2026-08-27  
分支：`feature/pcum-cross-layer-arp`  
源提交：`4d59a18cec1ce10c427d8d040d7082fe056748ab`

## 1. 唯一研究问题与边界

本轮只判断严格 causal 的 prediction-only temporal information 是否能：

1. 区分一个 remote sender 对当前 receiver 是 helpful 还是 harmful；
2. 判断当前 frame 是否至少存在一个 helpful remote；
3. 在 Local、sender0-only、sender1-only 中形成 target-group OOF 的 Safe Report 增益。

本轮不训练 tracker/backbone/CENTER/V1 adapter，不加载旧 TemporalGate/GRU checkpoint，不实现 runtime gate，不修改 residual scale 或 representation，不访问 `threemdot_test`。所有 selector 只改变 reported bbox，下一帧 state/crop 始终来自已经冻结的 local rollout。

## 2. 冻结输入与无需 rollout 的决定

输入为：

```text
docs/results/plain_collaboration_sender_counterfactual_20260827/
prediction_only_sender_counterfactual.csv
```

其冻结 SHA256 为：

```text
75ebb3cb292202346619e6f81cc46d050fa394a73e7c0d01ad9376545ba95f43
```

该 artifact 已包含 sender/receiver 的 score、APCE、entropy、bbox、center motion 和 scale change，足以完成 E2A 主分析。因此本轮不重新运行 tracker。

冻结数据不包含 response top1/top2、response peak sharpness、image/search border proximity 或 dense target feature。它们不通过新增 shadow rollout 补齐：response/border 项在 E2A feature definition 中标为 unavailable；target consistency 固定为 future E2B，不生成 T5。

## 3. Causal history contract

每条 history 唯一 key：

```text
(target_id, receiver_view, sender_view)
```

不同 target、receiver 或 sender 绝不共享 history。按 frame_id 升序逐条构造；frame `t` 的任何值只允许读取当前值及 `<=t` 的 prefix。frame gap 或 frame 非严格递增立即报错，不跨 gap 插值。

主窗口固定 `W=8`。根据任务预注册要求，同时生成窗口 4 的 nested summary，但不搜索其他窗口，也不根据结果选择 window。

所有缺失值填充、scaling 和模型拟合均位于 sklearn Pipeline 内，只使用当前 fold 的 4 个训练 targets。held-out target 的未来和总体统计不进入任何 feature normalization。

## 4. 冻结 feature definitions

### Group A：sender current

- `sender_score_t`、`sender_apce_t`、`sender_entropy_t`
- `sender_bbox_width_t`、`sender_bbox_height_t`
- `sender_center_motion_t`、`sender_scale_change_t`

### Group B：sender temporal

对 sender score/APCE/entropy 固定生成：

- `mean_4`、`mean_8`
- `std_4`、`std_8`（population std，prefix 长度 1 时为 0）
- `slope_4`、`slope_8`（窗口内以相对 frame index 做一元最小二乘；少于 2 个 finite 点为 NaN）
- `delta_1`（当前减上一帧；无前一帧为 NaN）

窗口不足时使用已有 causal prefix，不做未来 padding。

### Group C：可重建 motion temporal

从 bbox causal prefix 重建：

- normalized center velocity x/y/norm：当前与上一帧中心差除以上一帧 bbox diagonal；
- normalized center acceleration x/y/norm：当前 velocity 与上一 velocity 的差；
- normalized motion residual：当前中心与“上一中心 + 上一 velocity”常速度预测的距离，除以当前 bbox diagonal；
- log-area scale change；
- log-area slope 4/8；
- log-area variance 4/8。

sender 与 receiver 对称构造。response-shape 与 border 字段因冻结输入缺失而不加入模型。

### Group D：receiver temporal

- receiver score/APCE/entropy current；
- 与 Group B 对称的 mean/std/slope/delta；
- receiver bbox width/height、center motion、scale change current；
- Group C 的 receiver motion stability。

### Group E：sender-receiver temporal difference

- current score/APCE/entropy difference；
- score/APCE/entropy 的 mean_8 difference 与 slope_8 difference；
- normalized motion residual difference；
- velocity norm difference；
- acceleration norm difference；
- scale-change difference。

### Group F / T5

不实现。冻结 artifact 没有无歧义的 compact/dense target feature。记录为 future E2B Target Consistency Reliability。

## 5. 固定 ablation

- T0：Group A。
- T1：A + B。
- T2：A + B + D。
- T3：A + B + D + E。
- T4：A + B + D + E + C（sender 与 receiver motion）。

逐级报告 T0–T4，不按 held-out 指标选择主 feature set。**OOF tracking policy 主结果固定使用 T4。**

## 6. 固定模型与 CV

唯一主模型为 sklearn Logistic Regression：

```text
SimpleImputer(strategy="median")
StandardScaler()
LogisticRegression(
    C=1.0,
    class_weight="balanced",
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
```

CV 固定 Leave-One-Target-Out：5 folds，每次 4 targets train、1 target held out。禁止 random frame split、feature search、fold-specific feature selection 和查看 held-out 后调阈值。

## 7. Task A：helpful vs harmful

仅用 valid 且非 tie sender rows；helpful=1、harmful=0。对 T0–T4 生成 OOF probability。固定阈值 `0.5`，报告 overall ROC-AUC、PR-AUC、positive rate、全 active sender row 中的 label coverage、precision、recall、TN/FP/FN/TP，并用同一全局 OOF 模型预测按 receiver A/B/C 做 descriptive breakdown；不训练 view-specific model。

## 8. Task B：Remote Help Available

每个 receiver/frame 一条记录；正类为任一 single sender `delta_iou > 0.02`。输入通过对两个 sender 的同名 temporal feature做 permutation-invariant `min/max/mean/absdiff`，再加入 receiver feature，避免把 sender slot 当作固定 view prior。对 T0–T4 做同一 LOTO 和固定阈值 0.5。

该任务使用 remote 信息，属于 post-communication 判断，不是 communication trigger。

## 9. Task C：Sender Ranking

仅保留两个 sender 的 GT delta IoU 绝对差 `>0.02` 的 valid frame；label=sender0 优于 sender1。主 variant 不使用 pair identity，输入为 `sender0 feature - sender1 feature`，并对训练 fold加入交换顺序的反对称样本，防止 slot bias。

单独报告 `with_pair_identity` diagnostic：在上述差分外加入 receiver/sender view one-hot，同样只在训练 fold fit encoder，并做交换增强。该 variant 不用于 OOF tracking policy；即使提高，也明确标记 5-target pair-prior 过拟合风险。

## 10. 冻结 OOF Safe Report policy

Policy 使用一个与 Task A 分开的 branch-level T4 Logistic Regression：

- 每个 valid 训练 sender row：`helpful=1`；harmful/tie=0；
- 模型、imputer、scaler 和 class weight 与第 6 节完全相同；
- 对 held-out target 的两个 sender 分别输出 `p_helpful`；
- local 的固定决策基准为 `0.5`；
- 若 `max(p0,p1) < 0.5`，选择 Local；
- 若唯一最大值 `>=0.5`，选择该 sender；
- 若两 sender probability 在 `1e-12` 内相等，选择 Local；
- frame 0 固定选择 Local；模型失败或 probability 非 finite 时 fail closed 到 Local。

Policy 不使用 Both，不使用 pair identity，不使用 GT、不使用 future，不按 target/view 设置规则。branch bbox 按与冻结 E1.5 相同方式转为整数后，用 OSTrack `calc_seq_err_robust` 等价曲线计算 AUC、Precision、Normalized Precision 和 Mean IoU。

该 selector 只改变 reported bbox。由于所有 candidate 来自同一个冻结 Safe Commit rollout，选择结果不会影响任何下一帧 local state。

除 T4 primary policy 外，可以按完全相同规则输出 T0–T3 的 descriptive OOF tracking ablation，但不得在结果出来后把最好的一组改称 primary。

## 11. Success / safety gates

在任何 E2A OOF 结果生成前冻结：

- Primary：T4 OOF Temporal Selector AUC `>= Local + 0.30` AUC point，即 fractional AUC gain `>=0.0030`。
- Secondary：Task A T4 OOF ROC-AUC `>=0.65`，且 PR-AUC 高于该任务 OOF positive-class prior。
- Safety：报告每个 target 的 AUC delta；若任一 target 相对 Local `<= -5.00` AUC points，判为 catastrophic regression，不能判稳定成功。

Oracle utilization 固定为：

```text
(AUC_T4_selector - AUC_local) /
(AUC_oracle_single - AUC_local)
```

GT Oracle 仅 post-hoc upper bound，不能参与 feature、threshold、fold model 或 runtime decision。

## 12. 冻结决策映射

- Case A：Primary、Secondary、Safety 全通过 → 下一步才可提议默认关闭的 runtime temporal sender selector + Safe Report。
- Case B：Task A ROC 明显提高但 Primary 未过 → 只研究 expected utility / high-confidence sparse selection，不上 GRU。
- Case C：temporal ROC 仍约 0.5–0.6 且 selective AUC 未超过 Local → 下一步 E2B Target Consistency Reliability。
- Case D：只有 E2B 仍失败后才转 Target Semantic Prompt。

本 plan 冻结后，不因本轮 held-out target 结果修改 feature、threshold、policy 或成功门槛。
