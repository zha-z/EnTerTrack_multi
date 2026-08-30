# E3-D1-U1 冻结协议

## 1. 唯一问题与授权边界

使用唯一冻结的 E3-D1 epoch25 checkpoint，只在 `threemdot_val` 对 Local、sender0-only、sender1-only、Both 做与 E3-U1 同口径的 per-sender counterfactual audit，回答 D1 是增强 single-sender rescue，还是主要放大 residual/harm。

允许：最小泛化既有 E3-U1 runner/analyzer、单元测试、两帧 smoke、一次完整 `threemdot_val` prediction-only rollout、预测冻结后的离线 GT join 与描述性分析、写本目录产物。

禁止：训练、backward、optimizer、`threemdot_test`、official test、checkpoint/epoch/K/residual/cap/degradation/sender-weight sweep、任何 gate/selector/classifier、sampler/adapter/checkpoint 修改、receiver-visible 或 D2 实现。

## 2. 冻结输入

- branch：`feature/pcum-cross-layer-arp`
- source HEAD：`8d3fa0db6c8ad0fe83f447033df497fd0303cd25`
- dataset：`threemdot_val`，固定 5 target / 15 sequences
- tracker param：`target_prompt_collaboration_e3_d1`
- checkpoint：`output/diagnostics/target_prompt_collaboration_e3_d1/run_20260830_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3_d1/EnTeRTrack_ep0025.pth.tar`
- expected SHA256：`231e20df89c184dbe5411fa687b625b809d4c1580df64059ec9d50f3c62a1c2b`
- runid：`e3_d1_u1_val_20260830`
- E3-U1 reference：`docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829`

SHA256 不匹配立即停止。输出目录在开始检查时不存在；runner 必须拒绝非空目录。

## 3. 四分支与 Safe Commit

每个 target/receiver/frame 只运行一次 Local backbone；S0/S1/Both 共享完全相同的 pre-frame state、crop、local feature、CENTER local output 和 local prompt。

- Local：唯一 persistent state；
- S0/S1：复用标准 `remote_valid` mask，仅关闭另一个 sender；
- Both：直接调用标准 E3-D1/E3 `target_prompt_candidate()`；
- finalize：报告 Both、提交 Local；S0/S1 是 shadow branch，不得提交状态。

不修改 cross-attention、mean aggregation、normalization、K、residual scale 或 cap。

## 4. Smoke 停止门槛

先运行 1 target / 2 frames。以下均须 PASS，否则停止 full val：

1. D1 Local bbox 与冻结 E3-U1/B0-compatible Local 在相同 target/view/frame 精确一致；
2. finalize 返回的 Both bbox 与标准 Both candidate 精确一致；
3. `state_mutation=0`；
4. `local_state_mismatch=0`；
5. `uses_gt_rows=0`；
6. 每 active view/frame `local_forward_delta=1`。

## 5. Prediction-only schema 与冻结顺序

主 artifact 为 `prediction_only_d1_sender_counterfactual.csv`；可在不增加 backbone forward 的前提下同时保存 `prediction_only_d1_prompt_features.csv`。

主 artifact 至少记录 target/receiver/frame/branch、local/branch bbox、score、APCE、residual norm、relative residual norm、residual scale、valid remote count、used_remote、state/reported source、uses_gt。禁止出现 GT bbox、IoU、visibility、helpful/harmful 或 oracle。

冻结顺序：完整 rollout -> 原子写 prediction artifact 和 manifest -> 记录 SHA256 -> 独立进程复验 bytes/rows/hash -> 之后才加载 val GT。任一 hash 不一致立即停止。

## 6. Post-hoc 冻结定义

```text
delta_iou = IoU(sender/collaborative branch) - IoU(Local)
helpful = delta_iou > +0.02
harmful = delta_iou < -0.02
tie = otherwise
```

GT Oracle-single 在每个 visible frame 从 Local/S0/S1 选最大 IoU；Oracle-4 从 Local/S0/S1/Both 选最大 IoU。指标复用 E3-U1 的 OSTrack `_curves`：AUC、Precision、Normalized Precision、Mean IoU；所有 bbox 按原 E3-U1 的 integer writer 口径。

`remote_help_available`：至少一个 single sender `delta_iou>0.02`。百分比分母为 GT-visible valid receiver-frame；sender helpful/harmful/tie 的分母为 GT-visible valid directed sender rows。

## 7. Residual 描述性分析

- 分别对 S0、S1、Both 在各自 GT-visible 且 `frame_id>0` 的有效样本内部，以 `relative_residual_norm` 的 25/50/75 百分位切成 Q1–Q4；边界使用 NumPy linear quantile，等于边界进入较低 quartile。quantile 只用于 post-hoc 描述，禁止产生 runtime threshold。
- cap 值不手写推断：runner 从实际加载模型 `TargetPromptCollaboration.relative_norm_cap` 写入 manifest。cap hit 沿用仓库归档判断：`relative_residual_norm >= actual_cap - 1e-6`。
- Spearman：分别报告 pooled single-sender 与 Both 的 `relative_residual_norm` 对 `delta_iou`，使用 average ranks 后的 Pearson correlation；常量或少于 2 个有限样本记 NaN。

## 8. E3 直接对照与决策

冻结 E3 reference：Local 64.8856；S0 64.7323；S1 64.9073；Both 64.8705；Oracle-single 66.3173（+1.4317 vs Local）；Oracle-4 66.3704；remote-help 12.2327%；sender helpful/harmful/tie 7.23/8.17/84.60%。

为避免结果后解释漂移，本轮把 D1 Oracle-single 相对 E3 的变化按描述性 materiality `0.10 AUC point` 分为：

- `> +0.10`：明显增加；
- `[-0.10,+0.10]`：基本相当；
- `< -0.10`：明显下降。

Case A：Oracle 明显增加、remote-help/helpful 增加，但 Both 不改善且 harmful 增加，支持 D2 偏 rescue+preserve。

Case B：Oracle 基本相当，但 residual/harm/cap 增加，支持 D2 重点 Clean Receiver Preservation。

Case C：Oracle 明显下降，停止 D1，重新设计 objective/degradation source，禁止参数 sweep。

Case D：Oracle 与 Both inner-val 增加而冻结 official test 下降，只能提出 val/test domain mismatch 诊断；禁止再次访问 test。

本轮结论不授权 D2 实现。Visibility audit 继续约束：不修改 sampler；future natural receiver-visible 必须独立研究 receiver-specific loss mask 与 target balance。
