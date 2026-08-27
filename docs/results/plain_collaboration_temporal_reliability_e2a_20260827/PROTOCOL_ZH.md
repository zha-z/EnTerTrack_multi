# E2A 协议与因果审计

完整预注册协议见：

```text
docs/plans/plain_collaboration_e2a_temporal_reliability.md
```

该 plan 在任何 E2A OOF 结果生成前单独提交为 `089b826`。

## 固定项

- 数据：仅 `threemdot_val`，5 targets，Leave-One-Target-Out。
- 输入：冻结 E1.5 prediction-only artifact；不重新运行 tracker。
- history：`(target_id, receiver_view, sender_view)`，严格 frame `<=t`。
- 窗口：W=8 primary，W=4 nested control；不搜索其他窗口。
- 模型：median imputer + StandardScaler + balanced LogisticRegression，`C=1`、liblinear、seed 42。
- ablation：T0–T4 固定逐级加入；无 T5。
- policy primary：T4 branch helpful-vs-nonhelpful；阈值 0.5；只选 Local/sender0/sender1；平局/非 finite/初始化均 Local。
- state：frozen Local；selector 只改变 reported bbox。
- primary：至少 `+0.30` AUC point。
- secondary：Task A ROC 至少 0.65，PR 高于 positive prior。
- catastrophic：任一 target `<= -5.00` AUC point。

## Freeze / GT join 顺序

1. E1.5 source SHA 校验。
2. 仅 prediction-only 字段构造 28,224 temporal rows。
3. 写 `temporal_prediction_only_features.csv`。
4. 计算并写 temporal SHA：`717d41c95bd8276e254f2af0baabc8e9a17f2213fe65389e176b2be9993d9a24`。
5. post-hoc 阶段重新验证该 SHA。
6. 验证 E1.5 post-hoc labels 所绑定的 source SHA。
7. 才允许 label join、LOTO fitting、Oracle 和 tracking evaluation。

## Prediction-only schema audit

- temporal rows：28,224。
- receiver/frame groups：14,112。
- histories：30。
- targets：`md3016/md3027/md3034/md3048/md3055`。
- `uses_gt=true`：0。
- GT/IoU/label/visibility columns：0。
- policy decisions：14,112，`uses_gt_at_decision=true`：0。

GT 只用于：4 个训练 targets 的 fold label、held-out evaluation metric 和 GT Oracle upper bound。held-out target GT 不参与模型、scaler、imputer、threshold 或 decision。

## Communication boundary

E2A 读取 remote sender current/temporal feature，属于 post-communication sender selection。它不是 communication trigger，也没有解决“是否发送/请求 remote”的问题。真正的 trigger 必须只使用 remote 到达前的 local causal history。

## 未实现字段

E1.5 freeze 不包含 response map、image/search grid metadata 或 dense target feature，因此 response gap、peak sharpness、border proximity 和 target consistency 均未伪造。Target consistency 明确留给 future E2B。
