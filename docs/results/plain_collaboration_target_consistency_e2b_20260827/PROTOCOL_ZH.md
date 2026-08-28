# E2B 冻结协议摘要

唯一规范见 `docs/plans/plain_collaboration_e2b_target_consistency.md`，其预注册提交为 `f72d247`。

## Runtime shadow

- checkpoint、V1 adapter、Safe Commit 和 sender counterfactual protocol 与 E1.5 相同。
- 仅 `threemdot_val`；禁止 official test。
- 每活动帧只读取 local `backbone_feat=[1,320,192]` 和 CENTER raw `score_map=[1,1,16,16]`。
- search 为末 256 token；template-conditioned control 为前 64 token 均值。
- weighted prototype=`sum softmax(raw_score_map/1.0)*search_token`；mean control=`mean(search_token)`。
- detach、prediction-only、无额外 forward、无 state/crop/bbox write；frame 0 为 NaN placeholder。

## Causal feature

- history key 固定 `(target_id, view_id)`；当前帧只读 previous prefix。
- EMA beta=0.9；window 固定使用前最多 7 帧（W=8 contract）。
- weighted/mean 均计算 sender/receiver self-prev、self-EMA、self-window、template consistency、cross cosine 及冻结 difference/interaction。

## Freeze / post-hoc

1. 先对 E1.5 验证 Local/sender0/sender1/Both bbox 和 persistent state identity。
2. 写 prediction-only NPZ/CSV、SHA256 和 manifest。
3. SHA 验证后才 join E1.5 标签：helpful `delta_iou>0.02`、harmful `<-0.02`、其余 tie。
4. Logistic Regression：median imputer、standard scaler、C=1、balanced、liblinear、seed=42；所有 fit 均 fold-local。
5. CV：Leave-One-Target-Out，5 targets；禁止 random frame split。
6. primary policy：weighted S5，阈值 0.5，只选 Local/sender0/sender1，probability tie/nonfinite/frame0 回退 Local；只改 report。

## 冻结 gates

- selector 相对 Local `>=+0.30` AUC point；
- Task A ROC `>=0.65` 且 PR>prior；
- weighted 相对 mean 同时满足 ROC `>=+0.03`、tracking `>=+0.10` point；
- 任一 target `<=-5.00` point 为 catastrophic failure。

本轮结果 Case D；协议、阈值和 Case 映射均未在查看 held-out 结果后修改。
