# E3-U1 Target Prompt Per-Sender Counterfactual and Utility Audit 预注册

## 1. 目的与执行边界

本审计只回答 E3 K=8 target prompt 在 `threemdot_val`（冻结 inner-dev）上的逐 sender utility、Oracle headroom 和 prediction-only 可预测性。它不训练模型，不修改 E3 adapter/backbone/CENTER head，不运行 `threemdot_test`，不做 checkpoint、K、temperature、residual scale、sender weight 或 view-specific policy sweep。

冻结 checkpoint：

```text
output/diagnostics/target_prompt_collaboration_e3/run_20260828_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3/EnTeRTrack_ep0025.pth.tar
SHA256=d1c92bfe710e99d68b5f4a0eb2c151222204ca289bb1392f567fe76718d70c78
```

运行时所有分支 strictly prediction-only。GT 只能在 prediction artifact 完成、manifest 写入且 SHA256 冻结后用于 post-hoc join、Oracle 和标签构造。

## 2. Safe Commit 与一次本地前向

每个同步帧先分别生成 Local A/B/C。每个 view 只运行一次 local crop、backbone 和 CENTER head，并缓存同一 pre-frame state 下的 local feature、256 search tokens、raw 16x16 score map 和 local K=8 prompt。所有反事实分支复用这些缓存，只重跑 E3 adapter 和 CENTER head。

所有分支固定：

```text
state_output = Local
next-frame crop/state = Local
sender prompt source = sender Local branch
reported_output = 当前被分析的 branch
```

协作分支不得修改 persistent state。每帧结束只提交 Local。禁止 Gated Commit。

## 3. 冻结四分支

对每个 receiver/frame 保存：

- `local`：不使用 remote；
- `sender0_only`：只使用 canonical sender0 prompt；
- `sender1_only`：只使用 canonical sender1 prompt；
- `both`：使用两个 sender，保持当前 E3 valid-sender mean aggregation。

canonical sender 顺序：A <- (B,C)，B <- (A,C)，C <- (A,B)。`sender0/1` 只是固定槽位，不代表可靠性优先级。

在相同 checkpoint、frame、pre-state 和 local feature 下，`both` bbox 必须与标准 E3 Safe Commit bbox逐元素/容差内一致；`local` 必须与同一冻结 rollout 的 local candidate 一致。若任一 identity 检查失败，停止 GT join。

## 4. Prediction-only prompt diagnostics

每个 sender 的 K=8 prompt 固定记录：

- top-k raw response：mean/std/min/max、top1-top8 gap；
- token L2 norm：mean/std；
- intra-prompt cosine（8 token L2 normalize，去 diagonal）：mean/std/min/max；
- sender score、APCE；
- sender/receiver view、frame、target、slot。

receiver-sender prompt-set compatibility 不做 pooling。对 L2-normalized `P_r @ P_s^T` 的 8x8 cosine matrix固定记录：

- `set_cos_mean`、`set_cos_max`；
- `sender_to_receiver_best_mean`；
- `receiver_to_sender_best_mean`；
- `symmetric_best_match`。

不做 Hungarian、optimal transport、learned matching 或 temperature search。

## 5. Branch diagnostics

`sender0_only`、`sender1_only`、`both` 固定记录 bbox、score、APCE、相对 Local 的 center displacement 与 log-area scale difference、residual norm、relative residual norm、residual scale、remote count。运行时 artifact 不含 GT、IoU、visibility、occlusion 或 oracle label。

## 6. Prediction freeze 与 GT join

第一阶段只产生：

- `prediction_only_e3_sender_counterfactual.csv`；
- `prediction_only_e3_prompt_features.csv`（若体积不可接受才改为 NPZ）。

要求 5 targets / 15 sequences 全部完成；每个有效 receiver/frame 有四 branch、两个 sender feature row。生成 `prediction_manifest.json`，记录路径、bytes、rows、columns、SHA256、checkpoint SHA256、dataset、targets、runid、`uses_gt=false`。冻结后重新校验 manifest，才允许 join `threemdot_val` GT。

post-hoc 对每个 receiver/frame 计算 Local、sender0、sender1、both IoU。sender label 固定：

```text
helpful: delta_iou > +0.02
harmful: delta_iou < -0.02
tie: otherwise
```

阈值不根据结果修改。

## 7. Oracle、sender 与 aggregation 分析

按 OSTrack `calc_seq_err_robust` 等价口径，固定报告 AUC、Precision、Normalized Precision、Mean IoU：

- Local；
- fixed sender0-only；
- fixed sender1-only；
- Both E3；
- GT Oracle-single = per-frame max-IoU(Local, sender0, sender1)；
- GT Oracle-4 = per-frame max-IoU(Local, sender0, sender1, both)。

报告 overall、per-view、per-target，以及 Oracle gain vs Local。`remote_help_available` 定义为至少一个 single sender `delta_iou > 0.02` 的有效 frame 比例。

六个有向 pair 分别统计 helpful/harmful/tie、mean delta IoU 和 fixed-pair AUC delta。仅描述，不据此硬编码 view weight。

aggregation interaction 使用 sender0/sender1 的 helpful/harmful/tie 九种组合，统计组合频次及 Both 的 helpful/harmful/tie。重点观察 good/bad 是否常见、Both 是否保留 harmful；不改变 mean aggregation。

## 8. 固定 feature groups 与 target-LOTO

只使用 Logistic Regression。5-fold Leave-One-Target-Out：每 fold 4 targets 训练、1 target held out。imputer、standard scaler 和 classifier 全部 fold-local；禁止 random frame split、feature search和阈值搜索。

- P0：sender score、sender APCE；
- P1：P0 + top-k score mean/std/min/max/top1-top8 gap；
- P2：P1 + prompt norm mean/std + intra-prompt cosine mean/std/min/max；
- P3：P2 + set cosine mean/max、双向 best mean、symmetric best match；
- P4：P3 + single-branch score/APCE、local-branch center displacement、scale difference、residual/relative residual/residual scale。

Task A：helpful vs harmful，仅 non-tie。Task B：helpful vs harmful+tie。每组报告 OOF ROC-AUC、PR-AUC、positive prior、precision、recall和覆盖行数。E2A/E2B 数值仅作为历史参考，不混合其训练 rows。

缺失或 non-finite feature 由 fold-local median imputer处理；训练折某列全缺失则以 0 填充并记录。classifier 固定 `LogisticRegression(max_iter=2000, class_weight=None, random_state=42)`；预测阈值固定 0.5。

## 9. OOF Safe Report policy

每个 held-out receiver/frame 只在 Local、sender0、sender1 中选择。若两个 sender 的 Task-B helpful probability 均低于 0.5、相等、non-finite或行缺失，则选 Local；否则选概率较高的 sender。policy 只改变 reported bbox，state 永远 Local，不使用 Both、不读取 GT、不使用 view identity。

primary OOF policy 使用 P4。报告 OOF AUC、Precision、Normalized Precision、Mean IoU、branch selection counts、per-view和per-target delta。

## 10. 预注册决策规则

- Case A：Oracle-single 明显高于 Local，P4 Task-A LOTO ROC >=0.65，且 P4 OOF policy >= Local +0.30 AUC point。下一步才允许 Selective Target Prompt Report，仍为 Safe Commit。
- Case B：Oracle-single 明显高于 Local，但 selector 不可泛化。停止 selector feature 堆叠，下一步优先 E3-D1 Asymmetric Degradation Training。
- Case C：Oracle-single 相对 Local仅有很小 headroom。下一步同样优先 E3-D1，而不是 gate/K tuning。
- Case D：single sender 明显优于 Both，且 helpful/harmful sender interaction频繁、Both 系统性削弱 good sender，才考虑 sender selection/aggregation redesign。

本审计不自行实现任何下一阶段。若多个 Case 条件部分重叠，优先按 Oracle headroom 判断 B/C；只有 aggregation interaction 对总体损失具有足够覆盖时才采用 D。否则保守停止。

## 11. 完整性与停止条件

在 GT join 前必须 PASS：checkpoint hash、15/15 sequences、每 active frame 四 branch、每 frame/view一次 local forward、Both 标准 E3 identity、Local identity、state mutation=0、sender prompt source local、`uses_gt=true` rows=0、prediction manifest hash复验。

任一失败则标记 `INVALID`，停止 Oracle、LOTO 和 policy。完成本轮报告后 STOP：不运行 official test、不训练、不修改 adapter或正式结果。
