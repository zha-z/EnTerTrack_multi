# E2B Target Consistency and Cross-view Compatibility Audit 预注册方案

状态：**FROZEN BEFORE E2B GT JOIN / OOF EVALUATION**  
冻结日期：2026-08-27  
分支：`feature/pcum-cross-layer-arp`  
源提交：`3abcb338e6f5b0bd7f6cc895be97fce2f95924ea`

## 1. 唯一研究问题与执行边界

本轮只回答：从冻结 Plain Collaboration V1 ep25 的 local branch 中提取的 compact target prototype，能否提供比 E2A 标量/时序可靠性更有效的 sender 语义一致性信号，并在严格 target-group OOF 下安全选择 Local、sender0-only 或 sender1-only。

本轮允许新增默认关闭、prediction-only、shadow-only 的 prototype 提取与日志；使用与 E1.5 完全相同的 checkpoint、Three-MDOT inner validation、V1 adapter、Safe Commit 和 sender counterfactual protocol 做一次 rollout。禁止训练或修改 backbone、CENTER head、V1 adapter；禁止修改 cross-attention、residual scale、persistent state、crop 或 reported bbox；禁止 runtime gate、神经 selector、target semantic prompt fusion、ARP/ATP/PCUM/C3R/FCVC 和 `threemdot_test`。

新增日志必须与 E1.5 的 Local、sender0-only、sender1-only、Both bbox 逐元素一致，且 persistent-state digest 不变。任一 mismatch 立即停止，不进入 GT join。

## 2. 冻结输入与 token layout

模型只使用 local branch 最终一层 `backbone_feat`。CENTER head 已将最后 `feat_len_s` 个 token 定义为 search token；当前冻结配置必须实测并断言：

```text
backbone_feat: [1, 320, 192]
template tokens: 前 64 个
search tokens X: 后 256 个，对应 16 x 16
CENTER raw score_map: [1, 1, 16, 16]
```

若实测 token 数、顺序或维度不满足该 contract，停止实验，不猜测或自动适配。本轮所谓 template-conditioned prototype 是“当前 local forward 中、位于 search token 之前的全部 template tokens 的均值”；不把它表述为固定初始模板。固定 initial-template anchor 需要额外 backbone forward 或状态，因此不在 E2B 实现。

初始帧没有现成 local backbone output。frame 0 只写 shape-stable 的 NaN placeholder 和初始化 bbox，不新增 forward；所有 prototype/consistency 分析从第一个活动帧开始。

## 3. 冻结 prototype definitions

对每个 `(target_id, view_id, frame_id)`，从 local branch 提取以下三个 192-D、detach 后的 float32 prototype：

1. response-weighted prototype（primary）：

```text
w = softmax(flatten(raw_score_map) / 1.0)
P_weighted = sum_i w_i * X_i
```

权重只能来自未乘 Hann window、未做 argmax、未使用 bbox/GT 的 CENTER raw `score_map`。温度固定 1.0。

2. global-mean prototype（control）：

```text
P_mean = mean_i X_i
```

3. template-conditioned prototype：

```text
P_template = mean_j T_j
```

同时记录三者 L2 norm、token counts、token dim、temperature、local candidate bbox、`uses_gt=false`、state digest before/after。prototype 不进入模型 forward、state、candidate 或 bbox 计算，不参与梯度。

## 4. 冻结 causal consistency definitions

每个 `(target_id, view_id)` 各自维护独立 history；target/view 间绝不共享。当前帧 `t` 的 history feature 只读 `<t` 的 prefix，先计算再更新 history。frame gap、重复或乱序立即报错。

对 `P_weighted` 与 `P_mean` 分别计算：

- `self_prev`：当前 prototype 与 `t-1` prototype 的 cosine；
- `self_ema`：当前 prototype 与仅由历史帧构成的 EMA 的 cosine，`beta=0.9`；
- `self_window`：当前 prototype 与之前最多 7 帧（`t-7 ... t-1`）prototype 均值的 cosine；冻结窗口记为 `W=8`（当前帧加最多 7 帧 causal context），prefix 不足时使用已有历史；
- `template_consistency`：当前 search prototype 与当前帧 `P_template` 的 cosine。

对同一 target、同一同步 frame 的 receiver/sender，分别计算：

- `cross_cosine_weighted`：receiver 与 sender 的 response-weighted prototype cosine；
- `cross_cosine_mean`：receiver 与 sender 的 mean prototype cosine。

不假定 cosine 越高越好；所有 cosine 均作为连续特征交给 fold-local 线性模型。

每条有向 receiver/sender row 固定包含：sender 的 prev/EMA/window/template consistency、receiver 的对应项、cross cosine、`sender_ema - receiver_ema`、`sender_template - receiver_template`、`cross * sender_template`。同样定义 weighted primary 和 mean control 两套特征。

## 5. prediction freeze 与因果隔离

full validation 先只生成 prediction-only artifact：每序列 NPZ prototype、合并后的 `prediction_only_target_prototypes.npz`、标量 feature CSV、manifest 和 SHA256。冻结前不读取 GT、不读取 E1.5 helpful/harmful label。

冻结检查必须包括：

- 5 targets、15 views、每序列 frame 行数与 bbox 输出一致；
- 活动帧 prototype shape `[192]`、finite，frame 0 为唯一允许的 placeholder；
- `uses_gt=false`；
- Local/sender0/sender1/Both bbox 与冻结 E1.5 artifacts 完全一致；
- state digest before/after 一致；
- 每个活动帧、每个视角仍恰好一次 local backbone forward。

只有 prediction artifact 和 manifest 的 SHA256 已写盘后，才允许 post-hoc join E1.5 冻结标签：`helpful` 当且仅当 `delta_iou > +0.02`，`harmful` 当且仅当 `< -0.02`，其余为 `tie`。GT 不进入 prototype、history、fold feature normalization 或 runtime/policy decision。

## 6. 固定 semantic ablation

E2B 同时报告 response-weighted 与 global-mean 两种 representation。每一种固定以下 nested feature groups：

- S0：E2A current scalar reference，sender/receiver score、APCE、entropy 及 current differences；
- S1：sender `self_prev`、`self_ema`、`template_consistency`；
- S2：`cross_cosine` only；
- S3：S1 + cross cosine；
- S4：S3 + receiver `self_prev`、`self_ema`、`template_consistency`；
- S5（primary）：S4 + sender/receiver `self_window` + `sender_ema - receiver_ema` + `sender_template - receiver_template` + `cross * sender_template`。

S0 作为 scalar reference，不与 S1-S5 自动拼接，除非组定义明确包含。主 representation 和主 policy 固定为 **response-weighted S5**；mean S5 是控制组。结果出来后不得改称其他组为 primary。

“response-weighted 明显优于 mean”的冻结定义为同时满足：

1. Task A weighted S5 ROC-AUC 至少比 mean S5 高 `0.03`；
2. weighted S5 selector tracking AUC 至少比 mean S5 高 `0.10` AUC point（fractional `0.0010`）。

## 7. 固定模型、LOTO 与任务

唯一主模型：

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

CV 固定 Leave-One-Target-Out：每 fold 用 4 targets fit imputer/scaler/model，完整第 5 target 只作 held-out prediction。禁止 random frame split、跨 target history、fold 外 normalization、feature search 和查看 held-out 后调阈值。

### Task A：helpful vs harmful

只使用 valid non-tie sender rows，helpful=1、harmful=0。报告 S0-S5 的 overall OOF ROC-AUC、PR-AUC、positive prior、固定阈值 0.5 的 precision/recall/confusion matrix 和按 receiver view 的描述性 breakdown。

### Task B：replace Local

每个 sender row：helpful=1，harmful/tie=0。它既是 branch-level replace-local classifier，也是 OOF tracking policy 的训练任务。报告 ROC-AUC、PR-AUC、positive prior 和 coverage。

### Task C：sender ranking

仅保留两个 sender 的 `delta_iou` 差绝对值 `>0.02` 的 valid receiver/frame；label 为 sender0 是否优于 sender1。主输入为 sender0 feature 减 sender1 feature，不含 view/pair identity；训练 fold 加入交换顺序的反对称样本。单独报告加入 receiver/sender view one-hot 的 identity control，但它不参与 primary policy。

## 8. 冻结 OOF Safe Report policy

每个 fold 用 Task B 的 weighted S5 模型对 held-out target 的两个 sender 分别输出 `p_replace_local`：

- frame 0、模型失败或 non-finite：Local；
- 两 sender probability 都 `<0.5`：Local；
- 唯一最大 probability `>=0.5`：对应 sender-only branch；
- 两者在 `1e-12` 内相等：Local。

候选只允许冻结 E1.5 Safe Commit rollout 中的 Local、sender0-only、sender1-only；primary 禁止 Both。selector 只改 reported bbox，persistent local state、下一帧 crop 和 sender packet 不变。

用 OSTrack `calc_seq_err_robust` 等价曲线报告 AUC、Precision、Normalized Precision、Mean IoU，并比较：Local、Both Safe、E2A T4、E2B mean S5、E2B weighted S5、Oracle-single。Oracle 只作 post-hoc upper bound。

## 9. 冻结成功、表示与安全门槛

- Primary selector：weighted S5 AUC `>= Local + 0.30` AUC point，即 fractional gain `>=0.0030`；
- Semantic separability：Task A weighted S5 ROC-AUC `>=0.65` 且 PR-AUC 高于 OOF positive prior；
- Representation：满足第 6 节 weighted-over-mean 双门槛；
- Safety：任一 held-out target 相对 Local AUC delta `<= -5.00` points 即 catastrophic regression，不能判成功。

Oracle utilization 固定为：

```text
(AUC_weighted_S5 - 0.648856) / (0.662076 - 0.648856)
```

其中 Local=`64.8856`，Oracle-single=`66.2076`；最终同时用 artifacts 中的精确值复核，若不一致则报告而不改 denominator。

## 10. 冻结 stop/go 映射

- Case A：Primary、Semantic、Safety 全通过 → 下一阶段才可提议默认关闭的 semantic selector；
- Case B：Semantic 通过且相对 E2A T4 Task A ROC-AUC 提升至少 `0.10`，但 Primary 未过 → 只研究 high-confidence sparse Safe Report，不做 dense runtime gate；
- Case C：Representation 门槛通过，但 Task A ROC-AUC `<0.65` → 下一步只提议 Target Semantic Prompt / compact K-token representation；
- Case D：Task A 仍近随机（ROC-AUC `<0.60`），且 Representation 未过 → 停止当前 feature engineering，转 E3 representation redesign。

若落在门槛间隙（例如 ROC `0.60–0.65` 且 representation 未过），结论固定为 inconclusive/stop，不把它包装为 Case A，也不根据结果临时放宽阈值。

## 11. 预注册输出

结果目录固定：

```text
docs/results/plain_collaboration_target_consistency_e2b_20260827/
```

至少包含：`REPORT_ZH.md`、`COMMANDS_ZH.md`、`PROTOCOL_ZH.md`、`prediction_only_target_consistency_features.csv`、`prediction_only_target_prototypes.npz`、`prediction_manifest.json`、`posthoc_target_consistency_labels.csv`、`prototype_definition.csv`、`self_consistency_analysis.csv`、`cross_view_compatibility_analysis.csv`、`semantic_ablation_group_cv.csv`、`sender_ranking_group_cv.csv`、`oof_policy_predictions.csv`、`oof_tracking_summary.csv`、`oof_per_view.csv`、`oof_per_target.csv`、`oracle_utilization.json`、`provenance.json`。

报告必须逐项回答任务要求 Q1-Q10，明确比较 E2A `64.8152`、Local `64.8856`、Oracle-single `66.2076`，并记录未运行训练和未访问 official test。
