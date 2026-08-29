# E3-U1 执行协议与因果边界

## 冻结顺序

1. 在分支 `feature/pcum-cross-layer-arp`、clean worktree 上核对 HEAD；
2. 完整阅读 E3/E1.5/E2A/E2B 冻结报告；
3. 验证 E3 ep25 checkpoint SHA256；
4. 在任何新 GT utility 结果前提交预注册 `63646b5`；
5. 实现独立 prediction-only audit runner 与 post-hoc analyzer；
6. 1 target / 2 frame smoke；
7. 完整运行 `threemdot_val`；
8. 原子写出 prediction CSV 与 manifest；
9. 独立重算 rows/SHA256；
10. SHA 匹配后才读取 val GT；
11. 固定阈值 GT join、Oracle、target-LOTO 和 OOF Safe Report；
12. 报告后停止。

## 四分支与 Safe Commit

每个 view/frame 先产生一次 Local candidate。`sender0_only`/`sender1_only` 保留两个 canonical packet 的 orchestration 形状，但将未使用 sender 的现有 `valid` mask 置 false；因此只复用 `TargetPromptCollaboration` 已有 valid-sender normalization，没有修改 adapter 数学结构。

所有 branch 复用同一 pre-frame state、crop、local feature 和 prompt。只有 `target_prompt_finalize_frame(local, both)` 在帧末提交状态；该函数报告 Both、提交 Local。single/local branch 全部是 shadow report，不能写 state。

## GT 边界

prediction runner 不读取 frame GT、visibility、IoU 或 oracle label。标准首帧 tracker 初始化仍使用数据集 init bbox；之后全部 crop/state/prompt 来自 Local tracker。prediction schema 显式拒绝 GT/IoU/label 列。

post-hoc analyzer首先重新验证两个 frozen artifact SHA256；不匹配即拒绝 GT join。GT 只用于 IoU、helpful/harmful/tie、Oracle 与训练折中的 utility label。OOF held-out probability 不使用 held-out target GT，policy state 永远 Local。

## 固定统计口径

- helpful：`delta_iou > +0.02`；
- harmful：`delta_iou < -0.02`；
- tie：其余；
- metrics：仓库 OSTrack `calc_seq_err_robust` 等价 curves；
- bbox：按正式 writer 的整数 bbox 口径评测；
- CV：5-fold Leave-One-Target-Out；
- imputer/scaler/classifier：fold-local；
- classifier：`LogisticRegression(max_iter=2000, class_weight=None, random_state=42)`；
- threshold：0.5；相等、缺失、non-finite、两 sender 都低于 0.5时 fail closed 到 Local；
- OOF candidates：仅 Local/sender0/sender1，不使用 Both。

## 完整性结果

- 47/47 E3-U1 + E3 + V1 tests PASS；
- 两帧 smoke runtime 与 post-hoc PASS；
- full val active receiver frames 14,097；
- local-forward mismatch=0；state mutation=0；Both mismatch=0；Local-state mismatch=0；runtime GT rows=0；
- prediction SHA 复验通过后才运行 GT join；
- 未运行 `threemdot_test`，未训练、未 backward、未修改 checkpoint。

