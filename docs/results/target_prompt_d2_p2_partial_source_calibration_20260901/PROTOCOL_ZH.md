# D2-P2 Partial Occlusion Source Calibration 冻结协议

## 1. 唯一问题与禁止范围

本轮只回答：在保持 D2-P1 的 B0 ep25 Local core、CENTER、K=8 prompt、Clean/Natural inventory、crop/normalization 与 fill mechanism 不变时，降低单块 hard occlusion coverage 是否能让 synthetic weak-view representation 更稳定地接近 Three-MDOT Natural occlusion。

禁止 training、backward、optimizer、checkpoint update、tracking rollout、`threemdot_test`、official test、AUC selection、loss/sampler/model/adapter/K/residual/cap 修改、gate/selector/classifier/learned metric、追加候选，以及根据 VAL 换 candidate。

## 2. 冻结来源

- branch：`feature/pcum-cross-layer-arp`
- task-start HEAD：`8c978515803bdf3bfe69d0a83281688695acb3ea`
- D2-P1 inventory SHA256：`ca20bc2b0482dde2088bff5e8a2ff2e2c545cceb4ac6302de3e6ad68fe17b911`
- D2-P1 feature SHA256：`f8c42d9ae52fb03a260292cfddfee00018b7986b9b3ee42afeb3e6c08cf36863`
- D2-P1 prompt SHA256：`ab667bf09b08208c33b41cd4d356e8dfc0c8ff0b09d55fcf4b4fd5b5a3d367ec`
- B0 checkpoint：`output/diagnostics/b0_abc_plain/run_20260818_seed42_4gpu_r001/checkpoints/train/entertrack/b0_abc_plain_4gpu/EnTeRTrack_ep0025.pth.tar`
- checkpoint SHA256：`363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3`

Clean 与 balanced Natural 直接复用 D2-P1 frozen reference，不重新抽样。Train 每组 2,341；VAL 每组 599。模型仍通过 E3 config 构造 parameter-free `TargetPromptExtractor(K=8)`，所有 170 个 non-adapter core key 必须与 B0 完全一致，adapter forward 必须为 0。

## 3. 候选与空间定义

仅允许 `P25/P50/P75` 三个 selectable candidate；`P100` 仅为历史 reference，不可选回。所有候选使用 normalized fill `0.0`，只改变 coverage。

对 D2-P1 Clean sample 的 crop-normalized bbox，先直接调用 D1 `_normalized_box_to_pixels` 得到离散整框。orientation 固定为：

```text
digest = SHA256("D2-P2-orientation-v1" + NUL + d2_p1_clean_sample_id)
index = big_endian_uint64(digest[0:8]) mod 4
orientation = [left, right, top, bottom][index]
```

同一 Clean sample 的 P25/P50/P75/P100 共用 orientation。left/right 遮挡整框高度、对应 coverage 的边缘连续宽度；top/bottom 遮挡整框宽度、对应 coverage 的边缘连续高度。离散覆盖长度固定为 `ceil(axis_length * coverage)` 并 clip 到 `[1, axis_length]`，所以小框的 realized coverage 可略大于请求值，必须记录 min/mean/max。P100 无论 orientation 均覆盖完整 D1 pixel bbox。

helper 只位于 representation diagnostic 路径，不接入 actor、sampler、model 或 D1 文件。

## 4. P100 identity gate

每个运行 split 必须同时检查：

1. P100 input tensor 与 D1 全框 fill 逐像素完全相同；
2. 重新 forward 的 P100 `max_score/APCE/entropy` 与全部 K8 prompt statistics 相对 D2-P1 Synthetic 的 absolute difference `<=1e-6`；
3. 保存后的 K8 FP16 prompt 逐元素完全相同。

任一 mismatch 立即 STOP，不做 selection/holdout。

## 5. 冻结 representation

统一复用 D2-P1 deterministic GT-centered crop：template factor 2/size 128，search factor 4/size 256，ImageNet normalization，无随机 jitter/flip。模型运行 `eval()` + `inference_mode()`；GT 只用于 frozen crop，不传入模型。

Primary feature 严格为七项：

1. `max_score`
2. `apce`
3. `score_entropy`
4. `prompt_topk_score_mean`
5. `prompt_top1_top8_gap`
6. `prompt_norm_mean`
7. `prompt_pairwise_cos_mean`

Train artifact 包含 Clean、balanced Natural、P25/P50/P75/P100。Train candidate freeze 前不生成 VAL representation。VAL artifact 只允许包含 Clean、balanced Natural、唯一 frozen selected candidate 与 P100。

为保持与 D2-P1 `batch_size=64` 的 CUDA kernel shape 完全一致，末尾不足 64 的 batch 用最后一个 input 做 duplicate-only padding；forward 后立即丢弃 padding output，padding 不写入 artifact、不参与统计。此规则只消除 batch-shape 数值漂移，不改变任何真实样本输入。

## 6. Train-only selection

对每个 feature `j`：

```text
R_j(candidate) = W1(candidate_j, Natural_j)
                 / (W1(Clean_j, Natural_j) + 1e-12)
D_source(candidate) = arithmetic_mean_j(R_j(candidate))
```

只在 Train 从 P25/P50/P75 选 `D_source` 最小者；完全相同时选择 coverage 更低者。另报 `synthetic_closer_count` 与 `|SMD(candidate,Clean)|>=0.8` 的 `large_shift_count`。

Train PASS 必须同时满足：

- `synthetic_closer_count >= 5/7`；
- `D_source(selected) < D_source(P100)`；
- `large_shift_count(selected) < large_shift_count(P100)`；
- `mean(abs(SMD(selected,Clean))) < mean(abs(SMD(P100,Clean)))`。

后两项是对原规范“不能仍为 6/7 且整体 magnitude 与 P100 基本等价”的保守、无歧义冻结。若 Train FAIL，仍写 selected_source 记录最小 candidate 与失败门槛，但禁止生成 VAL representation。

## 7. Candidate freeze 与 VAL holdout

Train selection 后原子写 `selected_source.json` 和 SHA256 manifest，记录 candidate/coverage/orientation/fill、七项 W1、Train D_source/counts、预注册代码 commit 与 artifact SHA256。VAL runner 必须先验证该 SHA 及所有 Train artifact SHA。

VAL 只读 frozen selected candidate、Clean、Natural 和 P100。VAL PASS：

- 至少 `4/7` feature 满足 `W1(selected,Natural) < W1(Clean,Natural)`；
- `D_source(selected) < D_source(P100)`。

禁止用 VAL 调 candidate、coverage、orientation、阈值或 feature。

## 8. Paired 与 robustness

Train 对 P25/P50/P75/P100、VAL 对 selected/P100 报 C→candidate 的 score/APCE/entropy delta、bbox center displacement 与 K8 symmetric best-match cosine；均不参与 selection。

selected 在 Train/VAL 按 A/B/C 及 target 报七项 normalized distance。仅作 robustness warning 的“极端”固定为：

- 任一 view `D_source >= 2.0`；或
- 样本数至少 30 的任一 target `D_source >= 2.0`。

warning 不覆盖 aggregate gate，也不允许 view-specific severity。

## 9. 决策

- Case A：Train PASS + VAL PASS；允许建议未来 D2-S1 source-only training，但本轮不训练，D2-S1 必须另行预注册。
- Case B：Train PASS + VAL FAIL；禁止训练，下一阶段重新设计 fill/degradation mechanism。
- Case C：Train FAIL；STOP partial-hard route。
- Case D warning：aggregate PASS 但存在冻结 extreme robustness warning；是否允许未来训练仍以 aggregate gate 为主，报告风险。

S-N 不是严格因果比较；结果最多支持 frozen representation distribution calibration，不能证明 tracking gain。
