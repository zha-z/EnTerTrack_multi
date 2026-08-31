# D2-P1 Synthetic-vs-Natural Degradation Representation Audit 冻结协议

## 1. 唯一问题与边界

本轮只回答：D1 的 synthetic target-box hard occlusion 在同一个冻结 Local/B0 representation 中，是否比 Clean 更接近 Three-MDOT 的 natural occlusion。允许 Train 与 VAL annotation 构造、冻结样本 inventory、冻结 Local forward、K=8 prompt 与描述性 post-hoc 分析。

禁止训练、实验性 backward/optimizer、checkpoint 更新、`threemdot_test`、official test、D2/sampler/actor/model 修改、K/degradation/residual/cap sweep、classifier/selector/gate/learned metric、collaboration/cross-attention/fusion，以及 GT runtime decision。

## 2. 冻结仓库与模型

- branch：`feature/pcum-cross-layer-arp`
- source HEAD：`2abaeea8adaa6a0a2b4eaf93460c1e1780336ccf`
- primary core：B0 ep25 Plain ViT-Tiny + CENTER
- checkpoint：`output/diagnostics/b0_abc_plain/run_20260818_seed42_4gpu_r001/checkpoints/train/entertrack/b0_abc_plain_4gpu/EnTeRTrack_ep0025.pth.tar`
- SHA256：`363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3`
- config contract：`target_prompt_collaboration_e3` 只用于构建 parameter-free `TargetPromptExtractor(K=8)`；local forward 不调用 adapter、remote prompt 或 collaboration。

模型以 E3 的 strict B0 initialization loader 加载上述原始 B0 checkpoint。必须逐 key 验证所有 non-adapter core tensor 与 checkpoint 完全相同；adapter output 不进入任何 primary representation。

## 3. Annotation 与样本定义

只读取 `THREEMDOT train` 与 `THREEMDOT_VAL`，分开统计。对每个 synchronized target/view/frame：

```text
bbox_valid = finite(x,y,w,h) AND w>0 AND h>0
visible = occlusion==0 AND out_of_view==0 AND bbox_valid
Natural N = occlusion==1 AND out_of_view==0 AND bbox_valid
common_visible = visible_A AND visible_B AND visible_C
```

annotation 不存在 partial/full occlusion 或 absence，报告为 `MISSING_DIAGNOSTIC`，不自行细分。

### C/S paired candidate

Search 必须是 `common_visible` 且 `frame_id>0`。Template 是此前 200 帧内最近的 `common_visible` frame。Clean C 与 Synthetic S 共享完全相同的 split/target/view/template/search、原始图像、crop、normalization 和 bbox，只有 S 在 normalized search crop 中执行 D1 hard occlusion。

### Natural candidate

Search 满足 Natural N。Template 是同一 view 此前 200 帧内最近的 `visible` frame；Natural 图像不做人工修改。没有合格 causal template 的 search frame排除并计数。

这是一份 deterministic inventory，不调用随机 sampler，也不做无限重采样。

## 4. Raw 与 balanced inventory

保留所有满足定义且有 causal template 的 Natural rows，形成 raw Natural inventory。Primary balanced subset 按 `split + target_id + view` 分层：

```text
q = min(clean_candidate_count, natural_candidate_count)
```

在各自按 search frame 排序的候选中，以固定 midpoint systematic index `floor((i+0.5)*n/q)` 选 q 个 Natural 和 q 个 Clean；每个 Clean 复制一条严格 paired Synthetic。因此每个可支持 stratum 的 C/S/N 数完全相等。若某 stratum 一侧不足则只取实际 `q`，不从别的 target/view 强行补齐；`q=0` 时明确记录。

任何 model forward 前写 `sample_inventory.csv`，随后以 bytes SHA256 写 `sample_inventory_manifest.json`。模型结果不得反向改变 inventory。

## 5. Deterministic crop 与 D1 exact transform

三组统一复用仓库 `lib/train/data/processing_utils.py`：

- template：GT-centered `sample_target`，factor `2.0`，size `128`；
- search：GT-centered `sample_target`，factor `4.0`，size `256`；
- bbox crop mapping：`transform_image_to_crop(..., normalize=True)`；
- RGB tensor：`uint8/255` 后 ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`；
- 不做 center/scale jitter、brightness jitter、grayscale 或 flip，避免随机 augmentation 成为混杂变量。

Synthetic 直接复用 `lib/train/target_prompt_asymmetric_degradation.py::_normalized_box_to_pixels` 的 floor/ceil/clipping 逻辑，并使用冻结 `fill_value_normalized=0.0`。不另写近似 rasterization，不改变 template 或 annotation。

## 6. Frozen representation artifact

所有样本执行完全相同的 `model.eval()`、`torch.inference_mode()` Local forward。保存：

- CENTER：`max_score`、APCE、`score_map_max/mean/std`、top1-top2 gap、entropy、normalized predicted bbox；
- K8：top-k score mean/std/min/max、top1-top8 gap、prompt norm mean/std、prompt pairwise cosine mean/std/min/max；
- compact prompt：`prompt_features.npz`，`sample_id` 与 `[N,8,192]` FP16 prompt；统计在保存前用 FP32 计算。

本轮 score map 是冻结 CENTER head 输出的 raw sigmoid `16x16` map，不乘 tracker Hann window。`max_score=max(score_map)`；APCE 固定为：

```text
(max-min)^2 / (mean((score-min)^2) + 1e-8)
```

entropy 固定为把 256 个非负 score 归一化为概率 `p_i=score_i/sum(score)` 后的 normalized Shannon entropy：

```text
-sum(p_i*log(p_i+1e-12)) / log(256)
```

若 score sum 非有限或 `<=1e-12` 则 entropy 为 NaN。`pred_bbox_xywh` 直接保存 CENTER crop-normalized输出；不做 tracker closed-loop 或原图 map-back。

先原子写 `representation_features.csv`、`prompt_features.npz` 与 `representation_manifest.json`，独立复验 rows/sample order/shape/dtype/SHA256 后才做分析。

## 7. 冻结分析定义

### C-S paired

对每个 pair 计算 S-C 的 max score/APCE/entropy delta、bbox center Euclidean displacement（normalized 与 256-pixel）、log-scale L1，以及 prompt：

- C-to-S token nearest-neighbor cosine mean；
- S-to-C token nearest-neighbor cosine mean；
- symmetric best-match cosine（上述两者均值）；
- 全 `8x8` cross-token cosine mean（control）；
- centroid cosine（descriptive control）。

禁止 Hungarian、optimal transport 或 learned matching。

### Distribution

固定 primary scalar feature：

1. `max_score`
2. `apce`
3. `score_entropy`
4. `prompt_topk_score_mean`
5. `prompt_top1_top8_gap`
6. `prompt_norm_mean`
7. `prompt_pairwise_cos_mean`

对 balanced C/S/N 和 raw Natural 分别报告 count、mean、std、median、P10/P25/P75/P90。Balanced primary 对 C-S、S-N、C-N 报 Wasserstein-1、KS statistic 与 pooled-std standardized mean difference；pooled std `<1e-12` 时 SMD 为 NaN。

Q5 的 abnormal prompt 只对四个 prompt scalar（top-k mean、top1-top8 gap、norm mean、pairwise cosine mean）描述：以同 split/view balanced Natural 的 P10/P90 为固定区间，分别统计 C/S 落在区间外的比例。它不是 runtime threshold 或选择器。

Synthetic-to-Natural prompt nearest similarity 只比较同 split/view 的 prompt centroid cosine，取每个 Synthetic 对 balanced Natural 的最大 cosine，做描述性 summary；不用于配对优化、训练或筛样本。

## 8. 预注册判例

先检查 Natural reference adequacy：每个 split 的 balanced N 至少覆盖 3 个 target，且 A/B/C 每个 view 至少 30 rows；任一失败直接 **Case D**。

其余判例只看上述 7 个 primary feature，Train/VAL 分开：

- `large_shift_count`：`|SMD(S,C)| >= 0.8` 的 feature 数；0.8 是预注册的 conventional large standardized effect；
- `synthetic_closer_count`：严格满足 `W1(S,N) < W1(C,N)` 的 feature 数；不另设事后绝对相似阈值。

判定顺序：

1. **Case A**：Train 与 VAL 均 `large_shift_count>=4`，且均 `synthetic_closer_count<=3`；source 存在强 domain gap，优先重新设计 degradation source。
2. **Case B**：未命中 A，且 Train 与 VAL 均 `synthetic_closer_count>=4`；Synthetic 在多数核心指标相对更接近 Natural，D1 failure 更支持 objective 问题，优先 Rescue/Preserve loss design。
3. **Case C**：adequacy PASS 但 A/B 均未命中；source 与 objective 都可能有问题，下一阶段只能先改一个变量。
4. **Case D**：adequacy FAIL；Natural reference 支持不足，优先 target-balanced protocol/额外 train-side study。

`top-1/top-3 target share`、prompt anomaly 与图形只作解释，不覆盖上述顺序。S-N 不是 causal comparison；即使 Case A/B/C 成立，也只能支持 representation distribution mismatch/similarity，不能直接证明 augmentation 导致 tracking failure。
