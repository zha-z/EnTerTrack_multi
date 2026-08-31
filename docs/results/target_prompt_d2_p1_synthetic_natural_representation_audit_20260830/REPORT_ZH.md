# D2-P1 Synthetic-vs-Natural Degradation Representation Audit

## 结论

本轮命中预注册 **Case C**。

- D1 的 target-box hard occlusion 对冻结 Local/B0 representation 造成强烈且稳定的变化：Train/VAL 均有 `6/7` 个 primary feature 达到 `|SMD(S,C)| >= 0.8`。
- 但 Synthetic 是否比 Clean 更接近 Natural 的证据跨 split 不稳定：Train 仅 `1/7`，VAL 为 `5/7` 个 primary feature 满足 `W1(S,N) < W1(C,N)`。
- 因而不能把 D1 failure 单独归因于 objective，也不能宣称 synthetic 已稳定模拟 natural occlusion。当前证据同时支持 source mismatch 与 objective risk，其中最直接、最强的 paired 证据首先指向 degradation source 过强。
- 下一阶段只改一个变量：优先 **A. 重新设计 degradation source**；保持 objective、B0 core、K=8 与其他设置不变。本轮不实现 D2、不训练、不调参。

## 范围与冻结性

| 项目 | 状态 |
|---|---|
| branch / source HEAD | `feature/pcum-cross-layer-arp` / `2abaeea8adaa6a0a2b4eaf93460c1e1780336ccf` |
| 数据 | Three-MDOT Train 与 VAL，分开统计；未读取 test |
| 模型 | B0 ep25 Plain ViT-Tiny + CENTER，K=8 prompt extractor |
| checkpoint SHA256 | `363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3` |
| local core identity | 170 个 core key，mismatch `0`，strict load PASS |
| adapter / collaboration | adapter forward `0`；未使用 remote/cross-attention/fusion |
| execution | `eval()` + `inference_mode()`；无 training/backward/optimizer/checkpoint update |
| GT 使用 | 仅 deterministic crop 与冻结 inventory；未传入模型 |
| spatial GT join | `MISSING_DIAGNOSTIC`：未做 score peak-to-GT post-hoc join，避免猜测坐标语义 |
| annotation 细分 | `MISSING_DIAGNOSTIC`：没有 partial/full occlusion 或 absence 字段 |

## 样本与充分性

样本 inventory 在任何 model forward 前冻结，之后没有按输出重新挑样本。C/S 是同 target、view、template、search frame、crop 与 bbox 的严格配对；S 唯一变化是在 normalized search crop 的目标框区域复用 D1 rasterization 并填 `0.0`。N 使用真实原图，不做人工修改。

| split | raw Natural | balanced C / S / N | balanced N targets | A / B / C | adequacy |
|---|---:|---:|---:|---:|---|
| Train | 4,593 | 2,341 / 2,341 / 2,341 | 17 | 652 / 1,011 / 678 | PASS |
| VAL | 599 | 599 / 599 / 599 | 5 | 468 / 41 / 90 | PASS |

Natural raw target concentration：Train top-1/top-3 为 `26.87% / 60.61%`；VAL 为 `53.76% / 92.82%`。VAL 虽通过预注册的每-view 最少 30 rows 门槛，但 target/view concentration 明显，跨 split 解释必须保守。

## Q1：Synthetic 对 Local representation 造成多大变化？

变化很大。严格 paired C→S 的均值如下：

| split | pairs | Δ max score | Δ APCE | Δ entropy | bbox center displacement | bbox log-scale L1 | symmetric prompt best-match cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,341 | -0.4824 | -134.36 | +0.3625 | 0.2377（60.84 px@256） | 0.4012 | 0.7646 |
| VAL | 599 | -0.4525 | -152.27 | +0.2971 | 0.2192（56.11 px@256） | 0.3699 | 0.7520 |

在 7 个 primary scalar 上，Train/VAL 均有 `6/7` 个 `|SMD(S,C)| >= 0.8`。唯一未达到大效应的是 `prompt_topk_score_mean`（Train `+0.125`，VAL `+0.310`）；其余主要体现为置信度/APCE/score gap/prompt norm 大幅下降，entropy 与 prompt token 相似度上升。

这证明 hard occlusion 对同一个冻结 Local/B0 的输入因果效应强，但不等于证明它模拟了真实遮挡。

## Q2：Synthetic 是否比 Clean 更接近 Natural？

不稳定，不能给出统一的“是”。

| split | `W1(S,N) < W1(C,N)` | 结论 |
|---|---:|---|
| Train | 1 / 7 | 只有 `prompt_topk_score_mean` 更近；其余 6 项 Synthetic 更远 |
| VAL | 5 / 7 | score/APCE/entropy/top-k mean/gap 更近；prompt norm 与 diversity 更远 |

Train 的关键 W1 对比：

| feature | W1(S,N) | W1(C,N) | 更近者 |
|---|---:|---:|---|
| max_score | 0.2964 | 0.1860 | Clean |
| APCE | 73.4245 | 60.9790 | Clean |
| score_entropy | 0.1938 | 0.1689 | Clean |
| prompt_topk_score_mean | 0.0192 | 0.0255 | Synthetic |
| prompt_top1_top8_gap | 0.3244 | 0.2239 | Clean |
| prompt_norm_mean | 0.1313 | 0.0525 | Clean |
| prompt_pairwise_cos_mean | 0.0799 | 0.0277 | Clean |

VAL 的 5/7 相对接近说明 Synthetic 不是完全无关的扰动，但 Train 的 1/7 与明显 target/view composition 差异使该结论无法跨 split 稳定成立。S-N/C-N 都不是严格因果比较，只能描述分布相似性。

## Q3：CENTER score / APCE / entropy 分布

以下均为 balanced primary subset 的 `mean ± population std`，括号内为 median：

| split | group | max_score | APCE | score_entropy |
|---|---|---:|---:|---:|
| Train | Clean | 0.8431 ± 0.1284（0.8773） | 222.99 ± 42.70（246.69） | 0.4155 ± 0.2078（0.4198） |
| Train | Synthetic | 0.3607 ± 0.1517（0.3574） | 88.63 ± 50.86（77.52） | 0.7780 ± 0.0863（0.7902） |
| Train | Natural | 0.6571 ± 0.2159（0.6780） | 162.01 ± 81.44（177.15） | 0.5843 ± 0.2098（0.6188） |
| VAL | Clean | 0.8001 ± 0.0934（0.8018） | 217.71 ± 31.94（224.40） | 0.5278 ± 0.1264（0.5646） |
| VAL | Synthetic | 0.3476 ± 0.1290（0.3345） | 65.43 ± 43.68（54.62） | 0.8249 ± 0.0651（0.8331） |
| VAL | Natural | 0.5642 ± 0.1782（0.5366） | 122.74 ± 74.53（100.54） | 0.7074 ± 0.1365（0.7287） |

两个 split 中 Synthetic 均落在“低 score、低 APCE、高 entropy”的极端侧，而 Natural 通常位于 C 与 S 之间。P10/P25/P75/P90 与 raw Natural 统计均保存在 `group_distribution_summary.csv`。

## Q4：K8 prompt 如何变化？

| split | group | top-k score mean | top1-top8 gap | prompt norm mean | pairwise cosine mean |
|---|---|---:|---:|---:|---:|
| Train | Clean | 0.1674 | 0.8190 | 12.5676 | 0.6652 |
| Train | Synthetic | 0.1741 | 0.2707 | 12.3839 | 0.7717 |
| Train | Natural | 0.1925 | 0.5951 | 12.5152 | 0.6922 |
| VAL | Clean | 0.1772 | 0.7735 | 12.5966 | 0.6548 |
| VAL | Synthetic | 0.1895 | 0.2339 | 12.4000 | 0.7790 |
| VAL | Natural | 0.2209 | 0.4692 | 12.5031 | 0.7078 |

Synthetic 的 top1-top8 gap 与 prompt norm 均显著降低，而 pairwise cosine 显著升高，即 K8 token 更同质、有效多样性下降。C-S symmetric best-match cosine 均值仅 Train `0.7646`、VAL `0.7520`，说明 prompt set 也发生实质迁移。Synthetic 到同-view balanced Natural prompt centroid 的最近 cosine 均值为 Train A/B/C `0.9331/0.9477/0.9467`，VAL `0.9488/0.9342/0.8641`；该指标只作描述，不用于筛样本或训练。

## Q5：Synthetic 是否产生 Natural 中不常见的异常 prompt？

有，尤其在 norm 与 token diversity；但不是所有指标、split 都一致。以同 split/view balanced Natural 的固定 P10-P90 为区间，跨 view 加权的区间外比例为：

| split | feature | Clean | Synthetic |
|---|---|---:|---:|
| Train | top-k score mean | 32.64% | 35.58% |
| Train | top1-top8 gap | 51.00% | 38.87% |
| Train | prompt norm mean | 17.98% | 41.22% |
| Train | pairwise cosine mean | 10.25% | 44.21% |
| VAL | top-k score mean | 41.40% | 44.24% |
| VAL | top1-top8 gap | 67.45% | 50.92% |
| VAL | prompt norm mean | 37.23% | 33.89% |
| VAL | pairwise cosine mean | 24.37% | 36.89% |

Synthetic 在 pairwise cosine 上两边都比 Clean 更常落到 Natural 中央 80% 之外；Train 的 norm 也明显异常。gap 指标则 Clean 更常出界，反映 Natural 的 gap 本身位于 C/S 之间。该统计不是 runtime threshold、selector 或 gate。

## Q6：D1 failure 更支持哪一种解释？

选择 **C. 两者都有**，对应预注册 Case C。

- source 侧：C-S 是本轮最强的 paired causal comparison；hard occlusion 在两个 split 都造成 6/7 大效应，且 Train 中 Synthetic 对 Natural 仅 1/7 更近。
- objective 侧：VAL 中 5/7 指标 Synthetic 更近，说明 source 并非在所有切分都完全脱离 Natural；而既有 D1 训练结果仍 harmful，因此 objective 仍是合理风险点。
- 限制：S 与 N 来自不同真实 frame、target difficulty 与时间阶段；VAL Natural 高度集中。不能据此证明某个 augmentation 导致 tracking failure。

## Q7：下一阶段优先项

选择 **A. 重新设计 degradation source**，只改这一项。

理由是 source 的 paired 证据直接且强，跨 split 的 natural similarity 又不稳定；在 source realism 尚未建立前同时加入 Preserve/Rescue loss 会混淆归因。建议下一步只预注册一个更自然的 partial degradation 或 direct natural-occlusion source，并保持 objective、B0 core、K=8、residual/cap 等全部冻结。当前证据不支持直接进行 target-balanced natural training，也不支持把 E3 degradation 路线永久停止；它只要求继续停止 D1 hard-occlusion 配方。

## Artifact 完整性

| artifact | rows / shape | SHA256 |
|---|---:|---|
| `sample_inventory.csv` | 11,072 | `ca20bc2b0482dde2088bff5e8a2ff2e2c545cceb4ac6302de3e6ad68fe17b911` |
| `representation_features.csv` | 11,072 | `f8c42d9ae52fb03a260292cfddfee00018b7986b9b3ee42afeb3e6c08cf36863` |
| `prompt_features.npz` | `[11072,8,192]`, FP16 | `ab667bf09b08208c33b41cd4d356e8dfc0c8ff0b09d55fcf4b4fd5b5a3d367ec` |
| `clean_synthetic_paired_analysis.csv` | 2,940 | `26b6ce681d2fcfcdc248924e947ca9c1c587df10e169e9ec7b3c1d6ec3a7faeb` |
| `group_distribution_summary.csv` | 136 | `e83ba71e17b71cc5787208c47d70d495207cca54a713e80bd705dac3df59a4ca` |
| `distribution_distance.csv` | 42 | `d78d2052d24639ab8ca1c3609774187fe4098a6e74ed80562bcc6793a5a1f921` |
| `per_target_summary.csv` | 69 | `2160f0f8ec1bcc8feab607357c6553af464b722a8b35609c42f9c7d91068f477` |
| `per_view_summary.csv` | 18 | `c7a1fe319cd7c5a3af91114ac9d25a7b1d2982217c6bafc63fadd3b242df6a0e` |

完整的执行命令见 `COMMANDS_ZH.md`，冻结定义见 `PROTOCOL_ZH.md`，机器可读来源见 `provenance.json` 与三个 manifest。
