# Plain ViT-Tiny 跨视角协作最小验证方案

## 文档目的与边界

本文基于当前已验证的 `B0-ABC-Plain`，只回答一个问题：在不改变 Plain ViT-Tiny、CENTER head、采样器和 tracking loss 的前提下，来自同一目标其他无人机视角的信息，是否能提升单视角跟踪的鲁棒性与 AUC。

本文是静态代码与实验设计分析，不包含代码实现、训练配置或新实验结果。当前阶段继续关闭 ARP、ATP、token pruning、token compensation、PCUM full system、C3R full system、FCVC full system。

本文采用的基线事实是：

- `B0-ABC-Plain`：Plain ViT-Tiny，A/B/C flatten training，common-visible sampler，无跨视角协作。
- 当前 inner-val 结果约为 AUC `64.89`；该数值不能与项目文档中采用不同 checkpoint、split 或 official test protocol 的历史结果直接混用。
- `B1-ABC-ARP` 明显弱于 Plain，因此当前设计不再以 ARP/ATP/pruning 为研究变量。

---

# 1. 当前 multiview 数据流程分析

## 1.1 Three-MDOT 的样本标识

Three-MDOT 的训练序列按 `target-view` 组织。当前数据代码从序列名末尾解析视角：

- `1` 对应 view A；
- `2` 对应 view B；
- `3` 对应 view C；
- 去掉视角后缀后得到共同的 `target_id`。

数据集内部使用从 0 开始的 `frame_id`，读图时转换为磁盘上的 `{frame_id + 1:08}.jpg`。每个序列还读取 bbox、occlusion 和 out-of-view 标注；有效可见性由 bbox 合法、未遮挡且未出视野共同决定。

相关实现：

- `lib/train/dataset/threemdot.py`：split、序列路径、frame 读取和 visibility。
- `lib/train/data/sampler_threemdot.py`：`target_id`/`view_id` 解析以及 A/B/C 配对。

## 1.2 A/B/C sample 的构造

`TrackingSamplerThreeMDOT` 的一个逻辑样本不是随机抽三个无关序列，而是：

1. 先选一个 primary sequence；
2. 从同一 `target_id` 找到其余两个视角；
3. 按 A、B、C canonical order 排列；
4. 在 `REQUIRE_ALL_VIEWS_VISIBLE=true` 时，取三个视角 visibility 的交集；
5. 从共同可见帧中采样 template/search frame，并维持因果顺序，即 search 晚于 template；
6. 用相同的 template/search `frame_id` 分别读取 A/B/C 图像与 bbox。

因此，在 common-visible 模式下，一个逻辑样本具有：

```text
target_id = T
view_ids = [A, B, C]
template_frame_ids = [t, t, t]
search_frame_ids   = [s, s, s]
s > t
```

采样器输出中保留 `target_id`、`view_ids`、`template_frame_ids`、`search_frame_ids` 和各视角 visibility 计数。这些字段足以在后续协作时验证 remote 是否来自同一目标、同一时刻和正确 sender view。

## 1.3 template/search 的生成

`STARKProcessing` 对每个视角独立执行 bbox jitter、crop、resize 和 transform，生成 template/search image、attention mask 和归一化 bbox。batch collate 使用 view-major 维度，actor 接收到的主要张量形态为：

```text
template_images: [V, B, Nt, C, Ht, Wt]
search_images:   [V, B, Ns, C, Hs, Ws]
V = 3
```

当前 B0 通常每个样本使用一张 template 和一张 search，因此忽略 `Nt/Ns=1` 后可理解为 `[3, B, C, H, W]`。

训练增强仍按现有 pipeline 执行。三个视角共享 target/frame 配对关系，但各自的图像 crop 和特征计算仍是独立的。

## 1.4 flatten 的实际含义

`EnterTrackActorThreeMDOT` 有两个不同路径：

- 普通 `forward_pass()` 会通过 `_select_first_view(...)` 选择 multiview tensor 的第一个视角；如果直接走这条路径，确实只会使用 A。
- B0 的 `FLAT_MULTIVIEW_BASELINE=true` 会进入 `_forward_flat_views(...)`，把 view-major 的 `[V, B, ...]` 变成 `[V*B, ...]`，再进行一次普通网络前向和统一 loss/backward。

flat index 的对应关系为：

```text
flat_index = view_index * B + batch_index

[A0, A1, ..., A(B-1),
 B0, B1, ..., B(B-1),
 C0, C1, ..., C(B-1)]
```

因此 B0 的 A、B、C 都会参与同一次 optimizer step 和反向传播；它不是 `A=100%, B=0%, C=0%` 的 first-view 训练。

## 1.5 为什么这仍不属于“协同”

虽然模型在训练时同时看到 A/B/C，但它看到的是 `3B` 个互相独立的 tracking samples：

- 每个 flat sample 单独经过 patch embedding、Transformer 和 CENTER head；
- Transformer attention 只发生在单个 template/search token 序列内部；
- A 的 forward 不读取 B/C 的 token、feature、score 或 bbox；
- B/C 只通过共享参数和 batch loss 间接影响权重；
- 推理时单个 tracker 仍可完全独立运行。

所以当前 B0 证明的是“多视角数据共同训练对共享单机模型有效”，而不是“在线跨视角信息传递有效”。协作实验必须建立显式的 sender-to-receiver 数据路径，且该路径关闭时恢复 B0。

---

# 2. 当前模型 forward 分析

## 2.1 Plain ViT-Tiny forward

当前 B0 的有效网络路径为：

```text
template 128×128 ─┐
                  ├─ patch16 embedding + position embedding
search   256×256 ─┘
                  ↓
        concatenate template/search tokens
                  ↓
       Plain ViT-Tiny Transformer × 6
       dim=192, heads=3, no token pruning
                  ↓
        final normalized token sequence
                  ↓
       extract final 16×16 search tokens
          [B, 256, 192]
                  ↓
       reshape to [B, 192, 16, 16]
                  ↓
              CENTER head
                  ↓
        score map / size map / offset map
                  ↓
                 bbox
```

`base_backbone.py` 对 template/search 分别做 patch embedding 后拼接，并让全部 token 经过全部 Transformer block。`entertrack.py::forward_head()` 从最终 token 序列末尾切出完整的 256 个 search token，再送入冻结接口不变的 CENTER head。

## 2.2 四种协作插入位置

### 方案 A：输入 token 级融合

在第一个 Transformer block 前，将 remote token 与本地 template/search token 拼接或注入。

优点：

- remote 信息可影响全部 6 个 block；
- 理论上可学习较深的跨视角联合表征。

缺点：

- 改变 backbone 输入长度、位置编码语义和计算量；
- 跨相机的相同 token index 不代表相同空间位置，直接相加没有几何依据；
- remote 噪声会从最早层污染整个本地路径；
- 很难做到关闭时与 B0 执行路径完全一致；
- 同时引入 token 组织、位置编码、融合层和训练稳定性等多个变量。

结论：不适合作为最小 V1。

### 方案 B：Transformer 中间层融合

在一个或多个 Transformer block 后注入 remote feature，再继续执行后续 blocks。

优点：

- 中间 feature 比原始 patch token 更具语义；
- 后续 block 能进一步消化协作信息；
- 若未来验证成功，可研究 layer choice 和多层交互。

缺点：

- 需要打断或重放 backbone forward；
- layer 选择本身成为额外实验变量；
- 当前 FCVC feature tap/replay 按 ARP 风格 block 接口实现，不能直接等同于 Plain ViT block；
- 多层输入、replay 和残差策略容易演变成完整系统，而不是单因素验证。

结论：有研究价值，但应在最终 feature 的协作价值先被证明后再做。

### 方案 C：最终 search feature 融合

在完整 Plain ViT-Tiny 输出之后、`forward_head()` reshape/CENTER head 之前，对 `[B,256,192]` 的本地 search feature 注入 remote search feature。

优点：

- backbone、CENTER head、sampler 和 loss 的接口均可保持不变；
- 256 个完整 search token 已具有较强语义，且不存在 pruning/recovery；
- 关闭协作时可以显式 bypass，直接把原 tensor 送入原 head；
- 可复用当前 flat ABC 的 target/frame/view 对齐和一次 backbone 前向结果；
- 可以只训练很小的 adapter，最大限度冻结 B0 能力；
- 不要求跨相机 token index 一一对应：本地 token 可作为 query，remote token 作为 key/value，按内容匹配。

缺点：

- remote 信息不能改变 backbone 的早期表征；
- 朴素逐位置相加仍然不合理，必须采用内容匹配或无空间假设的聚合；
- 高带宽 remote feature 不是最终通信系统，但这正适合先做 information-value test。

结论：这是当前最合理的 V1 插入位置。

### 方案 D：prediction-level 融合

对 A/B/C 的 bbox、score map 或置信度做选择、平均或重打分。

优点：

- 实现最简单；
- 几乎不改变单机网络；
- 适合作为 oracle、选择器上界或故障诊断。

缺点：

- A/B/C bbox 位于不同相机坐标系，未经标定不能直接平均或 NMS；
- 只传 score 主要验证“选择”而非视觉信息能否帮助定位；
- 容易借助后验指标或 GT 形成泄漏；
- 无法充分回答 remote visual information 是否补足本地遮挡和外观退化。

结论：不作为主 V1，可保留为离线诊断对照。

---

# 3. 复用已有代码分析

## 3.1 PCUM

当前 PCUM 包含 prompt selection/encoding、remote aggregation、alignment、fusion 和若干 reliability 输入。可复用的是工程原则和局部原语，而不是整个系统：

可考虑复用：

- disabled/no-remote 时 identity-preserving 的接口模式；
- sender/receiver prompt 的 shape 校验与 fail-closed 思路；
- `RemotePromptAggregator` 中“多个 sender 聚合”的抽象；
- `PromptAligner` 或 cross-attention 的内容匹配思想；
- 有界 residual，而不是覆盖本地特征。

不应直接复用：

- top-k token selection、saliency、压缩和 token 恢复；
- pseudo remote、历史 ranking、suppression 等附加机制；
- PCUM full forward 与它的多项 loss/诊断组合；
- 将已有 PCUM 结果当作 Plain collaboration 已被否定或已被验证。

原因是完整 PCUM 同时改变 remote 表达、选择、压缩、对齐、融合和可靠性，无法隔离“remote 信息本身是否有用”。此外，当前普通 head 路径主要把最终 template/search token 交给 PCUM；即使模块内部支持 `layers`，也不能据此认定当前 B0 路径已经实现有效的 cross-layer collaboration。

## 3.2 C3R / gate

C3R 中的 `RemoteMessageAdapter` 和 local-first residual 形式与 V1 方向接近，可作为接口参考。它体现了两个值得保留的原则：

- local feature 始终是主分支；
- remote 只产生有界残差，异常时退回 local。

但完整 C3R 还包含压缩、packet、learned gate 和通信策略，不适合作为 E1。已有 `ReliabilityGate` 是学习式多输入 gate，若直接用于 V1 会把“remote 是否有用”和“gate 是否学对”混在一起。

因此：

- E1 不使用 gate，两个合法 remote sender 等权；
- E2 只增加固定、prediction-only 的 reliability weighting；
- learned gate、阈值搜索和 temporal gate 留到 E1/E2 明确正收益之后。

## 3.3 memory

当前仓库没有一个可直接用于本任务、已经验证的视觉 `MemoryBank` 实现：

- tracker 中的 `z_dict1` 更接近固定初始 template，不是跨视角动态 memory bank；
- `TemporalGateRuntime` 的 deque 保存的是 reliability/history feature，不是可检索的 remote visual memory；
- FCVC 的持久状态约束说明了 state isolation，但不等于存在成熟的协作 memory。

memory-based collaboration 会额外引入更新时机、容量、陈旧特征、错误累积、target 隔离、序列 reset 和未来信息泄漏等问题，因此不适合第一个 proof-of-value 实验。

## 3.4 FCVC / cross-layer

FCVC 已有 feature tap、semantic matcher、deformable cross-attention 和 Safe Commit 等组件。

可复用：

- “本地状态提交、协作结果只用于当前帧报告”的 Safe Commit 原则；
- local prediction 与 collaboration prediction 分开记录的诊断方式；
- target/receiver/sender 维度的状态隔离和 reset 规则；
- residual norm、remote usage、fallback 等审计字段。

不直接复用：

- FCVC full high-bandwidth system；
- 多层 tap、replay、query builder、deformable sampler 和 teacher/auxiliary objective；
- 当前按 ARP block signature 编写的 feature replay 路径；
- 未解释 score/APCE collapse 前的 learned gate 或 state-writing 闭环。

这些组件的复杂度会使负结果无法归因。V1 应保留 Safe Commit 原则，但只做单层、单次、最终 search feature 融合。

## 3.5 multiview utilities

这是当前最值得直接复用的一组代码能力：

- common-visible sampler 生成同 target、同 frame 的 A/B/C；
- canonical view order；
- `[V,B,...] ↔ [V*B,...]` flatten/split；
- `target_id`、`view_id`、template/search frame metadata；
- actor 中“给每个 receiver 找另外两个 sender”的映射；
- A/B/C sample count 和 target×view 诊断。

未来实现必须在建立 remote 输入前验证：

```text
receiver.target_id == sender.target_id
receiver.search_frame_id == sender.search_frame_id
receiver.view_id != sender.view_id
sender feature 来自 sender 的 local-only forward
```

`pcum_remote_state.py` 的 prediction-only schema 也可用于 E2 的质量字段，但 source 必须是 `tracker`/local prediction；`gt_legacy` 只能保留兼容，不得进入正式运行。

---

# 4. 最小可验证方案：V1 Final-Search Collaboration

## 4.1 核心假设

假设：当本地视角发生遮挡、模糊或外观退化时，同一目标、同一时刻的另外两个视角所产生的最终 search feature，包含能改善本地 CENTER head 定位的信息。

V1 只验证这个信息价值，不优化带宽，不引入时间 memory，不改变通信触发策略。

## 4.2 remote 传什么

每个 sender 传输其 local-only forward 产生的：

- 最终完整 search feature：`[256,192]`；
- sender local score map 的 prediction-only 摘要，仅供 E2 reliability weighting，例如 `max_score` 和 APCE；
- 路由元数据：`target_id`、`view_id`、`search_frame_id`、valid flag。

明确禁止传输：

- GT bbox、GT visibility、GT IoU 或由 GT 推导的 reliability；
- future frame 信息；
- sender 的 collaboration output 作为本帧其他 receiver 的输入；
- ARP/ATP/pruned/recovered token；
- 历史 memory 或 event-trigger 状态。

V1 使用完整 256-token remote feature 是有意为之：当前目标是判断“信息是否有用”，不是先优化通信量。若高带宽版本都没有正收益，不应先归因于压缩方法。

## 4.3 在哪里融合

建议未来在 actor 中先对 flatten 后的 `3B` 样本做一次 B0 local backbone forward，取得每个 view 的最终 search feature；然后按 batch slot 和 view 映射构建两路 remote feature，只重新执行协作 adapter 与原 CENTER head，不重跑或改写 backbone。

对 receiver `r`：

```text
X_r     : local final search tokens, [256,192]
X_s1/s2 : two remote final search tokens, each [256,192]

Delta_s = CrossAttention(query=LN(X_r), key=LN(X_s), value=LN(X_s))
Delta_r = Aggregate(Delta_s1, Delta_s2)
X'_r    = X_r + bounded_residual(Delta_r)

X'_r -> unchanged CENTER head -> bbox
```

这里的 cross-attention 按 feature content 匹配，不做跨相机的同位置 token 相加。E1 的两个 sender 等权平均；E2 保持完全相同的 adapter，只改变 sender 聚合权重。

## 4.4 关闭时与 B0 完全一致

未来实现必须提供显式 bypass：

- `collaboration_enabled=false` 时不实例化或不调用 adapter；
- no-remote、metadata 不匹配、shape 非法或出现非有限值时返回原始 `X_r`；
- bypass 后 `X_r` 直接进入原 `forward_head()`，不能经过额外 norm、projection 或 dtype conversion；
- backbone、CENTER head、sampler、augmentation 和 GIoU/L1/Focal loss 均不改变；
- zero-residual test、no-remote test 和 disabled test 必须与 B0 输出及本地状态一致。

这应在任何训练前通过 tensor identity/数值一致性 smoke test。

## 4.5 如何避免 negative transfer

V1 采用以下最小保护，不新增 loss：

1. **Local-first residual**：remote 只能提供残差，不能替换本地 feature。
2. **Bounded residual**：限制 remote residual 相对本地 feature 的范数；具体上限必须预注册，不能按 val 反复调阈值。
3. **Zero/near-zero initialization**：协作分支从接近 B0 开始学习。
4. **Fail closed**：remote 缺失、非有限、target/frame/view 不匹配时退回 local。
5. **Safe Commit**：V1 在线评估时，下一帧 crop/template 状态由 local prediction 更新；协作 bbox 只作为当前帧 reported output。这样先隔离“remote 信息价值”，避免单次坏融合污染后续轨迹。
6. **Prediction-only reliability**：E2 的权重只由 sender local output 计算，并 `detach`；运行时不接触 GT。

Safe Commit 的 V1 结果应明确标注为 assisted-reporting/information-value 实验，而不是最终闭环协作 tracker。只有它产生稳定正收益后，才值得单独研究 collaboration output 是否写入 tracking state。

## 4.6 如何判断 remote 是否真的有效

必须同时做功能、因果和结果三类检查。

功能检查：

- disabled/no-remote/zero-residual 与 B0 identity；
- A/B/C 每个 receiver 都实际获得另外两个 sender；
- remote target/frame/view metadata 全部合法；
- adapter 参数获得有限梯度，CENTER head/backbone 按设计冻结或保持既定训练策略；
- 无 GT、无 future frame、无 remote-of-remote。

因果对照：

- correct remote：同 target、同 frame 的其他视角；
- null remote：不提供 remote；
- shuffled remote：batch 内打乱为错误 target，但保持相同 shape 和计算量；
- optional single-sender：分别只给 A/B/C 中一个 sender，用于解释贡献，不用于调参。

只有 correct remote 优于 null，且不被 shuffled remote 复现，才能说明收益来自正确跨视角内容，而不是额外参数或正则化效应。

结果统计：

- OSTRack 同标准的 AUC、Precision、Normalized Precision；
- overall、per-view、per-target；
- 相对 E0 的 helpful/harmful/tie frame 数；
- target-clustered bootstrap 或至少逐 target 差值，避免把大量相关 frame 当作独立样本；
- remote used/fallback 比例、residual norm、E2 sender weights、local max_score/APCE；
- 预测文件先冻结并记录 digest，之后才允许离线 join GT 计算 `delta IoU`；运行时日志不得包含 GT-derived gate 输入。

建议预注册的 go/no-go 条件：E1 overall AUC 至少比 E0 高 `+0.50` 个百分点，Precision/Normalized Precision 不明显下降，没有单一 view 灾难性退化，并且多数 target 的方向不为负。若 E1 未通过，不使用更复杂 gate、memory 或 event trigger 去“救”结果，而应先诊断 remote 匹配、feature 质量和 residual 行为。

---

# 5. 最小 ablation 设计

## 5.1 公共冻结项

E0/E1/E2 必须共同冻结：

- 同一个 B0 checkpoint/provenance；
- Plain ViT-Tiny：dim 192、depth 6、heads 3、patch16；
- template/search size；
- common-visible sampler 和 A/B/C flatten；
- train/val target split；
- CENTER head；
- GIoU/L1/Focal loss 定义及权重；
- augmentation、batch size、samples per epoch 和 evaluation protocol；
- seed 集合和 checkpoint selection 规则；
- OSTRack 风格评测代码。

V1 建议从同一个已验证 E0 checkpoint 出发，冻结 B0 backbone 与 CENTER head，只训练协作 adapter。这样新增可学习能力集中在 remote-to-local 映射，不用重新解释 baseline 漂移。E1/E2 的训练步数、seed 和初始化必须一致。

## 5.2 E0：Plain baseline

```text
local final search feature
        ↓
unchanged CENTER head
        ↓
local bbox
```

- collaboration disabled；
- 复用已验证的 `B0-ABC-Plain`；
- 重新评估时使用完全相同的 checkpoint 和 OSTRack protocol；
- 作为 identity、速度、显存和指标基准。

## 5.3 E1：Plain + collaboration

```text
local final search feature ─────────────┐
remote B/C final search features        ├─ content cross-attention
                                        ↓
                              equal-weight residual
                                        ↓
                              unchanged CENTER head
```

- remote 来自同 target、同 frame 的 sender local-only forward；
- 两个 sender 等权，不使用 reliability、memory 或 learned gate；
- backbone/head/sampler/loss 不变；
- 只训练最小 adapter；
- 这是“remote visual information 是否有用”的主实验。

## 5.4 E2：Plain + collaboration + reliability weighting

结构、remote payload 和 adapter 与 E1 完全相同，唯一变化是两个 sender 的聚合权重：

```text
w_s = fixed_mapping(detached sender local max_score, APCE)
sum_s w_s = 1
```

- 使用固定、预注册、prediction-only 的 weighting；
- 不使用 GT、learned temporal gate、阈值 sweep 或 event trigger；
- 不改变 residual cap 或 adapter 容量；
- 回答“可靠性加权是否比简单等权更能避免 negative transfer”。

建议按阶段执行：先比较 E0/E1。只有 E1 显示 correct remote 的信息价值后，再正式解释 E2；如果 E1 与 shuffled remote 相当或整体为负，应停止增加复杂度。

## 5.5 本阶段明确不做

- ARP、ATP、pruning、compensation；
- PCUM/C3R/FCVC full system；
- 多层/cross-layer replay；
- memory bank；
- learned/temporal gate；
- event-trigger 和通信量优化；
- 新 sampler、新 loss、新 optimizer 或 backbone/head 改造；
- official Three-MDOT test 上调参或模型选择。

---

# 6. 下一步开发建议与排序

## 排名 1：B. feature fusion

最值得首先实现，具体是“最终 256 个 search feature 的单层 content cross-attention + local-first residual”。

理由：

- 与 Plain B0 的结构边界最清楚；
- remote 信息保留最完整，最适合 proof-of-value；
- 不需要假设跨相机空间坐标对齐；
- 可以显式 bypass，并复用 flat multiview 和现有 head seam；
- 失败时归因相对简单：remote 内容、匹配或 residual，而不是压缩/memory/trigger。

## 排名 2：A. target prompt collaboration

如果 feature fusion 证明 remote 有用，再将 remote feature 压缩为少量 target prompts，研究能否保留收益并降低带宽。

它与当前 PCUM 的 prompt encoder/aggregator 更接近，工程复用度高；但若把它放在第一步，负结果无法区分“remote 本身无用”还是“prompt selection/compression 丢失了信息”。因此它应是 V2 的通信表征实验，而不是 V1 的信息价值实验。

## 排名 3：C. memory-based collaboration

适合解决 sender 临时缺失、异步或当前帧质量差，但会引入时序状态、更新策略、陈旧信息和错误累积。当前仓库没有一个已验证且与 Plain baseline 对齐的视觉 memory bank，因此应等单帧同步协作明确有效后再做。

## 排名 4：D. event-trigger communication

event trigger 优化的是“什么时候传”，前提是已经知道“传什么有效”。如果 E1 尚未证明协作收益，trigger 只会增加 gate、阈值和通信预算变量，不能回答核心科学问题。它应放在 feature/prompt collaboration 均稳定以后。

最终建议路线：

```text
B0 identity/static tests
        ↓
V1 final-search feature collaboration: E0 vs E1
        ↓ correct remote benefit established
E2 fixed prediction-only reliability weighting
        ↓ stable positive result
V2 target-prompt compression
        ↓ benefit retained
memory / closed-loop state commit / event-trigger communication
```

当前正确的开发起点不是继续 ARP，也不是接入完整 PCUM/FCVC，而是在 `final search tokens -> CENTER head` 这一清晰边界上建立一个可完全关闭、local-first、无 GT、无 memory 的高带宽协作原型。
