# EnTeR-Track 多无人机协同跟踪：给 ChatGPT 的项目上下文

> 用途：把本文件作为新对话的首条上下文交给 ChatGPT。它不是新的实验协议，也不替代仓库中标为“唯一规范”、frozen protocol、manifest 或 acceptance criteria 的文件。若本文件与专项协议冲突，以专项协议为准。

## 1. 希望 ChatGPT 扮演的角色

你是这个项目的研究与代码协作助手。请先理解已有算法、历史实验、负结果和科学边界，再提出建议或修改代码。不要把仓库里存在的实验配置理解为都值得继续，也不要把一次离线诊断、单折结果或 test 集上的最好观测值写成正式结论。

处理任何任务时，请遵守以下顺序：

1. 先确认本轮授权范围：只读分析、静态检查、实现、单元测试、训练、验证、正式测试是不同权限。
2. 查找并遵守对应专项协议；带“唯一规范”、frozen、manifest、acceptance criteria 的文件优先级最高。
3. 明确本轮使用的数据、target split、checkpoint、runid、预测文件和输出目录。
4. 先保证默认关闭时与原模型一致，再讨论模块启用后的收益。
5. 把“运行时可用的 prediction-only 信息”和“仅用于训练标签/事后评估的 GT”严格分开。
6. 报告成功、失败和 `MISSING_DIAGNOSTIC`，不要根据缺失日志推断运行时事实。
7. 若冻结门槛失败，停止该阶段，不降低阈值、不从失败数据中挑子集继续训练。

## 2. 项目一句话定义

本项目以 EnTeR-Track/OSTrack-Tiny 单目标跟踪器为本地基础，在 Three-MDOT 同一目标的 A/B/C 三个同步无人机视角之间引入视觉协作，使每个接收端在没有 GT、没有未来帧、没有 oracle 可见性信息的条件下利用其他无人机的信息，同时尽量保持轻量、低带宽、可部署，并避免错误远端信息污染本地闭环状态。

## 3. 最终目标

### 3.1 算法目标

### 2.1 最新冻结实验状态（2026-08-28）

最新受控结果见 [`docs/results/target_prompt_collaboration_e3_ep25_test_20260828/REPORT_ZH.md`](results/target_prompt_collaboration_e3_ep25_test_20260828/REPORT_ZH.md)。同一 Three-MDOT test、同一 OSTrack `calc_seq_err_robust`、105 个 A/B/C 独立序列宏平均口径下：

- E0/B0 Plain ep25 AUC 48.9464；
- E1/V1 full-search collaboration ep25 AUC 48.0028；
- E3/K8 target semantic prompt ep25 AUC 48.7651。

E3 相对 B0 为 -0.1813 AUC point，相对 E1 为 +0.7622 point。E3 的 no-GT、Safe Commit、remote-active 和 32x payload reduction 检查通过，但预注册主门槛 `B0 +0.30 point` 失败，因此结论是 **E3 FAIL / stop test-side tuning**。后续只能回到 inner-val/inner-dev 做 prediction-only utility 诊断，不能依据该 test 调 K、残差强度、阈值或逐视角权重。


- 提升 Drone A/B/C 各单视角的 AUC、Precision 和 Norm Precision，而不只提高离线 fused 指标。
- 重点改善遮挡、长遮挡、重出现、快速运动、模糊和视角不均衡条件下的鲁棒性。
- 远端视角有用时产生可测收益，无用、延迟、错误或缺失时能够安全退回本地结果。
- 保留本地跟踪器的速度和轻量优势，并对参数量、FLOPs、FPS、时延、通信字节数做统一核算。

历史上曾把“三个单视角 AUC 都接近或超过 0.55”作为进取目标。它是研究愿景，不是允许在 test 集上反复调参的理由；正式结论必须服从冻结协议和受控对照。

### 3.2 科学目标

需要回答的不只是“某个配置分数是否更高”，而是：

- EnTeR 的 ARP/ATP 轻量化路径本身相对无剪枝 B0 的真实影响是什么？
- PCUM 的收益来自远端信息、训练方式、主干共同适配，还是 checkpoint/数据差异？
- 低带宽 C3R 是否保留了足够的远端信息？
- 负迁移是否能被 prediction-only 信号可靠识别，而不是依赖 GT 或 test sweep？
- 协作结果能否只影响当前帧报告，不污染下一帧 crop、模板、bbox、运动和置信度历史？

### 3.3 正式评价维度

- 总体：AUC、Precision、Norm Precision。
- 分视角：Drone A、B、C 分别报告。
- 分 target：A/B/C 三个视角按同一 target 绑定，报告正/负/不变 target 数。
- 统计：以 target 为重采样单元做 clustered bootstrap，不能把 105 个序列视作独立样本。
- 安全性：负迁移率、harmful/helpful/tie、fallback 原因、状态身份一致性。
- 工程性：参数量、FLOPs、FPS、latency、显存和真实 serialized bytes/frame。

## 4. 本地 EnTeR-Track 算法流程

### 4.1 输入与状态

每个无人机拥有独立 tracker 实例和独立状态。第 1 帧由 GT 初始化模板和 bbox；后续帧只允许读取上一帧本地持久状态。

- template crop：通常为 `128 x 128`，patch size 16，对应 64 个模板 token。
- search crop：通常为 `256 x 256`，对应 `16 x 16 = 256` 个搜索 token。
- Tiny ViT：embedding 192、6 个 Transformer block、3 个 attention head。
- 输出头：CENTER head，产生中心响应图、尺寸、偏移和 bbox。

### 4.2 单帧本地闭环

```text
上一帧本地 bbox/state
    -> 从当前图像裁剪 template/search
    -> patch embedding + 位置编码
    -> template/search token 拼接进入 one-stream ViT
    -> ARP/ATP 进行自适应 token 保留与被剪 token 补偿
    -> 恢复搜索 token 的空间顺序
    -> CENTER head 输出响应图、bbox、score、APCE
    -> 映射回原图坐标
    -> 更新下一帧本地 bbox、crop、运动/置信度历史
```

### 4.3 EnTeR 的轻量机制

- Entropy-guided saliency：用浅层注意力分布的熵/集中度判断搜索 token 重要性。
- ATP：根据当前样本复杂度预测自适应剪枝阈值。
- Reversible token compensation：被剪 token 不直接永久丢弃，而是通过轻量残差估计恢复上下文，再送入定位头。

这条路径的研究意义是提高速度/算力效率，但受控 B0/B1 结果表明，当前完整 ARP/ATP 结构与训练包在 Three-MDOT 上低于无剪枝 B0。因此不能默认“轻量路径已经不损失精度”；它本身仍是需要解释和修复的基础问题。

## 5. 多机协作的三条主要路线

三条路线解决不同通信预算和信息粒度的问题，不应在没有协议的情况下随意混合。

### 5.1 PCUM：提示级协作

PCUM 是当前最成熟的跨机视觉提示模块，位于 backbone 输出与 CENTER head 之间。

```text
本地 search/template tokens
    -> SaliencyTokenSelector 选 top-k token
    -> MultiLayerPromptEncoder 压缩为少量 prompt
远端两个无人机 prompt + tracker-derived confidence
    -> PromptAligner 做语义对齐与远端聚合
    -> PromptFusion 以 gated-add 或 FiLM 写回本地 search tokens
    -> CENTER head 输出协作候选框
```

当前正式 PCUM-v2A A0 设置使用：

- 4 个 prompt，prompt dim 192，top-k 16。
- `confidence_softmax` 聚合两个远端，temperature `0.10`。
- 远端状态来自 tracker 自身，`USE_REMOTE_VISIBLE_MASK=false`。
- validation/test 推理不使用 GT 可见性、GT bbox 或 IoU。

PCUM 的历史认识是：

- 远端 prompt 对部分视角、可见帧和重出现恢复有帮助。
- 加权聚合缓解了早期 `zero > raw` 的问题，但没有消除负迁移。
- 简单置信度 selector、增强 selector、ranking、visible-safe 和 remote suppression 均未形成稳定的安全解法。
- 不能通过 test 集 selector sweep 挑结果，也不能把 oracle 上限当作可部署收益。

### 5.2 C3R：严格低带宽协作

C3R 用固定 320-byte application packet 研究低带宽条件下的协作。

每个发送端 packet 包含：

- 32-byte header。
- 预测 bbox、bbox delta 和四个 response-quality 标量。
- 4 个量化 scale。
- `4 x 64` 的 int8 prompt，共 256 bytes。

接收端流程：

```text
解析和校验 packet
    -> 拒绝 malformed/stale/wrong-sequence/duplicate sender
    -> RemoteMessageAdapter 对本地 token 做跨注意力
    -> 10 维 prediction-only reliability 输入
    -> 10 -> 32 -> 1 reliability gate，最大 gate 0.25
    -> 单 peer residual norm cap 0.25
    -> aggregate residual norm cap 0.35
    -> local-first residual fusion
    -> 仅在收到有效远端时重跑 CENTER head
```

C1 是当前主要 C3R 变体。C3R 的本地 backbone/head 冻结，模块默认关闭；无 packet、packet 无效或模块关闭时必须回到本地 identity。

Temporal Gate 是 C3R reliability-to-fusion 边界上的 sidecar，而不是新 tracker：

- v1 使用 W=8 的 `GRU(10,16)+Linear`，但 IoU 非 tie 覆盖只有约 82.38%，低于冻结的 95% 门槛，因此 v1 为 `INVALID`，不能继续利用其样本。
- v2 改用连续 `delta_diou` 监督，固定 SmoothL1 和非正 utility 精确置零的 gate 公式；fold1 的 prediction freeze、GT 后连接和反事实完整性 Gate-0-v2 已 `PASS`。
- 这个 PASS 只说明数据和因果链可用于下一阶段审计，不等于已经训练成功，也不授权 validation/test。

### 5.3 FCVC：高带宽稠密特征协作

FCVC 用于验证一个核心假设：PCUM/C3R 可能因压缩过强丢失空间细节，因此直接交换较高带宽的中层和高层稠密特征，再逐步研究压缩。

FCVC 单帧流程：

```text
A/B/C 各自完成冻结 E0 本地前向
    -> 在 block 3 和 block 6 捕获 dense feature tap
    -> receiver 从本地响应、置信度和目标原型构造 8 个 query
    -> 对两个 sender 做 global semantic matching
    -> 在 sender 自己的坐标域做 deformable cross-view sampling
    -> mid residual writer 写入 receiver search tokens
    -> 重放冻结 block 4-6
    -> high cross-view block + high residual writer
    -> 冻结 CENTER head 输出 collaborative candidate
```

固定结构包括 128 维 query embedding、4 heads、两个 sender、每 sender/query 4 个采样点、null token 和两级 cross-view block。

FCVC 的关键不是“总是相信协作框”，而是 Safe Commit：

```text
state_output    = local_candidate
reported_output = collaborative_candidate（无效时回退 local）
```

下一帧 crop、bbox、模板/记忆、运动、score、APCE 和所有持久状态只能读取 `state_output`。`reported_output` 只能写结果文件，不能回流到 tracker state。

FCVC 固定训练目标为：

```text
L = L_track + 0.50 L_align + 1.00 L_recon
    + 0.50 L_safe + 0.10 L_cycle
```

teacher 只在训练时读取 GT ROI，重建目标 stop-gradient；推理中不存在 teacher 或 GT。

当前 FCVC 还不能作为成功方案。配对训练指标曾显示收益，但闭环在线验证和正式结果没有建立稳定增益。此前复盘还观察到 FCVC score/APCE 显著塌缩，说明训练量、报告量和闭环跟踪量可能并不等价；在再次全量训练前，应先完成 no-remote、`force_null`、zero-residual 和 remote-present/zero-residual 身份诊断。

## 6. 运动与重检测路线的定位

运动信息曾用于 motion prior、motion-state shadow 和 MCR/redetection 诊断。历史结果出现过 NaN、不稳定或无法通过 grouped OOF utility gate。因此当前原则是：

- 不让运动模型直接覆盖 detector bbox。
- 不在长遮挡期间无约束累积速度。
- 运动只可作为 prediction-only 的弱诊断、候选搜索范围或触发依据。
- 任何 motion trigger 都必须先在 shadow 模式证明事件级收益，再允许写入主流程。

Reliability v2 对已有 temporal、trajectory、state-divergence 信号做过严格 nested target-group 审计，最终为 `S4`：没有候选达到冻结门槛。因此不能直接用这一批旧信号再包装一个新 Gate。

## 7. 当前已确认的结果与正确解释

以下数字来自仓库现有冻结报告，单位均为百分数。

| 方法 | 角色 | AUC | Precision | Norm Precision | 正确解释 |
|---|---|---:|---:|---:|---|
| B0 | 无 ARP/ATP、无 PCUM 的受控 ViT 基线 | 49.804 | 66.018 | 79.177 | 当前严格受控总体最高 |
| B1 | 独立 EnTeR/ARP 基线 | 48.152 | 63.699 | 77.256 | ARP/ATP 完整包相对 B0 下降 |
| A0 | B1 + PCUM-v2A | 48.566 | 64.332 | 77.850 | 相对 B1 小幅回升，但仍低于 B0 |
| B0+PCUM | 冻结 B0 core、只训练 PCUM | 49.350 | 65.114 | 78.775 | 未超过 B0；是 post-hoc exploratory 对照 |

关键受控差值：

- `B1 - B0 = -1.652 AUC`。
- `A0 - B1 = +0.414 AUC`。
- `A0 - B0 = -1.237 AUC`。
- `B0+PCUM - B0 = -0.454 AUC`。

这些 target-cluster bootstrap 区间均包含 0，所以应报告为冻结 test 上的描述性方向，不能写成已经统计解决的普遍因果效应。

PCUM-v2A A0 仍可作为当前论文协作主结果：它相对相同 EnTeR/ARP 背景的 B1 有小幅提升，并优于自己的 zero/delay/none 消融。但它不是全系统最优，也不能声称解决负迁移。

### Plain Collaboration V1 配对结果（2026-08-27）

最终 search-feature 等权融合的 `B0-ABC-Plain-Collaboration-V1 ep25` 已完成 105/105 Three-MDOT test 序列。按同一 OSTrack evaluator，它的 AUC 为 `48.0028`；与它直接配对的 `B0-ABC-Plain-4GPU ep25` run `26025` 为 `48.9464`，差值 `-0.9435` 个百分点。逐视角差值为 A `-2.7513`、B `-1.2748`、C `+1.1955`。remote 在所有非初始化帧真实启用，但无可靠性判断的等权融合没有通过整体增益目标。

这里的 E0 是 V1 实际初始化并冻结的 run `26025`，不能与上表来自其他受控 run/sampler 的 B0 `49.804` 或旧 inner-val `64.89` 混用。完整报告和 ChatGPT 可直接读取的训练/评测产物位于 [`docs/results/plain_collaboration_v1_ep25_test_20260827/REPORT_ZH.md`](results/plain_collaboration_v1_ep25_test_20260827/REPORT_ZH.md)。

## 8. 思路演化：为什么现在不能继续盲目堆模块

1. 最初思路是用 PCUM 把同步远端视觉 prompt 融入本地 token。实验说明远端信息确实有价值，但收益视角不均衡，且接近一半 target/sequence 可能出现负迁移。
2. 随后尝试 selector、ranking、visible-safe 和 suppression，希望只在远端有益时启用。oracle 上限很高，但 prediction-only 特征可分性不足，很多 gate 最终退化为几乎全本地或停留在初始化附近。
3. C3R 把通信压到 320 bytes，并加可靠性 gate 和 residual cap。它工程边界清晰，但现有 reliability 信号没有证明能稳定区分 helpful/harmful。
4. Temporal Gate v1 因标签结构性 tie 被判 `INVALID`；v2 修正了监督定义并通过数据完整性 Gate-0，但还不是有效性结果。
5. FCVC 转向高带宽稠密特征，想先证明“信息充分时是否能协作”，再做压缩。但当前闭环结果仍低于本地，并伴随 score/APCE 失配，因此问题可能在 Safe Commit 接缝、坐标域、训练/推理量不一致或 head 响应校准。
6. 严格 B0/B1/A0 对照进一步表明：主干轻量化损失可能大于 PCUM 带来的增益。下一阶段必须同时关心本地基线和协作模块，不能只围绕远端 gate 微调。

## 9. 期望的方案形态

理想方案不是“再增加一个能看 GT 的 selector”，而是以下可审计结构：

```text
独立、稳定、可复现的本地轻量 tracker
    + prediction-only 的远端消息/特征
    + 受范数或结构限制的局部特征残差
    + Safe Commit（本地状态、协作报告分离）
    + 缺失/错误/延迟远端时的 fail-closed identity
    + target-group 冻结验证与一次性正式测试
```

短期工程上应保留三个可比较层次：

- E0/B0/B1：不协作本地参考。
- PCUM/C3R：低或中带宽、可部署协作。
- FCVC：高带宽信息上限和未来压缩教师。

若 FCVC 在身份诊断后能够证明高带宽收益，再按“通道压缩 -> 空间稀疏 -> response-guided sparsity -> FP16/INT8 -> codebook -> distillation -> 320-byte”的顺序一次只改变一个因素。若高带宽本身都无收益，则应停止压缩路线，回到坐标、闭环和训练目标诊断。

## 10. 建议的实现计划

### 阶段 0：冻结上下文与基线

- 固定数据 split、target 列表、A/B/C 绑定、checkpoint SHA256、runid 和配置摘要。
- 把 B0、B1、A0、B0+PCUM、C1、FCVC 的角色写入 registry，禁止混用训练数据或 checkpoint。
- 统一指标实现和结果目录，任何后处理都只读冻结 bbox/score/APCE。

完成条件：同一结果可由 manifest 重现，且没有 test sweep、outer-holdout 泄漏或 runid 混读。

### 阶段 1：先闭合本地与身份诊断

- 验证各模块 `ENABLED=false` 时 bbox、score、APCE 和持久状态 bitwise/digest identity。
- 对 FCVC 做 no-remote、`force_null`、zero-residual、remote-present/zero-residual 四组小样本检查。
- 对 3-5 个 target 逐帧记录 local/collaborative bbox、score、APCE、IoU、residual norm、`used_remote` 和 fallback reason。
- 检查训练 IoU 到底对应 local、collaborative、teacher-assisted、crop-local 还是闭环原图量。
- 检查 B1 相对 B0 的退化来自剪枝、ATP、补偿、训练数据还是联合训练设置；只允许预注册的一因素受控诊断。

停止条件：只要 zero-residual 不等于本地、协作输出写入状态、坐标域不一致或 checkpoint provenance 不闭合，就不启动新全量训练。

### 阶段 2：只选择一条主路线推进

- 若高带宽 FCVC 小样本身份和响应校准通过：先做固定 inner-dev 的一次预注册验证，不增加 selector 或新 Gate。
- 若 FCVC 仍失败但 PCUM 稳定：把 PCUM 作为论文协作主线，重点完善受控对照、效率和失败案例，不夸大负迁移安全性。
- 若目标是部署通信：以 C3R 320-byte contract 为唯一 packet 规范；Temporal Gate v2 只有在单独授权后才能从 Gate-0 进入训练。
- Reliability v2 的旧候选信号已 S4，除非消息设计或可用运行时观测发生实质变化，否则不重复同类 signal sweep。

### 阶段 3：模块级实现与测试

- 新模块必须 standalone、default-off、Python 3.8 兼容。
- 配置默认值写入 `lib/config/entertrack/config.py`，实验差异写入 `experiments/entertrack/*.yaml`。
- 先做 tensor-level 单元测试、shape/finite/gradient/freeze 测试，再做最小 tracker identity 测试。
- 不允许一个协作实验同时修改 backbone、head、prompt、gate、通信格式和 evaluator。

### 阶段 4：训练前审计

- optimizer 只包含授权参数；冻结参数和 BatchNorm buffer 必须核验。
- 检查 loss 的真实依赖图，GT 只进入训练 target/loss，不进入 inference feature/state。
- 检查 seed、DataLoader worker、epoch shuffle、resume guard 和 manifest hash。
- 先做 no-update loss/gradient scale audit；不得借 scale audit 调 lambda。

### 阶段 5：inner-dev 与正式测试

- 所有 view 按 target 分组，outer holdout 不参与选择。
- prediction-only rollout 先生成并冻结 SHA256，再事后 join GT。
- 按冻结 acceptance gate 决定 PASS/FAIL/INVALID。
- 只有 gate PASS 且用户单独授权，才能进行一次正式 test；test 后不再调 epoch、阈值或 selector。

### 阶段 6：最终报告

至少包含：

- 文件、配置、checkpoint、SHA256、runid、目标/序列/帧数。
- 默认关闭 identity、freeze/optimizer、GT boundary、split/leakage。
- 总体和 A/B/C 指标、target-level 正负数、clustered bootstrap。
- 参数量、FLOPs/FPS/latency、通信字节。
- 正结果、负结果、失败案例、未验证项和明确 stop/go 结论。

## 11. 关键代码入口

| 功能 | 主要位置 |
|---|---|
| 基础模型构建、backbone/head/PCUM/C3R 接口 | `lib/models/entertrack/entertrack.py` |
| ARP/ATP 与 token 恢复 | `lib/models/entertrack/vit_arp.py` |
| PCUM selector/encoder/aligner/fusion | `lib/models/entertrack/pcum.py` |
| C3R packet/adapter/reliability/residual fusion | `lib/models/entertrack/c3r.py` |
| Temporal Gate sidecar | `lib/models/entertrack/temporal_gate.py` |
| FCVC 模型 | `lib/models/entertrack/fcvc/` |
| 推理闭环、状态提交、多视角 runner 接缝 | `lib/test/tracker/entertrack.py` |
| Three-MDOT/PCUM/C3R 训练 actor | `lib/train/actors/entertrack_threemdot.py` |
| FCVC 训练图与验证 | `lib/train/fcvc_*`, `lib/train/actors/fcvc_actor.py` |
| 配置默认值 | `lib/config/entertrack/config.py` |
| 实验 YAML | `experiments/entertrack/` |
| 正式结果与受控对照 | `output/final_paper_results/`, `output/controlled_baselines/` |
| 冻结协议与审计 | `output/multi_agent_collaboration_clean/` |

## 12. 禁止事项和常见误区

- 禁止在 validation/test 推理中读取 GT bbox、visibility、target_visible 或 IoU。
- 禁止先 join GT 再声称特征是 prediction-only。
- 禁止用 outer-held-out target 选信号、阈值、epoch 或模型。
- 禁止把 A/B/C 三个视角拆开随机分 fold。
- 禁止把 offline AUC、oracle selector、配对 crop IoU 当成闭环 tracking gain。
- 禁止因 gate 失败而降低阈值、删除难例或复用 `INVALID` 样本。
- 禁止协作输出更新下一帧本地状态；FCVC 必须遵守 Safe Commit。
- 禁止默认开启新模块；disabled/no-remote/invalid-remote 都必须有 identity/fallback 证据。
- 禁止仅修改 YAML 就声称可复现；必须检查 executable seed、DataLoader 和 resume 路径。
- 禁止在用户未授权时启动训练、真实 rollout、validation、test 或参数搜索。

## 13. 给 ChatGPT 的任务回答模板

当我提出下一项任务时，请按下面格式回答：

1. **你对目标的理解**：本轮要验证的唯一问题是什么。
2. **授权边界**：只读、实现、测试、训练、验证、正式 test 中哪些允许，哪些不允许。
3. **事实基线**：使用哪些冻结协议、checkpoint、split 和已有结果。
4. **最小方案**：只改变哪些文件/参数，为什么能回答问题。
5. **身份与泄漏检查**：default-off、Safe Commit、prediction freeze、GT join、target grouping。
6. **验收门槛**：预先写清 PASS/FAIL/INVALID 和停止条件。
7. **执行与产物**：命令、输出目录、manifest、SHA256、报告。
8. **结论边界**：结果最多能支持什么，不能支持什么。

如果现有证据不足，请优先建议一个小型 identity/diagnostic 实验，而不是直接再训练一个完整模型。
