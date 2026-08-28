# E3：Target Semantic Prompt Collaboration 预注册方案

## 1. 研究问题与边界

E3 只检验一个表征假设：在 `B0-ABC-Plain` 的 Plain ViT-Tiny、CENTER head、数据、采样器、损失和训练协议全部不变时，发送端仅传递由其本地预测响应选出的少量目标语义 token，是否比 V1 传递完整 256 个 search token 更适合跨视角协同。

本阶段不研究 sender reliability、selector、gate、时间模型、通信触发、几何对齐、ARP/ATP/pruning、PCUM/C3R/FCVC 或额外损失。E3 是新的独立 V2 adapter；不得修改或迁移 V1 adapter，也不得用 V1 adapter checkpoint warm-start。

## 2. 依据与假设

冻结事实如下：

- E0/B0 是独立本地 tracker；ABC flatten 训练让三视角共同更新同一模型，但没有在线 sender-to-receiver 数据流。
- E1/V1 将每个 sender 的完整 `[256,192]` search feature 交给 receiver。统一 OSTrack 口径的配对结果低于 E0，说明“高带宽 full-search 协同”没有建立收益。
- E2A 的 causal temporal reliability feature 与 E2B 的 full-search pooled prototype 均没有形成可泛化的 sender 选择策略；本阶段停止可靠性 feature engineering。
- V1 的 full-search packet 混入大量背景、干扰物和视角特有空间内容。跨相机相同 token index 没有几何对应关系，完整发送会稀释目标语义并放大 negative transfer。

预注册假设 H-E3：发送端 CENTER raw score map 已经由本地 tracker 产生、且不使用 GT；用它选择响应最高的 `K=8` 个 search token，可以保留多个候选目标部位/模式，丢弃大部分背景。receiver 以自身 256 个 search token 为 query，对每个 sender 的 8-token prompt 独立 cross-attention，可能比 V1 更容易获得有用的跨视角目标语义。

## 3. 参数无关 Target Prompt Extractor

输入：

```text
sender local search tokens X: [B, 256, 192]
sender CENTER raw score map R: [B, 1, 16, 16]
```

固定算法：

1. 将 `R` flatten 为 `[B,256]`；
2. 执行 `torch.topk(R, k=8, dim=1)`；
3. 按 top-k index 从 `X` gather token；
4. 输出 prompt `P: [B,8,192]`，并保留 top-k score/index 仅用于预测侧诊断。

提取器没有 parameter、buffer 或可学习温度；不使用 bbox、坐标、位置编码、mask、visibility、GT、receiver 信息、remote 信息、未来帧或历史状态。prompt 必须来自 sender 同帧的普通本地分支；协同输出不能回流为下一帧 sender prompt。

### 为什么固定 K=8，而不是单个 pooled prototype

单个均值/prototype 会把目标不同部位、多个响应峰和潜在干扰压成一个向量；E2B 已表明现有 pooled semantic 表示不足以支持可靠决策。K=8 保留离散语义多样性，使 receiver 的不同 query 可匹配不同 remote token，同时仍将每 sender token 数从 256 降至 8。K 是本次冻结常数，不进行 K sweep。

## 4. TargetPromptCollaboration V2

输入：

```text
local search:  [B, 256, 192]
remote prompt: [B, R, 8, 192], R=2
remote valid:  [B, R]
```

结构：

```text
Q = LN_local(local search)
for each valid sender r:
    K_r, V_r = LN_remote(remote_prompt[:, r])
    delta_r = shared_MHA(Q, K_r, V_r)
delta = mean(delta_r over valid senders)
bounded_delta = relative_norm_cap(delta, local, cap=0.25)
output = local + alpha * bounded_delta
```

冻结参数：`dim=192`、`heads=3`、sender aggregation=`mean`、residual init=`0.01`、maximum residual scale=`0.25`、relative residual norm cap=`0.25`。MHA 对两个 sender 共享权重，但分别计算，之后只对 valid sender 做等权均值。`K=8` 不要求等于 local token 数 `256`。

安全规则：

- 无 remote 或所有 remote 无效时 exact bypass，输出与 local tensor 逐元素一致；
- non-finite sender fail-closed：该 sender 标为 invalid，不进入均值；
- 若全部 sender 无效则 exact bypass；
- 不用 reliability weight、selector、gate、temporal state 或 receiver-derived sender ranking；
- 相对残差范数不超过 0.25，协作只能有界修正本地表示。

## 5. 训练协议

E3 对齐 V1/E0 的数据和优化协议：

- 初始化使用与 V1 相同的 B0 Plain checkpoint；
- backbone 与 CENTER head 完全冻结并保持 eval；
- 只有 `TargetPromptCollaboration` 的 LayerNorm、shared MHA 与 residual scalar 可训练；
- `TargetPromptExtractor` 参数数量必须为 0；
- 不加载 V1 adapter 参数，E3 adapter 按与 V1 相同规则 fresh initialization；
- 使用 common-visible ABC grouping、view-major flatten、相同 seed、batch size、optimizer、adapter LR、scheduler、epochs、augmentation 和原 tracking loss；
- 每个本地 view 只运行一次冻结 backbone/head，先由本地 raw score 生成 prompt，再构造另外两个 view 的 remote prompt；
- 不增加 prompt supervision、consistency loss、reliability loss 或其他辅助 loss。

协同关闭时，actor、model 和 tracker 必须回到 B0 原路径；E3、V1、PCUM、C3R、FCVC 和其他 cross-view 路径互斥。

## 6. 推理与 Safe Commit

E3 默认且强制采用 Safe Commit：

```text
reported output = E3 collaborative prediction
persistent state = local prediction
next-frame crop = local prediction
sender prompt source = sender local prediction branch
```

每 view/frame 只允许一次本地 backbone/head forward。三视角先各自产生本地 candidate 和 K=8 prompt，再将另外两个本地 prompt 发送给 receiver。协同 head-only forward 不得改变模板、state、next crop 或 sender prompt。runtime 不读取 GT。

最小逐帧诊断字段预注册为：

- `frame_id`、`target_id`、`receiver_view`、`prompt_k`；
- `sender_view_0/1`、`sender_0/1_prompt_norm`；
- `sender_0/1_topk_score_mean/min/max`；
- `residual_norm`、`relative_residual_norm`、`residual_scale`；
- `valid_remote_count`、`used_remote`；
- `reported_output_source=target_prompt_collaboration_e3`；
- `state_output_source=local`、`sender_prompt_source=local`、`uses_gt=false`；
- 每 sender 的理论 `payload_fp32_bytes=6144`、`payload_fp16_bytes=3072`。

## 7. 通信核算

这里只报告理论 tensor payload，不代表网络协议、序列化或真实带宽：

| 方法 | 每 sender/event 元素 | FP32 bytes | FP16 bytes |
|---|---:|---:|---:|
| E1/V1 full search | `256*192=49,152` | 196,608 | 98,304 |
| E3 K=8 prompt | `8*192=1,536` | 6,144 | 3,072 |

E3 相对 V1 的 token 数与 payload 均恰好减少 `32x`。该压缩只是结构性质，不能单独判为实验成功。

## 8. 最小对照与正式评价

未来正式训练完成后，仅做同一 OSTrack evaluator、同一 split、同一目标集合和同一 protocol 的三组比较：

- E0：Local B0；
- E1：V1 full 256-token，Safe Commit；
- E3：K=8 Target Semantic Prompt，Safe Commit。

指标：AUC 为 primary；同时报告 Precision、Normalized Precision、Mean IoU，以及 per-view、per-target paired delta。模型选择仅使用授权的 validation，不用 outer holdout/test 调参。

成功条件全部预注册：

1. E3 AUC 至少高于 Local `+0.30 AUC point`；
2. E3 明显高于 V1 Both Safe；
3. 任一 target 相对 Local 不得下降 `>=5.00 AUC point`；
4. 理论 token/payload 相对 V1 减少 `32x`；
5. identity、Safe Commit、runtime no-GT、一次本地 forward 等完整性检查全部通过。

条件 4 通过而 tracking 条件失败时，结论仍是 E3 失败。若 E3 失败，第一诊断优先级是检查 frozen-local/trainable-only-E3、view grouping、sender prompt local-source、训练/推理不对称和 checkpoint provenance；不立即做 K/温度/layer/gate/selector/architecture sweep。

## 9. 实施与停止规则

允许实施：独立 extractor/V2 module、独立配置、训练 actor 接线、推理 Safe Commit、诊断 CSV、单元测试、CPU/最短 validation target smoke、trainable audit、通信核算和未来命令文档。

当前任务停止于实现与 smoke：不启动长训练，不运行 Three-MDOT official test/outer holdout，不修改既有实验结果。正式训练和正式比较需后续明确授权。
