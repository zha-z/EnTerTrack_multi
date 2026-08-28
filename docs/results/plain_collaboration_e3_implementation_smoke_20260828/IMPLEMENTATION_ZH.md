# E3 实现说明

## 精确变化

E3 新增独立 `TargetPromptExtractor` 与 `TargetPromptCollaboration`，没有修改 `PlainCollaborationV1` 类、参数名或 V1 checkpoint 规则。模型仅在显式 `TARGET_PROMPT_COLLABORATION.ENABLED=true` 且收到 E3 remote prompt 时进入 head-only V2 路径。

提取器输入 sender 本地 `[B,256,192]` search token 与 `[B,1,16,16]` CENTER raw score，固定 `torch.topk(..., K=8)` 后按 index gather，输出 `[B,8,192]`。它没有 parameter/buffer，不读取 GT、bbox、坐标、visibility、receiver、历史或未来帧。

V2 输入 local `[B,256,192]` 与 remote `[B,2,8,192]`。local LN 产生 query；每个 sender prompt 经共享 remote LN/MHA 独立产生 residual；valid sender 等权均值；residual scale 初值 0.01、最大 0.25、relative norm cap 0.25。K 不需要等于 local token 数。无 remote、显式 invalid 或全部 non-finite 时 exact bypass。

## 训练

Three-MDOT actor 复用 common-visible、canonical ABC 和 view-major flatten。本地 B0 backbone/head 在 `torch.no_grad()` 下只运行一次；每 view 从本地 raw score 提取 prompt，再构造另外两 view 的 remote `[V*B,2,8,192]`。第二次仅通过 DDP-visible E3 adapter/CENTER head 计算原 tracking loss。

真实模型审计：backbone 与 CENTER head 全冻结；只有 `target_prompt_collaboration.*` 的 9 个 tensor、148,993 parameters 可训练；extractor 为 0 parameter。V1、PCUM、C3R、search prompt 不存在于 E3 model。

## 推理与 Safe Commit

三 view 每帧先分别产生一次 local candidate 和 local-source K=8 packet。receiver 仅使用另外两个 local packet，协同阶段不改 persistent state。finalize 强制：

```text
reported output = target_prompt_collaboration_e3
state output = local
next crop = local
sender prompt source = local
uses_gt = false
```

逐帧 CSV 包含 target/view/frame、K、两 sender prompt norm 与 top-k score mean/min/max、residual、valid count、output/state/source、理论 payload 和 state digest。

## 文件职责

- `lib/models/entertrack/target_prompt_collaboration.py`：extractor 与 V2 adapter。
- `lib/models/entertrack/target_prompt_collaboration_checkpoint.py`：只接受纯 B0 的 strict core 初始化；完整 E3 checkpoint 仍走 strict load。
- `lib/train/actors/target_prompt_collaboration.py`：ABC prompt mapping 与训练 forward。
- `lib/train/target_prompt_collaboration_freeze.py`：adapter-only freeze/optimizer audit。
- `lib/test/tracker/target_prompt_collaboration.py`：local packet、receiver candidate、Safe Commit。
- `lib/test/evaluation/target_prompt_collaboration.py`：三视角一次 local forward 的每帧编排。
- `lib/test/utils/target_prompt_diagnostics.py`：严格 CSV schema。
- `experiments/entertrack/target_prompt_collaboration_e3.yaml`：独立 E3 配置。
- `tracking/smoke_target_prompt_collaboration_e3.py`：不写结果的最短 val target 2 帧 smoke。
- `tests/test_target_prompt_collaboration.py`：E3 单元/集成测试。

## 当前风险

1. Top-k raw response 可能包含高分干扰物；E3 没有 reliability gate，这是有意控制变量。
2. 两 sender 固定等权，坏 sender 只在 non-finite 时被剔除；negative transfer 仍可能发生。
3. K=8 没有空间坐标，保留的是语义 token 多样性，不提供跨相机几何对齐。
4. 148,993 个新参数需要正式训练；当前 smoke 只证明实现完整性，不证明 AUC 收益。
5. 训练继承的 LR drop epoch=28 高于总 epoch=25；这是 V1/B0 继承值，本次按一因素原则记录但未修改。
