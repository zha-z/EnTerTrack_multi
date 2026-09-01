# E3 D2-S1 P50 Source-only Collaboration Training 预注册

## 1. 唯一研究问题与授权边界

D2-P2 已在 Train-only 选择并冻结 P50，随后 frozen P50 在 VAL holdout 通过。D2-S1 只回答：把 E3-D1 的训练期 P100 source 换成冻结 P50，是否能在不改变 E3 adapter、sampler、loss 和优化协议的前提下改善 source-only collaboration training。

本预注册只授权独立实现、静态/数据流/gradient smoke 和报告。当前任务禁止启动 25 epoch 正式训练，禁止 `threemdot_test` / official test，禁止 Rescue/Preserve loss、gate/selector、sampler 修改、degradation sweep 或任一 K/residual/cap/LR 调参。smoke 完成后 STOP。

## 2. 冻结来源与身份

- 分支：`feature/pcum-cross-layer-arp`，禁止使用或合并 `main`。
- D2-P2 frozen selection：`docs/results/target_prompt_d2_p2_partial_source_calibration_20260901/selected_source.json`。
- selected source SHA256：`ca8587cd4d0eb8d7293c2e4c762da6f6c76a9612b43ccadf23411626ab912e60`。
- candidate：只能是 `P50`；`coverage=0.50`；P25/P75/P100 均不可重新选择。
- fill：Normalize 后常数 `0.0`。
- block：`single_contiguous_edge_anchored_block`。
- orientations：`left/right/top/bottom`。
- namespace：`D2-P2-orientation-v1`。
- orientation：`SHA256(namespace + NUL + D2-P1 clean sample_id)` 的前 8 bytes 按 big endian 转整数，再 modulo 4。

训练 loader 的稳定 identity 严格构造为：

```text
d2p1-train-<target_id>-<view_suffix>-<search_frame_id:06d>-clean
view suffix: A->1, B->2, C->3
```

identity 只读取现有 grouped loader 已提供的 `target_id`、canonical `view_ids` 和 synchronized `search_frame_ids`。不修改 sampler，不使用 Python `hash()`，不以 `torch.randint()`、worker seed、rank、batch position 或 epoch 替代 orientation。现有 PyTorch RNG 只继续承担 D1 已冻结的 selected-triplet 与 weak-view 均匀抽样。

## 3. 唯一实验变量

与 E3-D1 相比唯一变量：

```text
selected weak-view target source: P100 -> frozen P50
```

以下全部冻结：

- B0 ep25 core 与 fresh E3 adapter 初始化；
- Plain ViT-Tiny、CENTER head、完整 256 search tokens；
- common-visible sampler、canonical A/B/C、view-major grouped loader；
- 50% synchronized triplet，exactly one uniformly selected weak view；
- TargetPromptExtractor K=8；
- adapter 结构、3 heads、mean aggregation；
- residual init `0.01`、max `0.25`、relative norm cap `0.25`；
- 原 tracking loss 与 GIoU/L1/Focal 权重；
- AdamW、adapter LR `8e-5`、weight decay `1e-4`、scheduler；
- 25 epochs、per-rank batch 2、Train 6000 samples/epoch、VAL 1500 samples/epoch；
- backbone/head frozen、仅 148,993 E3 adapter parameters 可训练；
- inference no-GT 与 Safe Commit；
- validation exact bypass，不施加 P50。

独立配置为 `experiments/entertrack/target_prompt_collaboration_e3_d2_s1.yaml`，独立 source implementation 为 `lib/train/target_prompt_d2_s1_source_degradation.py`。历史 D1 实现与配置不修改；D1 P100 和 D2-S1 P50 同时启用时必须 fail closed。

## 4. P50 rasterization

对选中 triplet 的 exactly one weak view，在 processing/normalization 后的 `search_images [3,B,3,256,256]` 上：

1. 使用该 view 的 crop-space normalized `xywh` annotation；
2. 沿用 D1 的 `floor(x,y)` / `ceil(x+w,y+h)` 与边界 clipping 得到 full pixel bbox；
3. 用 stable clean sample ID 得到 frozen orientation；
4. left/right 取 `ceil(box_width * 0.5)` 的整高 edge block，top/bottom 取 `ceil(box_height * 0.5)` 的整宽 edge block；
5. 仅将该 contiguous block 填成 normalized `0.0`；
6. template、annotation、另外两个 views、未选 triplet 和 metadata 不变。

由于整数 rasterization，实际 bbox pixel coverage 应接近 50%，小 bbox 可因 `ceil` 略高；必须逐 batch 记录 realized coverage，不得通过 sweep 改规则。

## 5. 数据流、梯度与初始化边界

```text
real synchronized common-visible ABC batch
  -> train-only D2-S1 clone(search_images)
  -> selected weak view frozen P50 block
  -> frozen B0 local forward + K8 extraction
  -> fresh trainable E3 adapter
  -> unchanged frozen CENTER head / tracking loss
```

正式训练未来必须从以下 frozen core fresh initialize：

```text
output/diagnostics/b0_abc_plain/run_20260818_seed42_4gpu_r001/
checkpoints/train/entertrack/b0_abc_plain_4gpu/EnTeRTrack_ep0025.pth.tar
SHA256 363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3
```

禁止从 D1 checkpoint、E3 checkpoint 或任何已有 adapter state fine-tune。strict loader 必须拒绝 inherited `target_prompt_collaboration.*` / extractor tensors。smoke 只做 forward/backward，不执行 optimizer step，不写 checkpoint，不构成训练启动。

GT bbox 在本阶段只用于训练期 P50 input transform 与原 tracking supervision；不会进入 adapter input、validation/inference runtime feature 或 Safe Commit state。

## 6. 长训练前 smoke gates

以下全部 PASS 才可给出 `READY_FOR_D2_S1_LONG_TRAINING = YES`：

1. P50 输出与 frozen D2-P2 transform 逐像素 exact；
2. E3/default OFF search/data object exact identity；
3. 历史 D1 P100 route exact regression；
4. validation exact data/search tensor bypass；
5. B=2 恰好 1/2 triplet，且 exactly one weak view；
6. bbox realized coverage 约 50%，single contiguous edge-anchored，fill=0；
7. template 与 annotation object/tensor unchanged；
8. stable identity 完整，orientation 与 frozen SHA256 helper exact；缺失或非 canonical metadata fail closed；
9. D2-S1 与 D1 的模型、sampler、loss、optimizer/LR/epoch/batch/sample contract 相同；
10. B0 ep25 checkpoint SHA/strict core load/fresh adapter PASS；
11. trainable/optimizer membership 仅 148,993 E3 adapter parameters；
12. real Three-MDOT loader common-visible/canonical/synchronized PASS；
13. GPU loss/forward/backward/adapter gradients finite，frozen core gradient count 0；
14. 环境允许时 DDP >=2 ranks backward/all-reduce PASS；
15. 没有 optimizer step、checkpoint write、25-epoch training、VAL tuning 或 test access。

任一 identity、data-flow、initialization 或 gradient gate 失败即 `READY...=NO`，STOP，不得启动长训练或放宽门槛。

## 7. 未来命令与停止点

只有本轮 smoke 全 PASS 且用户后续单独授权，才允许给正式训练分配独立 run directory/runid，并从上述 B0 ep25 core 启动固定 25 epoch D2-S1。不得 resume D1。当前任务不执行该未来命令。

smoke 产物固定写入：

```text
docs/results/target_prompt_collaboration_e3_d2_s1_smoke_20260901/
```

报告必须包含命令、commit、config/source hashes、frozen checkpoint provenance、unit/loader/GPU/DDP gates、未运行项与最终 READY/STOP。完成后 STOP。
