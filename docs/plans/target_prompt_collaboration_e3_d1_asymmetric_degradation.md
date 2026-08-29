# E3-D1 Asymmetric Degradation Training 预注册

## 1. 研究问题与唯一变量

E3-U1 在 `threemdot_val` 上得到 Oracle-single `+1.4317 AUC point`，但 P4 sender-specific Task-A target-LOTO ROC 仅 `0.4436`，OOF Safe Report 只提升 `+0.0781 point`。该结果对应 Case B：K=8 representation 中存在 rescue candidate，但 common-visible 同质训练没有让 adapter 稳定学习 receiver weak / remote strong 的非对称使用方式。

E3-D1 只改变训练期输入分布：在固定比例的同步 A/B/C triplet 中，将 exactly one view 的 search target region 做固定 occlusion。其余模型、训练和推理协议全部继承 E3，不进行调参。

## 2. 冻结不变量

与 `experiments/entertrack/target_prompt_collaboration_e3.yaml` 保持一致：

- B0 checkpoint 初始化及其路径；
- Plain ViT-Tiny、完整 256 search tokens；
- CENTER head；
- TargetPromptExtractor `K=8`，参数数量 0；
- TargetPromptCollaboration 结构、3 heads、mean sender aggregation；
- residual init/max/relative norm cap；
- backbone 与 CENTER head 冻结、adapter-only training；
- common-visible sampler、canonical A/B/C、view-major flatten；
- optimizer、adapter LR `8e-5`、weight decay、scheduler；
- per-GPU batch=2、global batch=8、25 epochs；
- train/val samples per epoch、augmentation、tracking loss及其权重；
- inference K=8/no-GT/Safe Commit；
- validation 不增加 occlusion。

禁止 K、temperature、residual、loss、gate、selector、sender weight、view-specific policy、checkpoint sweep、official test。

## 3. 固定 degradation 算法

输入为 processing/normalization 后、进入 E3 actor 前的：

```text
search_images: [V=3, B, C=3, H=256, W=256]
search_anno:   [V=3, B, 4]，crop-space xywh pixels
```

仅当 `network.training=true` 且 D1 显式启用时：

1. 要求 `V=3`、formal local batch `B=2` 且 B 为偶数；
2. 在本地 batch 的 B 个 synchronized triplet 中，用当前 seeded PyTorch RNG 随机 permutation，选择恰好 `B/2` 个 triplet；
3. 对每个被选 triplet，以均匀分布从 A/B/C 选择 exactly one weak view；
4. 读取该 view 的 crop-space target bbox；按 `floor(x,y)` / `ceil(x+w,y+h)` rasterize，并裁剪到 search 边界；
5. 将该 bbox 的全部 RGB pixels 填为归一化空间常数 `0.0`；
6. bbox supervision、template、另外两个 search view、attention mask和所有 metadata保持不变。

冻结常数：

```text
TRIPLET_RATIO = 0.50
WEAK_VIEWS_PER_SELECTED_TRIPLET = 1
OCCLUSION_BOX_SCALE = 1.00
FILL_VALUE_NORMALIZED = 0.0
VIEW_SELECTION = uniform(A,B,C)
```

不使用随机 occlusion 大小、强度、纹理、噪声、blur 或 color sweep。`0.0` 是 Normalize 后的 dataset channel-mean 中性填充值，不引入额外可学习参数。

正式配置 per-GPU batch=2，因此每 rank/iteration 恰好 1/2 triplet被选；四卡 global batch=8 时恰好 4/8。view A/B/C 仅在长期统计上均匀，不强制逐 batch平衡，也不使用 view-specific规则。

## 4. 一个 selected triplet 产生的三种 receiver 场景

假设 B view 被遮挡：

- receiver B：local weak，两个 remote A/C strong；
- receiver A：local strong，remote B weak、C strong；
- receiver C：local strong，remote A strong、B weak。

因此一次 exactly-one-view degradation 同时提供 receiver-weak/remote-strong 与 receiver-strong/remote-weak。未选中的 50% triplet保持原 E3 common-visible 输入，防止训练完全偏离原分布。

## 5. E3 actor 数据流

```text
original synchronized ABC batch
  -> train-only D1 clone(search_images)
  -> selected target bbox neutral occlusion
  -> frozen local B0 forward once/view
  -> local raw CENTER score + local K=8 prompt
  -> other two local prompts mapped to receiver
  -> trainable E3 adapter + frozen CENTER head
  -> unchanged tracking loss
```

被遮挡 view 的 local feature、raw response 和 K=8 prompt允许自然退化；不人为 mask prompt、不把 weak-view identity传给 adapter、不新增 loss。GT bbox 只用于训练期图像增强和原 tracking supervision；它不进入 E3 adapter input或 inference。

## 6. Validation 与 inference identity

- `network.training=false` 时 degradation 必须 exact bypass，返回原 data/search tensor；
- `TEST` 配置完全继承 E3，不存在 D1 runtime字段；
- Safe Commit仍为 collaborative report、Local persistent state/next crop/local sender prompt；
- official/outer test 不运行；
- D1 checkpoint 将来只能先在授权的 inner-val/inner-dev评价。

## 7. 日志与数据流审计

训练 forward 记录：

- `E3D1/enabled`、`E3D1/applied`；
- local/global batch 的 selected triplet count/ratio；
- weak view A/B/C count；
- exactly-one-view violation count；
- clipped/invalid bbox count；
- occluded pixel count与占 search 面积比例；
- unchanged template/bbox supervision flags。

validation 应记录 `E3D1/applied=0`、selected=0。日志只用于协议核验，不参与 loss、gate 或模型选择。

## 8. Smoke acceptance

长期训练前必须全部 PASS：

1. default-off 与原 E3 search tensor exact identity；
2. eval/validation 即使 config enabled 也 exact identity；
3. B=2 时恰好 1 个 triplet degraded；
4. 每 selected triplet恰好一个 weak view；
5. 非 selected triplet和另外两个 view逐元素不变；
6. 只改变 bbox 内 search pixels，template与annotation不变；
7. fixed fill=0、bbox clipping与finite检查正确；
8. E3 K=8、remote mapping与CENTER output shape不变；
9. forward/backward/loss finite；
10. optimizer仍仅包含 E3 adapter 148,993 parameters；
11. validation forward不应用 D1；
12. E3/V1既有回归通过；
13. checkpoint save/load smoke通过；
14. 不访问 `threemdot_test`。

## 9. 未来训练命令与停止点

本任务只完成预注册、实现、数据流审计和最小 smoke，不启动25-epoch训练。若 smoke PASS，未来单独授权后从同一 B0 checkpoint fresh-initialize E3 adapter，使用独立 run id训练 E3-D1。

完成实现报告后 STOP；不运行 official test，不根据 smoke或历史 test调整上述常数。
