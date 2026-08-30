# D2-P0 Three-MDOT Natural Visibility 与 Receiver-visible Sampling Audit 协议

## 1. 唯一问题与授权边界

本轮只回答：Three-MDOT train / val 的同步 A/B/C 帧中，自然非对称 visibility 有多少；若训练候选从 common-visible 改成 receiver-visible，候选 receiver-frame 覆盖会增加多少。

允许：只读仓库代码、E3/E3-D1 配置与报告、Three-MDOT train/val split、对应 `groundtruth.txt`、`occlusion.txt`、`out_of_view.txt`；写本目录统计产物。

禁止：修改 dataset/sampler/actor/model/YAML；训练、backward、validation tracking rollout、`threemdot_test`、official test、GT-based model selection、augmentation/gate/selector/loss 实现。

## 2. 冻结输入

- branch：`feature/pcum-cross-layer-arp`
- source HEAD：`5b0e6a9b446b08e561c067b52db1bc38b1ec3081`
- dataset root：`/data2/Three-MDOT`
- train split：`lib/train/data_specs/threemdot/threemdot_train.txt`
- val split：`lib/train/data_specs/threemdot/threemdot_val.txt`
- view 映射：sequence suffix `-1/-2/-3` 对应 A/B/C。

不读取 test split，不遍历 test prediction/result。

## 3. Visibility 的冻结定义

严格复现 `ThreeMDOT.get_sequence_info()`：

```text
bbox_valid[t] = (bbox_w[t] > 0) AND (bbox_h[t] > 0)
visible[t] = (occlusion[t] == 0)
             AND (out_of_view[t] == 0)
             AND bbox_valid[t]
```

审计器必须先确认 `occlusion.txt` 和 `out_of_view.txt` 仅包含二值 `0/1`，并确认三种标注逐 view 等长；否则失败停止。

`occlusion.txt` 是仓库实际使用的字段名。仓库和本地 annotation 没有提供 `partial_visible`、部分/完全遮挡分级或独立 `absence` 字段，因此不得把 `visible=1` 解释为“完全无遮挡”，也不得从文件名推断遮挡程度。

## 4. 同步与统计单位

每个 target 必须恰有 A/B/C 三个 sequence。同步 triplet 使用三视角相同的 0-based frame index；同步长度严格复现 sampler `_common_visible()` 的边界：

```text
sync_len = min(len(A), len(B), len(C))
```

三视角长度若不同，超出 `sync_len` 的尾帧不属于同步 triplet，另行记录而不补齐。每个 triplet 最多展开为 3 个 receiver-frame，二者不能混用。

## 5. 冻结指标

### 5.1 Triplet pattern

按 A/B/C 固定顺序统计 `000` 至 `111`；百分比分母为该 split 全部同步 triplet。Visibility level 将八种 pattern 按可见 view 数合并为 3、2、1、0 views visible。

### 5.2 Common-visible

```text
common_visible = A_visible AND B_visible AND C_visible
N_common = count(pattern == 111)
```

这同时是 `REQUIRE_ALL_VIEWS_VISIBLE=true` 对单个 template/search frame id 的最终 visibility 约束。现有 causal sampler 还要求 search frame 晚于 template frame，并受 `MAX_SAMPLE_INTERVAL` 候选窗口影响；本审计统计的是 frame pool，不模拟随机 pair 抽样次数。

### 5.3 Receiver-visible

对 receiver `r in {A,B,C}`：

```text
receiver_admitted(r,t) = visible_r[t]
```

两个 sender 的 visibility 不参与准入。对每个 admitted receiver-frame：

- R2：两个 sender 都 visible；
- R1：恰好一个 sender visible；
- R0：零个 sender visible；
- natural asymmetric：R1 + R0。

```text
N_receiver_total = N_receiver_A + N_receiver_B + N_receiver_C
current_receiver_capacity = 3 * N_common
expansion = N_receiver_total / current_receiver_capacity
extra_receiver_frames = N_receiver_total - current_receiver_capacity
percentage_increase = extra_receiver_frames / current_receiver_capacity * 100
```

若 `N_common=0`，ratio 写为缺失并显式报告，禁止除零替代。

### 5.4 Natural annotation diagnostics

仅按明确字段做描述性、可重叠统计：visible receiver 且至少一个 sender 的 `occlusion==1`；visible receiver 且至少一个 sender 的 `out_of_view==1`；visible receiver 且至少一个 sender bbox invalid。它们不是互斥类别，也不等价于 D1 synthetic target-region occlusion。

### 5.5 Target 与时间分布

- 每 target：同步 triplet、common-visible、A/B/C receiver-visible、receiver 总量、natural asymmetric receiver-frame 及比例。
- asymmetric run：对每个 `target + receiver view`，在同步 frame index 上找连续满足 `receiver visible AND at least one sender invisible` 的最大连续段。
- 汇总：run count、frame count、mean、median、P90、max；P90 使用排序样本上 `0.90 * (n-1)` 位置的线性插值。无 run 时四个长度统计均记 0。

## 6. 一致性门槛

每个 split 必须同时满足：

1. 八种 pattern count 之和等于同步 triplet 数；
2. 四个 visibility level count 之和等于同步 triplet 数；
3. `N_receiver_total` 等于所有 pattern 的 `visible_view_count * frame_count` 之和；
4. 每个 receiver 的 `R2+R1+R0=N_receiver_view`；
5. 所有 asymmetric run length 之和等于 natural asymmetric receiver-frame 数；
6. train/val target 不重叠；每 target 恰好 A/B/C；
7. 未访问 test split、test annotation、test prediction。

任一失败则结果为 `INVALID` 并停止结论。

## 7. 决策规则

- Case A：coverage increase `>=20%` 且 natural asymmetric receiver-frame ratio `>=10%`，优先 Natural Asymmetric Receiver-visible Training。
- Case B：coverage increase `<10%`，不值得重构 sampler，回到 preserve/rescue objective。
- Case C：coverage 增加明显但主要集中在极少数 target，先设计 target-balanced receiver-visible sampler。
- Case D：coverage 增加明显且 asymmetric run 经常连续出现，构成最支持 natural asymmetric training 的时间证据。

统计建议不授权实现。未来若采用 receiver-visible，必须是 receiver-specific loss mask；不能简单把 `REQUIRE_ALL_VIEWS_VISIBLE=false` 后让不可见 receiver 继续计算正常 localization loss。GT sender/receiver visibility 仅可用于 training sampler、loss mask 或 auxiliary supervision，validation/inference 禁止把它作为 remote validity、weight、gate 或 selector。
