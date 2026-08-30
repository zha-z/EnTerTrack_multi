# Three-MDOT visibility schema 静态审计

## 结论

Three-MDOT train / val 每个 sequence 的原始可用字段只有：

- `groundtruth.txt`：每帧 bbox `x,y,w,h`；
- `occlusion.txt`：每帧二值标记；
- `out_of_view.txt`：每帧二值标记。

原始 annotation 中没有名为 `visible`、`valid`、`target_visible`、`absence`、`partial_visible` 或 `full_occlusion` 的独立文件。训练 dataset 在加载时计算：

```text
valid[t] = bbox_w[t] > 0 AND bbox_h[t] > 0
visible[t] = (occlusion[t] == 0)
             AND (out_of_view[t] == 0)
             AND valid[t]
```

所以 `visible=1` 的准确含义是“没有被两个二值 annotation 标为 occlusion/out-of-view，且 bbox 尺寸有效”，不能解释为“完全无遮挡”。

## 代码证据

1. Split 与 dataset：[`lib/train/dataset/threemdot.py`](../../../lib/train/dataset/threemdot.py) 第 54–71 行从 `threemdot_train.txt` / `threemdot_val.txt` 构建 sequence list；[`lib/train/base_functions.py`](../../../lib/train/base_functions.py) 第 75–80 行把 `THREEMDOT` / `THREEMDOT_VAL` 映射为 `ThreeMDOT(split='train'/'val')`。
2. 原始字段：`threemdot.py` 第 108–111 行读取 `groundtruth.txt`；第 119–136 行读取 `occlusion.txt` 和 `out_of_view.txt` 并构造 annotation visibility。
3. `valid` 与最终 `visible`：`threemdot.py` 第 145–152 行定义 `valid=(w>0)&(h>0)`，再以 `visible=_read_target_visible(...) & valid.byte()` 返回 `{'bbox','valid','visible'}`。
4. sampler 实际读取：[`lib/train/data/sampler_threemdot.py`](../../../lib/train/data/sampler_threemdot.py) 第 526–536 行读取 `seq_info_dict['visible']`；第 542–558 行按同一 target 找另外两个 view 的 `visible`。
5. common-visible：`sampler_threemdot.py` 第 124–134 行对三视角截到最短长度后逐元素 AND；第 268–275 行在 `require_all_views_visible` 时把该 mask 同时设为 `visible_for_sampling` 和 `valid_for_sampling`。
6. causal frame sampling：`sampler_threemdot.py` 第 296–314 行从上述 common mask 选 template/base 和更晚的 search frame；第 330–344 行再次要求 template 与 search 的三视角 flag 全为 true。
7. 同步：`sampler_threemdot.py` 第 357–382 行给 A/B/C 使用同一 template/search frame id；第 396–417 行保留 `template_view_valid`、`search_view_valid` 和 common-visible count 元数据。
8. dataloader 传参：[`lib/train/base_functions.py`](../../../lib/train/base_functions.py) 第 304–314 行读取 `REQUIRE_ALL_VIEWS_VISIBLE`、canonical order 和 independent sampling；第 342–350、371–379 行对 train/val 都构造同一个 `TrackingSamplerThreeMDOT` 逻辑。
9. E3 配置与硬约束：[`experiments/entertrack/target_prompt_collaboration_e3.yaml`](../../../experiments/entertrack/target_prompt_collaboration_e3.yaml) 第 36–42 行固定 `REQUIRE_ALL_VIEWS_VISIBLE=true`；[`lib/train/actors/target_prompt_collaboration.py`](../../../lib/train/actors/target_prompt_collaboration.py) 第 48–57 行拒绝非 common-visible 或非三同步视角的输入。
10. D1 发生位置：`target_prompt_collaboration.py` 第 58–63 行在 sampler 已产生 common-visible batch 后调用 D1；[`lib/train/target_prompt_asymmetric_degradation.py`](../../../lib/train/target_prompt_asymmetric_degradation.py) 第 54–135 行只 clone search pixels，并在半数 triplet 中随机选择恰好一个 view 覆盖 target-region，annotation 不变。

因此当前 `REQUIRE_ALL_VIEWS_VISIBLE=true` 的最终 visibility 表达式是：

```text
common[t] = visible_A[t] AND visible_B[t] AND visible_C[t]

visible_v[t] = (occlusion_v[t] == 0)
               AND (out_of_view_v[t] == 0)
               AND (bbox_w_v[t] > 0)
               AND (bbox_h_v[t] > 0)
```

template 与 search frame 都必须满足 `common=true`；此外 causal 模式还要求 search frame 晚于 template frame，并受最大采样间隔控制。

## 本地 annotation schema 复验

| 项目 | train | val |
|---|---:|---:|
| targets | 23 | 5 |
| sequences | 69 | 15 |
| 读取的 annotation files | 207 | 45 |
| occlusion 值域 | `{0,1}` | `{0,1}` |
| out_of_view 值域 | `{0}` | `{0}` |
| 同步范围内 bbox invalid | 0 | 0 |
| A/B/C 长度不一致导致丢弃的尾帧 | 0 | 0 |

在本轮 train/val split 中，所有 `visible=false` 都由 `occlusion==1` 产生；没有观测到 `out_of_view==1` 或 bbox invalid。该事实只描述本地这两个 split，不能外推到未访问的 test。

## MISSING_DIAGNOSTIC

- `occlusion.txt` 没有部分遮挡/完全遮挡强度分级；仓库没有给出可据此重建该分级的 schema。
- 没有独立 `absence` 字段。
- 没有 raw `target_visible` 字段；这里的 visibility 是 dataset 计算量。
- 因此只能报告“sender 被 `occlusion` 二值标记”，不能判断它对应 partial occlusion、full occlusion、scene-specific failure 或其他更细原因。
