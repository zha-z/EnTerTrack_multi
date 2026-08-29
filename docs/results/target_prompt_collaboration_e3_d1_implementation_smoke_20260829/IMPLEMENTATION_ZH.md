# E3-D1 实现说明

## 唯一变化

新增的 degradation 在 E3 actor 中、frozen local B0 forward 之前执行。开关默认关闭；只有独立 D1 YAML 显式开启，且 `model.training=true` 时才生效。

```text
TrackingSamplerThreeMDOT synchronized ABC
  -> STARKProcessing/crop/normalize
  -> search_images [3,B,3,256,256]
  -> search_anno [3,B,4] normalized xywh
  -> train-only D1 exactly-one-view target occlusion
  -> E3 原有 frozen local forward / K=8 prompt / remote mapping
  -> E3 adapter + frozen CENTER head
  -> 原 tracking loss
```

实现 fail-closed 固定：

- `TRIPLET_RATIO == 0.50`；
- `WEAK_VIEWS_PER_TRIPLET == 1`；
- `OCCLUSION_BOX_SCALE == 1.00`；
- `FILL_VALUE_NORMALIZED == 0.0`；
- view 数必须为 3；
- local batch 必须为正偶数；formal E3 batch=2；
- bbox 必须为 finite、positive，并在 rasterize 后非空。

每个 selected triplet 的同一组三视角 feature 后续既作为 receiver 又作为 sender。若 A 被遮挡，则 receiver A 是 local-weak/remote-strong；receiver B/C 是 local-strong 且 remote 中包含 weak A。因此一个 triplet 同时产生两种非对称场景，且 adapter 不接收 weak-view identity。

## 文件职责

- `lib/train/target_prompt_asymmetric_degradation.py`：train-only occlusion、固定常数校验及审计字段；
- `lib/train/actors/target_prompt_collaboration.py`：只在 frozen local forward 前接入 D1；
- `lib/train/actors/entertrack_threemdot.py`：将 `E3D1/*` 标量写入既有训练日志；
- `lib/config/entertrack/config.py`：新增 default-off 配置 schema；
- `experiments/entertrack/target_prompt_collaboration_e3_d1.yaml`：继承 E3 的独立正式配置；
- `tracking/audit_target_prompt_e3_d1_dataflow.py`：train/val 各一个真实 batch 的只读审计；
- `tests/test_target_prompt_asymmetric_degradation.py`：identity、比例、坐标、冻结配置和真实模型 smoke。

## 日志字段

每个 forward 保留：`enabled/training/applied`、triplet batch、selected count/ratio、weak A/B/C count、exactly-one violation、clipped/invalid bbox、occluded pixels/fraction、template/annotation unchanged。validation 应始终为 `E3D1/applied=0`。

## 未改变项

没有修改 backbone、CENTER head、sampler、loss、K、residual、gate、selector、optimizer、LR、epoch、train/val samples、Safe Commit或 inference。D1 没有被加入 TEST runtime；推理与原 E3完全一致。
