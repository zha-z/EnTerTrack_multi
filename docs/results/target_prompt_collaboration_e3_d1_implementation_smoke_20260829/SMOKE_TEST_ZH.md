# E3-D1 smoke test

## 最终 PASS 清单

| 检查 | 结果 |
|---|---|
| default-off E3 exact identity | PASS |
| D1 validation exact identity | PASS |
| B=2 exactly 1/2 triplet | PASS |
| B=4 exactly 2/4 triplet | PASS |
| selected triplet exactly one view | PASS |
| bbox 内 fill=0、bbox 外不变 | PASS |
| template/annotation identity | PASS |
| normalized bbox rasterize/clipping | PASS |
| odd batch/invalid bbox fail-closed | PASS |
| frozen D1 constants fail-closed | PASS |
| D1 resolved config保持 E3 contract | PASS |
| E3/U1 regression | PASS，28 tests |
| PCUM/actor regression | PASS，90 tests |
| 真实 train/val loader一批 | PASS |
| B0 strict core load + fresh E3 adapter | PASS |
| trainable scope | PASS，9 tensors / 148,993 parameters |
| K/search/CENTER | PASS，K=8 / 256 tokens / 16x16 |
| PCUM/pruning | PASS，均不存在 |
| real model forward/backward/loss finite | PASS |
| optimizer step不改变 frozen core | PASS |
| checkpoint save/load strict | PASS |
| real model validation D1 bypass | PASS |
| official test | 未运行/未访问 |
| long training | 未启动 |

真实模型使用 GPU 0 和 synthetic synchronized ABC input；不读取 dataset test，也不调用 result writer。观测值：train loss `5.8182797432`、validation loss `5.6324410439`、peak CUDA bytes `92,984,320`。这些数仅用于 finite/资源 smoke，不能解释 tracking quality或选择参数。

## 非模型失败与修正

第一次 GPU smoke 在 forward 前因测试错误读取不存在的 `initialization_audit.checkpoint_epoch` 字段而停止；实际 B0 strict load 已成功。测试改为核验 `checkpoint_path`、`strict_full_load` 和 fresh adapter keys 后原样 PASS。

第一次真实 loader 审计在全部数据流断言后，因 JSON serializer不能直接转换“tensor list”形式的 frame id而停止；改成递归只读序列化后同 seed PASS。这两次都未改变 degradation协议、模型或训练配置。
