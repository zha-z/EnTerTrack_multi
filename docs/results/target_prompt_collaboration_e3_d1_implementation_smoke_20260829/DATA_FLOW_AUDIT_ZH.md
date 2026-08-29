# E3-D1 数据流审计

## 静态链路

`TRAIN.MULTIVIEW.ENABLED=true` 使 `train_script.py` 选择 `build_dataloaders_threemdot()`。resolved E3-D1 配置保持：

- dataset：train `THREEMDOT`，validation `THREEMDOT_VAL`；
- sampler：`TrackingSamplerThreeMDOT`；
- `REQUIRE_ALL_VIEWS_VISIBLE=true`；
- `CANONICAL_VIEW_ORDER=true`；
- `INDEPENDENT_VIEW_SAMPLING=false`；
- per-GPU batch=2，train/val samples per epoch=6000/1500。

sampler 先按同一 target group 绑定 A/B/C，再取三视角 common-visible mask，并为三视角使用同一 template/search frame id。processing 将三视角 crop 成 template `128x128`、search `256x256`，bbox 转成 crop-space normalized `xywh`。loader 以 `stack_dim=1` 得到 `[V=3,B,...]`；actor 再 view-major flatten 成 `[V*B,...]`。

D1 在这个 `[3,B,...]` 边界工作，因此随机单位是 synchronized triplet，不是 flatten 后的独立 image。只有 search pixel tensor被 clone；loss仍读取原 data 中同一 annotation。

## 真实单批审计

命令见 `COMMANDS_ZH.md`，seed `20260829`。为只读 smoke 临时将 worker=0、train/val sample count=2；没有修改 YAML。

| 检查 | train | validation |
|---|---:|---:|
| search shape | `[3,2,3,256,256]` | `[3,2,3,256,256]` |
| template shape | `[3,2,3,128,128]` | `[3,2,3,128,128]` |
| annotation shape | `[3,2,4]` | `[3,2,4]` |
| view order | `A/A, B/B, C/C` | `A/A, B/B, C/C` |
| all views visible | PASS | PASS |
| selected triplets | 1/2 | 0/2 |
| changed views | exactly 1 | 0 |
| template identity | unchanged | unchanged |
| annotation identity | unchanged | unchanged |
| D1 applied | true | false |

本次 seed 选中 `[view A, batch index 1]`。这只是协议 smoke，不用于 view 选择或调参。审计报告明确 `official_test_accessed=false`。

## 训练/验证开关来源

trainer 在每个 loader 前调用 `actor.train(loader.training)`；D1读取实际 model training state。因此 train loader生效，val loader即使 YAML enabled 也 exact bypass。DDP formal run 每 rank `B=2`，每 rank恰好 1 个 triplet，四卡 global batch 恰好 4/8；view identity由各 rank seeded PyTorch RNG均匀采样。
