# ChatGPT 入口：B1-ABC-ARP 受控诊断

请读取本文件所在的 Git 分支，不要切换到 `main`。

- 仓库：`zha-z/EnTerTrack_multi`
- 分支：`feature/pcum-cross-layer-arp`
- 实验提交：`16e3dff89bc69f78469f3c372bb240daade00bd9`
- 实验名称：`B1-ABC-ARP`
- 结果日期：2026-08-19

## 任务目标

这个任务包含两个阶段：

1. Stage A：不重新训练，补测 E1 common-visible 和 E3 independent-view 的 ep26/27/28/29 checkpoint，在 `threemdot_val` 上判断 ep28→29 的 learning-rate drop 是否解释 closed-loop AUC 变化。
2. Stage B：以 E0 B0-ABC-Plain ep25 为固定对照，仅恢复仓库既有 ARP、ATP、token pruning 和 token compensation 训练包。保持 ABC flatten、common-visible sampler、seed 42、4 GPU、batch size 2/GPU、25 epoch 及其他 E0 变量不变。

禁止范围包括 `threemdot_test`、official test、outer holdout、PCUM、C3R、FCVC、MCR、motion、新 sampler、新 loss、新 augmentation 和 ARP 参数搜索。

## 已完成结果

| Model | Epoch | Inner-val AUC | Precision | Norm Precision |
|---|---:|---:|---:|---:|
| E0 B0-ABC-Plain | 25 | 64.885643 | 84.071170 | 82.892541 |
| B1-ABC-ARP | 25 | 60.853262 | 78.740005 | 77.833879 |
| B1 - E0 | | -4.032381 | -5.331166 | -5.058662 |

主要结论：

- E1 的 AUC 在 LR drop 前已经下降；E3 的提升在 LR drop 前已经开始。没有可复现证据证明 `LR_DROP_EPOCH=28` 是关键因素。
- B1 ep25 明显低于 E0 ep25，退化主要集中在 view C 和 view B。
- md3016、md3048 受到明显伤害；md3034、md3055 获益。
- A/B/C 每个训练 epoch 分别为 6000/6000/6000 个样本，全部参与反向传播。
- smoke、正式 25 epoch 训练及 ep15/20/25 inner-val 均已完成。
- 没有运行 Three-MDOT official test。
- 当前建议是保留 Plain backbone，先诊断 ARP 的晚期退化和 compensation 信息恢复质量，不进入 PCUM。

## 请按顺序读取

1. `docs/project_context_for_chatgpt_zh.md`
2. `docs/results/b1_abc_arp_controlled_20260819/REPORT_ZH.md`
3. `docs/results/b1_abc_arp_controlled_20260819/SMOKE_TEST_ZH.md`
4. `docs/results/b1_abc_arp_controlled_20260819/provenance.json`
5. `docs/results/b1_abc_arp_controlled_20260819/network_identity.json`
6. `docs/results/b1_abc_arp_controlled_20260819/b1_inner_val_sweep.csv`
7. `docs/results/b1_abc_arp_controlled_20260819/b1_per_view_metrics.csv`
8. `docs/results/b1_abc_arp_controlled_20260819/b1_per_target_metrics.csv`
9. `docs/results/b1_abc_arp_controlled_20260819/arp_epoch_metrics.csv`
10. `docs/results/b1_abc_arp_controlled_20260819/COMMANDS_ZH.md`

训练和评测的原始日志也已归档在：

- `docs/results/b1_abc_arp_controlled_20260819/B1/training.log`
- `docs/results/b1_abc_arp_controlled_20260819/B1/arp_epoch_metrics.jsonl`
- `docs/results/b1_abc_arp_controlled_20260819/B1/inner_val_ep*/`
- `docs/results/b1_abc_arp_controlled_20260819/stage_a/`

## 给 ChatGPT 的推荐提问

```text
请读取公开仓库 zha-z/EnTerTrack_multi 的
feature/pcum-cross-layer-arp 分支，不要读取 main。

先打开仓库根目录的 CHATGPT_START_HERE_B1_ARP.md，
再按照其中“请按顺序读取”的路径检查代码、报告、CSV、provenance
和 smoke 证据。请基于实际文件回答，不要把 main 分支或 Three-MDOT
official test 结果混入结论。
```
