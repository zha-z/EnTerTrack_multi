# D2-S1 P50 Source-only Collaboration Training Smoke

## 冻结结论

本轮所有关键 identity、data-flow、initialization 与 gradient gate 均通过。

```text
READY_FOR_D2_S1_LONG_TRAINING = YES
```

该 YES 只表示实现已具备在后续单独授权下启动冻结 25 epoch D2-S1 的条件；本轮没有启动正式训练、没有 optimizer step、没有写 checkpoint、没有访问 `threemdot_test` 或 official test。按任务要求，报告完成后 STOP。

## 范围与 provenance

| 项目 | 结果 |
|---|---|
| branch | `feature/pcum-cross-layer-arp`；未使用 `main` |
| task-start HEAD | `4db13427ba7d33e07826fa06498ff9443e5b9ba2` |
| implementation commit | `0e8febf16cd235e1d5e62ea9f836302a370bc3a9` |
| selected source | P50，coverage 0.5 |
| selected_source SHA256 | `ca8587cd4d0eb8d7293c2e4c762da6f6c76a9612b43ccadf23411626ab912e60` |
| B0 ep25 SHA256 | `363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3` |
| initialization | B0 ep25 strict core load + fresh E3 adapter；9 fresh state keys |
| D1 checkpoint | 未读取、未 resume、未 fine-tune |
| execution | CPU tests + real loader + 1-GPU backward + 2-rank DDP backward |
| forbidden scope | 25ep/train loop/test/official/sweep/gate/selector/loss/tuning 均未执行 |

## 实现边界

- 新增独立 config：`experiments/entertrack/target_prompt_collaboration_e3_d2_s1.yaml`。
- 新增独立 source：`lib/train/target_prompt_d2_s1_source_degradation.py`。
- 历史 D1 config/source 未修改，D1 P100 route fixed-seed bitwise regression PASS。
- actor 只新增 default-off D2 dispatch/audit；E3 和 D1 配置下 D2 exact bypass。
- D1 P100 与 D2-S1 P50 同时启用会 fail closed，避免双重 degradation。
- sampler 文件零改动；继续使用 common-visible、canonical A/B/C、50% triplet、exactly one weak view。

唯一实验变量是 selected weak-view bbox source 从 P100 改成 frozen P50。K=8、adapter architecture、residual/cap、tracking loss、AdamW/LR/epochs/batch/samples-per-epoch、B0 checkpoint 与 Safe Commit 均继承 E3/D1。

## Stable sample identity 与 frozen transform

orientation 从现有 loader metadata 构造的 D2-P1 clean identity 得到：

```text
d2p1-train-<target_id>-<A/B/C 对应 1/2/3>-<search_frame_id:06d>-clean
SHA256("D2-P2-orientation-v1" + NUL + sample_id)
```

实现没有使用 Python `hash()`，也没有使用 RNG 选择 orientation。`torch.randperm`/`torch.randint` 仅原样承担 D1 已冻结的 half-triplet 与 weak-view selection，不参与 P50 orientation。

P50 transform 与 frozen `tracking/target_prompt_d2_p2_partial_degradation.py` 逐像素 identity PASS：coverage 0.5、single contiguous edge-anchored block、fill normalized 0.0、left/right/top/bottom 映射一致。缺失 frame identity、非 canonical view 或冻结常数变化均 fail closed；P25/P75 不可配置。

## Smoke gates

| Gate | Evidence | 结果 |
|---|---|---|
| P50 vs D2-P2 pixel identity | unit + real loader selected sample exact tensor equality | PASS |
| E3 OFF identity | data object/search tensor exact identity | PASS |
| D1 P100 regression | fixed RNG direct route vs actor-compatible D1→D2-bypass bitwise equality | PASS |
| validation bypass | data object/search tensor identity，`applied=false` | PASS |
| exactly-one weak view | B=2：selected 1/2，changed views=1 | PASS |
| actual bbox coverage | real loader 0.5000；GPU/DDP 各 rank 0.5000 | PASS |
| template/annotation | original object identity retained | PASS |
| config freeze | model/sampler/loss/optimizer/epochs/batch/samples/Safe Commit equality tests | PASS |
| real Three-MDOT loader | `TrackingSamplerThreeMDOT`，[3,2,3,256,256]，common-visible canonical ABC | PASS |
| B0 + fresh adapter | frozen checkpoint SHA exact，strict core load，fresh adapter keys=9 | PASS |
| trainable params | only E3 adapter，148,993；frozen core gradient count=0 | PASS |
| GPU forward/backward | loss 1.032065，finite gradients，peak 84,107,264 bytes | PASS |
| DDP smoke | NCCL 2 ranks，backward/all-reduce completed，gradient checksum 两 rank exact | PASS |
| no formal execution | optimizer step/checkpoint/epoch loop/test all false | PASS |

真实 loader 样例为 `d2p1-train-md3005-1-000471-clean`，orientation `left`，requested/realized coverage 均为 `0.5`。该样例只是 data-flow smoke，不用于选择 source 或调参。

CPU 相关回归共 36 tests，failures/errors 均为 0。DDP 两 rank loss 可因各 rank 真实 batch 不同而不同，但 backward 后 adapter gradient checksum 均为 `0.024351375934202224`，证明 DDP all-reduce 已同步完成。

## 未来训练边界

若用户后续单独授权正式 D2-S1，只能从 frozen B0 ep25 core fresh initialize E3 adapter，禁止从 D1/E3 adapter checkpoint resume。正式固定值仍为 25 epochs、per-rank batch 2、Train 6000/epoch、VAL 1500/epoch、adapter LR `8e-5`。本报告不授权训练，也不授权 inner-dev/test 评价。

机器可读结果见 `smoke_summary.json`，文件/commit/hash 见 `provenance.json`，复现命令见 `COMMANDS_ZH.md`。
