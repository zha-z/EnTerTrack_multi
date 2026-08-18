# B0-ABC-Plain 2x2 Smoke Test

所有 smoke 均在 `feature/pcum-cross-layer-arp`、GPU `0,1,2,3`、seed `42` 上运行，输出位于 `/tmp`，没有写入正式 run 目录。

| 检查项 | E1 | E2 | E3 |
|---|---:|---:|---:|
| model forward | PASS | PASS | PASS |
| backward | PASS | PASS | PASS |
| finite loss / gradient | PASS | PASS | PASS |
| A/B/C 等量进入训练 | PASS | PASS | PASS |
| 同组 target_id 一致 | PASS | PASS | PASS |
| 三视角 frame pair 可不同 | N/A（同步采样） | PASS | PASS |
| 每个 view 使用自身 visibility | N/A（共同可见） | PASS | PASS |
| `search_frame > template_frame` | PASS | PASS | PASS |
| search token = 256 | PASS | PASS | PASS |
| Transformer block = 6 | PASS | PASS | PASS |
| PCUM/C3R/remote/pruning absent | PASS | PASS | PASS |
| CENTER map = 16x16、bbox dim = 4 | PASS | PASS | PASS |
| checkpoint save | PASS | PASS | PASS |
| checkpoint resume | PASS | N/A | N/A |
| validation forward | PASS | PASS | PASS |

E2/E3 smoke 的 train manifest 均为 48 行（A/B/C 各 16），val manifest 均为 24 行（A/B/C 各 8）；所有记录的 `template_visible/search_visible=true` 且 `delta_t>0`，`distinct_view_frame_ratio=1.0`。

证据文件见本目录 [`smoke/`](smoke/)；正式训练的逐 epoch 诊断、manifest inventory 和原始训练日志分别保存在 `E1/`、`E2/`、`E3/`。
