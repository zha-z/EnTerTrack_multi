# B1-ABC-ARP Smoke Test

结论：**PASS**。正式训练仅在本 smoke 全部通过后启动。

| # | 检查项 | 结果 | 证据 |
|---:|---|---|---|
| 1 | Backbone 为 ARP/ATP | PASS | `vit_tiny_patch16_224_arp`，dim192/depth6/heads3 |
| 2 | ATP 参数存在且训练 | PASS | 12 个 checkpoint ATP tensor；首 backward 12/12 gradient finite |
| 3 | pruning 发生 | PASS | smoke train mean pruned=106.396；独立推理 smoke 实际移除 5 token |
| 4 | compensation 发生 | PASS | smoke activation ratio=0.415609；threshold 与 compensation gradient 均 nonzero |
| 5 | A/B/C 进入 forward/backward | PASS | smoke 全局 ARP samples=48，A/B/C 数量相等 |
| 6 | loss finite | PASS | train/validation loss 均 finite |
| 7 | gradient finite | PASS | `[ARPGradient] parameters=12 finite=true threshold_nonzero=true compensation_nonzero=true` |
| 8 | checkpoint 含 ATP/ARP key | PASS | 182 tensors，其中 12 个 ATP key；0 个 PCUM key |
| 9 | PCUM/C3R/FCVC 关闭 | PASS | 构建期断言及 checkpoint 审计均通过 |
| 10 | 无 remote state | PASS | remote source none，actor 记录 `remote_state_present=0` |
| 11 | runtime 不使用 GT visibility | PASS | tracking 使用 `--no_gt_inference 1`；训练只用 common-visible sampling 约束 |
| 12 | common-visible sampler | PASS | `INDEPENDENT_VIEW_SAMPLING=false`，`REQUIRE_ALL_VIEWS_VISIBLE=true` |
| 13 | 初始 token 数 | PASS | template=64，search=256 |
| 14 | pruning token 有日志 | PASS | kept/pruned/ratio/threshold/compensation 均写入 epoch JSONL/CSV |
| 15 | CENTER 空间恢复正确 | PASS | 恢复 search=256；backbone `[1,320,192]`；score map `[1,1,16,16]`；bbox `[1,1,4]` |
| 16 | checkpoint 保存 | PASS | smoke ep1 checkpoint 可读，SHA256 `f983ab5ce5c338e6af7b3adfffdd2992ac057a7a7aaae058eb29a51593557a31` |
| 17 | validation 运行 | PASS | smoke validation 完成并输出 finite loss/IoU |

独立推理路径使用正式 ep25 checkpoint 严格加载，`missing=[]`、`unexpected=[]`，`training=false`。物理移除 5 个 search token，保留 251，compensation 激活 5 个位置，恢复后为 256；PCUM/C3R=false，score/bbox finite。机器可读记录见 `inference_path_smoke.json`。

正式训练的 ep25 全局采样检查同样 PASS：train A/B/C 各 6000，validation A/B/C 各 1496。全 25 epoch 均相等，见 `sampling_manifest.csv`。

说明：空间 token 数恢复和 compensation delta 非零，只证明机制被调用及形状正确；不等价于被删 token 的定位信息无损恢复。
