# E2B Target Consistency and Cross-view Compatibility Audit

结论：**Case D，停止当前 full-search pooled-prototype feature engineering；不实现 semantic runtime selector，不启动新训练。**

本轮只在 `threemdot_val` 上执行 prediction-only shadow rollout 和 post-hoc target-group OOF 分析。未运行 `threemdot_test`/official test，未训练或修改 backbone、CENTER head、V1 adapter，未改变 cross-attention、residual scale、persistent state、crop 或 bbox。

## 1. 冻结协议与完整性

- 分支：`feature/pcum-cross-layer-arp`。
- 预注册提交：`f72d24716e9344c2bd9a3ba1a8c82aaaa96100a2`，先于 E2B GT join/OOF。
- checkpoint：V1 ep25，SHA256 `0607e8dcd05d9732a0058bb7a3d9e356474dbeb219a65e0472ac83a83d58ad40`。
- rollout：runid `28421`，5 targets / 15 sequences，仅 `threemdot_val`。
- token layout 实测：64 template + 256 search，dim=192；CENTER raw response 为 16×16。
- response-weighted prototype：`softmax(raw_score_map / 1.0)` 加权 256 个 search token；control 为 search global mean；template-conditioned prototype 为当前 local forward 前 64 个 template token 的均值，不称为固定 initial template。
- frame 0 不新增 backbone forward，仅写 15 个 NaN placeholder；活动 prototype 14,097 行全部 finite。
- 每活动帧、每视角仍只做一次 local backbone forward；prototype detach、无梯度、无 GT、无 state write。
- 相对 E1.5：56,448 个 branch row 的 Local/sender0/sender1/Both bbox mismatch=0；15 个保存 bbox 文件 mismatch=0；state mutation=0；`uses_gt=true` row=0。

prediction freeze：

| Artifact | Rows | SHA256 |
|---|---:|---|
| `prediction_only_target_prototypes.npz` | 14,112 | `59406d54758c322150067b0bf556382929b26b449943376b0a077643002f4e23` |
| `prediction_only_target_consistency_features.csv` | 28,224 | `bac418da99df955e29fdeed0e466b53fd585a58c03345e3159cf3c75d4c08cf2` |

只有上述 SHA 写入 `prediction_manifest.json` 后，才连接 E1.5 已冻结的 helpful/harmful/tie 标签。

## 2. 任务结果

Task A（helpful vs harmful，non-tie 4,394 rows，positive prior 0.4461）：

| Feature | ROC-AUC | PR-AUC |
|---|---:|---:|
| E2A T0 current scalar | 0.4281 | 0.4379 |
| E2A T4 temporal full | 0.4014 | 0.4290 |
| weighted S1 sender self | 0.4346 | 0.4452 |
| weighted S2 cross cosine | 0.4021 | 0.4212 |
| weighted S5 primary | **0.3778** | **0.3970** |
| mean S5 control | 0.3753 | 0.3947 |

Task B（helpful vs harmful/tie，26,996 active valid sender rows）：weighted S5 ROC-AUC=0.5172、PR-AUC=0.0799，positive prior=0.0726。虽然略高于随机 ROC，但不足以形成安全 replace-local policy。

Task C sender ranking（1,874 eligible receiver frames）：weighted S5 no-identity ROC-AUC=0.4968；mean S5 no-identity=0.4944。加入 view/pair identity 后反而降至 0.3551/0.3566，且不用于主 policy。

单项中最接近随机上界的是 weighted EMA difference ROC-AUC=0.4996；weighted sender template consistency=0.4529；weighted cross cosine=0.4021。没有稳定的正向 semantic relation。

## 3. OSTrack 标准 OOF tracking

所有 candidate 来自同一个冻结 Safe Commit rollout；selector 只改 reported bbox，下一帧 state/crop 固定为 Local。曲线使用 `calc_seq_err_robust` 等价实现、sequence macro average。

| Variant | AUC | Δ vs Local (point) | Precision | Norm Precision | Mean IoU |
|---|---:|---:|---:|---:|---:|
| Local | 64.8856 | 0.0000 | 0.8407 | 0.8289 | 0.6915 |
| Both Safe | 64.8943 | +0.0086 | 0.8417 | 0.8302 | 0.6917 |
| E2A T4 | 64.8152 | −0.0705 | 0.8423 | 0.8301 | 0.6906 |
| E2B mean S5 | 64.7596 | −0.1260 | 0.8414 | 0.8281 | 0.6902 |
| **E2B weighted S5** | **64.7539** | **−0.1318** | 0.8414 | 0.8282 | 0.6901 |
| GT Oracle-single | 66.2076 | +1.3220 | 0.8445 | 0.8389 | 0.7063 |

weighted S5 的最差 target 是 `md3016`，相对 Local −0.4124 point；未触发预注册的 −5.00 point catastrophic safety gate，但五个 target 中只有 `md3034` 为正（+0.0803 point）。按 view：A −0.3357、B +0.0472、C −0.1068 point。

## 4. 预注册 gates

- Primary：`weighted AUC >= Local +0.30 point`：**FAIL**（实际 −0.1318 point）。
- Semantic：Task A ROC-AUC `>=0.65`：**FAIL**（0.3778）。
- Semantic：PR-AUC > prior：**FAIL**（0.3970 < 0.4461）。
- Representation：weighted ROC 至少高于 mean 0.03：**FAIL**（仅 +0.0025）。
- Representation：weighted tracking 至少高于 mean 0.10 point：**FAIL**（反而 −0.0057 point）。
- Safety：无 target `<= -5.00 point`：**PASS**。

冻结 Case 映射得到 **Case D**：E2B 接近/低于随机，且 response-weighted 与 mean 没有明显差异。Oracle utilization 为 `−9.96%`（按预注册常数）或 `−9.97%`（按 artifact 精确 Local/Oracle），即没有利用 headroom，反而损失。

## 5. Q1–Q10

### Q1：Target self-consistency 是否比 score/APCE/motion 更能预测 sender utility？

**没有形成有意义的提升。** weighted S1 ROC=0.4346，略高于 E2A T4 的 0.4014，也仅略高于 E2A T0 的 0.4281；PR=0.4452 仍低于 positive prior 0.4461。加入 receiver/cross/directional 后 primary S5 反而降到 0.3778。

### Q2：Cross-view target compatibility 是否和 helpful/harmful 有稳定关系？

**否。** weighted cross cosine 单特征 ROC=0.4021，mean cross cosine=0.4036；cross×sender-template interaction 也只有 0.4499/0.4510。关系不能跨 held-out target 泛化，且方向不是稳定的“越相似越 helpful”。

### Q3：response-weighted target prototype 是否优于 global mean feature？

**否。** weighted S5 ROC 仅比 mean 高 0.0025，远低于预注册 +0.03；tracking AUC 反而低 0.0057 point，未通过 representation 双门槛。

### Q4：哪类 semantic feature 最有预测能力？

没有任何一类达到可用水平。nested group 中 weighted S1 最高（ROC=0.4346）；单项的 EMA difference 约 0.50，但不优于随机。不能把其中任何一项称为可靠 semantic feature。

### Q5：能否在 Local、Sender0、Sender1 之间进行可泛化选择？

**不能。** Task B weighted S5 ROC=0.5172、ranking ROC=0.4968；固定 0.5 threshold 的 target-LOTO Safe Report policy 使 AUC 低于 Local。

### Q6：E2B OOF selector AUC 是多少？

primary response-weighted S5 为 **64.7539**；mean control 为 64.7596。

### Q7：相比 E2A=64.8152 是否真正改善？

**没有。** weighted E2B 低 `0.0613` AUC point；mean E2B 低 `0.0556` point。

### Q8：相比 Local=64.8856 是否达到预注册 +0.30 AUC？

**没有。** 实际是 **−0.1318 point**，距门槛还差 0.4318 point。

### Q9：Oracle-single +1.322 AUC 中，E2B 实际利用了多少？

**−9.96%**；selector 为负增益，没有利用 Oracle headroom。

### Q10：下一步应该 A/B/C/D 哪一个？

严格按预注册为 **D：停止当前 collaboration representation 的 pooled-prototype feature engineering**。不实现 semantic selector，不做 high-confidence sparse selection，也不在本任务后自行实现下一阶段。若另行立项，研究方向应是重新设计 remote representation（E3/Target Semantic Prompt），而不是继续给当前 full-search pooled vector 堆特征。

## 6. 研究边界

E2B 研究的是 **cross-view semantic compatibility**，不是 communication trigger；计算 compatibility 仍需先接收 remote prototype。本轮结果不能宣称解决了通信受限，也不能用于 official-test 模型选择。
