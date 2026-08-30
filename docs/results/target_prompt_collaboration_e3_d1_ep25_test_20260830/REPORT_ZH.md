# E3-D1 Asymmetric Degradation ep25 正式测试报告

## 结论

固定使用 E3-D1 epoch 25 checkpoint，在 `threemdot_test` 完成一次 105/105 序列测试。按照与 B0/E3 完全相同的 OSTrack `calc_seq_err_robust`、A/B/C 独立序列宏平均口径：

- B0 Plain：AUC **48.9464**；
- 原 E3 K8：AUC **48.7651**；
- E3-D1：AUC **48.6930**。

E3-D1 相对 B0 为 **-0.2533 AUC point**，相对原 E3 为 **-0.0720 point**。因此 asymmetric degradation training 没有改善主 tracking AUC；它是受控负结果，不应进入 K、residual、loss、gate或selector的 test-driven调参。

## OSTrack 主指标

| 方法 | AUC | Precision | Norm Precision | Mean IoU | 相对 B0 AUC |
|---|---:|---:|---:|---:|---:|
| B0 Plain ep25 | 48.9464 | 64.6659 | 77.6050 | 56.4467 | — |
| E3 K8 ep25 | 48.7651 | 64.6272 | 77.4222 | 56.2229 | -0.1813 |
| **E3-D1 ep25** | **48.6930** | **64.5827** | **77.3752** | **56.1166** | **-0.2533** |

E3-D1 的 per-view AUC：A `46.5976`、B `48.5341`、C `50.9474`。相对 B0 分别为 `-0.1099/-0.3635/-0.2866 point`；三个 view 均没有提升。相对原 E3 分别为 `+0.0535/-0.2399/-0.0297 point`。

## 配对与不确定性

相对 B0：

- sequence：28 上升、73 下降、4 相同；
- target：9 上升、26 下降；
- target-grouped 100,000 次 bootstrap 95% CI：`[-0.3894, -0.1350] AUC point`；
- 最差 target `md3047`：`-1.5130 point`；最佳 target `md3045`：`+0.2271 point`。

相对原 E3：

- sequence：42 上升、59 下降、4 相同；
- target：16 上升、19 下降；
- target-grouped 95% CI：`[-0.2087, +0.0574] point`，包含 0；
- 最差 target `md3047`：`-1.2770 point`；最佳 target `md3046`：`+0.9483 point`。

因此 D1 对 B0 的下降是方向稳定的；D1 与原 E3 的小差异不足以声明显著变化。

## 原生 Fused secondary 指标

仓库 `three_fuse_extract_results_APCE` 使用每帧 max-score/APCE 在 A/B/C prediction中选择 view：

| 方法 | Fused AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| B0 | 59.5908 | 73.4285 | 56.2971 | 75.7949 | 86.5107 |
| E3 | 59.2296 | 73.3747 | 55.2452 | 75.6613 | 86.1330 |
| **E3-D1** | **59.7025** | **73.9058** | **55.8064** | **76.2530** | **86.0883** |

E3-D1 fused AUC 相对 B0 `+0.1117 point`、相对 E3 `+0.4728 point`。这是 prediction-level selector 的 secondary 结果，不是 E3 feature collaboration 本身的 105-view主指标，也不能据此在 test上继续设计或调整 selector。

## 推理与 Safe Commit 完整性

- runid `33026`，测试进程 exit code 0；
- bbox/time/max-score/APCE/E3 diagnostics 均为 105/105；
- bbox 共 83,349 frames，无缺失、无长度 mismatch；
- E3 diagnostics 83,244 rows，全部 `K=8`、两个 valid sender、`used_remote=true`；
- `uses_gt=true` 行数为 0；sender prompt/state source均为 local；
- collaboration local-state mismatch=0，跨帧 persistent-state continuity mismatch=0；
- A/B/C receiver active rows各 27,748；
- residual relative norm均值 `0.17186`，范围 `0.08360–0.25000`；8,200/83,244 行触及 0.25 cap；
- residual scale固定为 `0.06936443`；
- K8 payload为每 sender FP32 6,144 bytes、FP16 3,072 bytes；
- 逐帧记录平均速度约 78.50 FPS，不是四卡端到端 wall-clock throughput。

D1 相比原 E3 的 residual 更强，并有约 9.85% 帧触及 cap；同时主 AUC没有提升。这支持“训练增强未学得更安全的 remote 使用方式”，而不是 remote path bypass或结果缺失。

## 评测口径与冻结结论

主评测复用 `lib.test.analysis.extract_results.calc_seq_err_robust`：success threshold `0.00–1.00`、步长 `0.05`、严格 `IoU > threshold`、读取 `target_visible`、不排除 invalid frames、105 sequences宏平均。GT只在预测全部生成后用于离线评估。

冻结结论：**E3-D1 FAIL，不优于 B0，也没有优于原 E3。** 不根据本次 official test调整 K、occlusion强度、比例、residual、loss、gate、selector或checkpoint；下一步若继续研究，应先回到新的 inner-dev假设和独立预注册。
