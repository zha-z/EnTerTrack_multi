# Three-MDOT 同口径 OSTrack 评测对比

## 结论

三组已有 bbox 预测均在 2026-08-18 使用同一份当前代码离线重算。`B0-ABC-Plain ep25` 的 OSTrack AUC 为 **48.9464**，比 `Current baseline ep25` 高 **0.7944** 个百分点，比 `Original ep21` 高 **1.7450** 个百分点。

| 模型 | Run ID | AUC | Precision | Normalized Precision | Mean IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0-ABC-Plain ep25 | 26025 | **48.9464** | **64.6659** | **77.6050** | **56.4467** |
| Current baseline ep25 | 16025 | 48.1520 | 63.6990 | 77.2557 | 55.5964 |
| Original ep21 | 15021 | 47.2014 | 63.7230 | 76.2913 | 54.9921 |

按视角的 AUC：

| 模型 | A | B | C |
| --- | ---: | ---: | ---: |
| B0-ABC-Plain ep25 | **46.7075** | **48.8976** | **51.2340** |
| Current baseline ep25 | 45.6307 | 48.7729 | 50.0524 |
| Original ep21 | 44.8338 | 46.7965 | 49.9739 |

## 冻结评测口径

- 实现：`lib/test/analysis/extract_results.py::calc_seq_err_robust`
- success thresholds：0.00 到 1.00，步长 0.05
- success comparator：严格 `IoU > threshold`
- 首帧：预测框重置为初始化 GT
- visibility：使用数据集 `target_visible`
- `exclude_invalid_frames=false`
- 汇总：先计算每个序列曲线，再对 105 个序列进行宏平均
- A/B/C 是独立序列结果，不是跨视角融合结果

Mean IoU 是附加诊断：每个序列只对 visible-valid 帧求均值，再对存在 valid 帧的序列求 `nanmean`。它不参与 AUC 计算。

## 可复现命令

以下命令只读取已经保存的预测文件，不重新运行 tracker：

```bash
cd /data/zjy/EnTeR-Track-main

PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-ostrack-comparison \
/home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py \
  --tracker_name entertrack \
  --tracker_param b0_abc_plain_4gpu \
  --dataset_name threemdot_test \
  --runid 26025 \
  --output_dir /tmp/ostrack_same_standard/b0_abc_plain_ep25

PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-ostrack-comparison \
/home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py \
  --tracker_name entertrack \
  --tracker_param pcum_ablation_current_baseline_ep0025_eval \
  --dataset_name threemdot_test \
  --runid 16025 \
  --output_dir /tmp/ostrack_same_standard/current_baseline_ep25

PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-ostrack-comparison \
/home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py \
  --tracker_name entertrack \
  --tracker_param entertrack_threemdot_original_ep21_eval \
  --dataset_name threemdot_test \
  --runid 15021 \
  --output_dir /tmp/ostrack_same_standard/original_ep21
```

## 文件

- `comparison_metrics.csv`：三组整体指标与相对提升。
- `comparison_summary.json`：机器可读的同口径结果、协议和本地预测来源。
- 每个模型子目录中的 `summary.json`：整体、逐 target、逐视角结果。
- 每个模型子目录中的 `sequence_metrics.csv`：105 个独立序列指标。

本目录没有提交 checkpoint 或逐帧 bbox 文件；这些指标是从本地冻结预测只读生成的。没有根据 Three-MDOT test 结果重新选择 checkpoint 或调整超参数。
