# E3-D1 ep25 official test 命令

## 推理

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
MPLCONFIGDIR=/tmp/mpl-e3-d1-test-33026 PYTHONUNBUFFERED=1 \
/home/user/.conda/envs/zjy/bin/python tracking/test.py \
  --tracker_name entertrack \
  --tracker_param target_prompt_collaboration_e3_d1 \
  --dataset_name threemdot_test \
  --runid 33026 \
  --threads 8 \
  --num_gpus 4 \
  --checkpoint output/diagnostics/target_prompt_collaboration_e3_d1/run_20260830_seed42_4gpu_r001/checkpoints/train/entertrack/target_prompt_collaboration_e3_d1/EnTeRTrack_ep0025.pth.tar \
  --fail_if_results_exist 1 \
  --no_gt_inference 1
```

## OSTrack 105-sequence macro分析

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl-e3-d1-analysis-33026 \
/home/user/.conda/envs/zjy/bin/python tracking/analysis_results.py \
  --tracker_name entertrack \
  --tracker_param target_prompt_collaboration_e3_d1 \
  --dataset_name threemdot_test \
  --runid 33026 \
  --compare_param b0_abc_plain_4gpu_26025 \
  --output_dir output/analysis/target_prompt_collaboration_e3_d1_ep25_test_33026_vs_b0
```

第二次将 `--compare_param` 改为 `target_prompt_collaboration_e3_28325`，输出到 `_vs_e3` 目录。
