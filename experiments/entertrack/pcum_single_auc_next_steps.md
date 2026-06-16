# PCUM Single-Drone AUC Next Steps

Goal: improve Drone A/B/C single-drone AUC, especially pushing all three above
0.50, while keeping the fused result above the current best target-view run.

## Current Best Checkpoint

The best current result is `pcum_real_target_stable` run id 1, tested at epoch 28:

- Drone A AUC: 0.4528
- Drone B AUC: 0.4887
- Drone C AUC: 0.5040
- Fused AUC: 0.5999

## Recommended Next Training

Use all views as supervised target views instead of only view A:

```bash
python tracking/train.py --script entertrack --config pcum_real_allviews_stable --save_dir output/pcum_real_allviews_stable --mode multiple --nproc_per_node 6 --use_wandb 0
```

If Drone A is still weak, use the A-focused run. It fixes the training view
order to A/B/C, gives view A a larger loss weight, and randomly drops remote
prompts during training to reduce the train/test mismatch for single-drone
evaluation:

```bash
python tracking/train.py --script entertrack --config pcum_real_allviews_a_focus --save_dir output/pcum_real_allviews_a_focus --mode multiple --nproc_per_node 6 --use_wandb 0
```

## Test

`pcum_real_allviews_stable` defaults to epoch 28. If validation indicates
another epoch is better, change `TEST.EPOCH` in
`experiments/entertrack/pcum_real_allviews_stable.yaml`.

```bash
python tracking/test.py --tracker_name entertrack --tracker_param pcum_real_allviews_stable --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
```

`pcum_real_allviews_a_focus` currently defaults to epoch 40. If epoch 28 or 32
is better on validation, change `TEST.EPOCH` in
`experiments/entertrack/pcum_real_allviews_a_focus.yaml` before testing:

```bash
python tracking/test.py --tracker_name entertrack --tracker_param pcum_real_allviews_a_focus --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
```

## Recompute Result Plots

`tracking/analysis_results.py` already includes `pcum_real_allviews_stable`
and `pcum_real_allviews_a_focus` with run id 0. Recompute after testing:

```bash
python tracking/analysis_results.py
```

## PCUM Effectiveness Analysis

```bash
MPLCONFIGDIR=/tmp/matplotlib-pcum python tracking/analyze_pcum_effectiveness.py --candidate pcum_real_target_stable_28
```

Outputs are written to:

```text
output/analysis/pcum_effectiveness/
```

Key files:

- `tracker_summary.csv`
- `per_sequence_delta.csv`
- `pcum_auc_bars.png`
- `success_curves.png`
- `delta_drone_a.png`
- `delta_drone_b.png`
- `delta_drone_c.png`
- `delta_fused.png`
