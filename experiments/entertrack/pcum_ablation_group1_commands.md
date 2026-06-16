# PCUM Ablation Group 1

These configs are copied from `entertrack_threemdot.yaml` and only change PCUM-related options. The original baseline config is not modified.

## Configs

| Name | Config | PCUM | Fusion | Pseudo Remote Alignment |
| --- | --- | --- | --- | --- |
| baseline | `pcum_ablation_baseline` | disabled | `gated_add` placeholder | off |
| pcum_local_gated | `pcum_ablation_local_gated` | enabled, local only | `gated_add` | off |
| pcum_local_film | `pcum_ablation_local_film` | enabled, local only | `film` | off |
| pcum_pseudo_align_gated | `pcum_ablation_pseudo_align_gated` | enabled | `gated_add` | on |
| pcum_pseudo_align_film | `pcum_ablation_pseudo_align_film` | enabled | `film` | on |

## Important

Older runs under `output/pcum_ablation_group1` were produced before PCUM was connected to `EnTeRTrack.forward_head()`. Those runs are baseline-equivalent and should not be used as PCUM ablation results. Re-run this group into `output/pcum_ablation_group1_v2`.

## Training Commands

Use `--use_wandb 0` as the lightweight/debug-friendly mode supported by this project.

```bash
python tracking/train.py --script entertrack --config pcum_ablation_baseline --save_dir ./output/pcum_ablation_group1_v2 --mode single --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_local_gated --save_dir ./output/pcum_ablation_group1_v2 --mode single --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_local_film --save_dir ./output/pcum_ablation_group1_v2 --mode single --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_pseudo_align_gated --save_dir ./output/pcum_ablation_group1_v2 --mode single --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_pseudo_align_film --save_dir ./output/pcum_ablation_group1_v2 --mode single --use_wandb 0
```

## Test Commands

The test script supports `--debug`; use `--threads 0 --num_gpus 1` for a minimal sequential run.

```bash
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_baseline --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6 --debug 0
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_local_gated --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6 --debug 0
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_local_film --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6 --debug 0
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_pseudo_align_gated --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6 --debug 0
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_pseudo_align_film --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6 --debug 0
```
