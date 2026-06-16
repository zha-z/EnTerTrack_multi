# PCUM Current Ablation Commands

All configs in this group use the current stable training setup:

- `LR=0.00008`
- `EPOCH=40`
- `LR_DROP_EPOCH=28`
- `BATCH_SIZE=8`
- `FLOPS_WEIGHT=0.0`
- `SEARCH_FACTOR=4.0`, `SEARCH_SIZE=256`
- ThreeMDOT all-visible sampling and canonical A/B/C view order

## Ablation Matrix

| Config | Purpose |
| --- | --- |
| `pcum_ablation_current_baseline` | Original tracker, PCUM disabled |
| `pcum_ablation_current_local_view0` | PCUM local prompt, view A only |
| `pcum_ablation_current_local_allviews` | PCUM local prompt with A/B/C supervised, remote always dropped |
| `pcum_ablation_current_real_target` | Real A/B/C remote prompts, only view A supervised |
| `pcum_ablation_current_allviews_equal` | Real remote prompts, A/B/C all supervised with equal weights |
| `pcum_ablation_current_a_weight` | Equal setup plus A-view loss weight `[1.6, 1.0, 1.0]` |
| `pcum_ablation_current_dropout` | Equal setup plus remote dropout `0.3` |
| `pcum_ablation_current_full` | A-view weight plus remote dropout, current recommended full setting |

## Train

```bash
python tracking/train.py --script entertrack --config pcum_ablation_current_baseline --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_current_local_view0 --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_current_local_allviews --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_current_real_target --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_current_allviews_equal --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_current_a_weight --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_current_dropout --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
python tracking/train.py --script entertrack --config pcum_ablation_current_full --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
```

## Test

```bash
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_baseline --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_local_view0 --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_local_allviews --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_real_target --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_allviews_equal --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_a_weight --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_dropout --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_full --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
```

## Smoke Test

```bash
python -m py_compile lib/config/entertrack/config.py lib/train/actors/entertrack_threemdot.py lib/train/base_functions.py lib/train/data/sampler_threemdot.py
python -m unittest tests.test_pcum
```
