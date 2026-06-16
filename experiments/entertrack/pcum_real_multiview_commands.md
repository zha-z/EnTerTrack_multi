# PCUM Real Multi-View Experiment

This experiment uses synchronized ThreeMDOT A/B/C views during training.

## Train

```bash
python tracking/train.py --script entertrack --config pcum_real_multiview_film --save_dir output/pcum_real_multiview_v1
python tracking/train.py --script entertrack --config pcum_real_target_film --save_dir output/pcum_real_target_v1
```

## Test

```bash
python tracking/test.py entertrack pcum_real_multiview_film --dataset threemdot_test --threads 8 --num_gpus 1
python tracking/analysis_results.py --tracker_param pcum_real_multiview_film --dataset_name threemdot_test

python tracking/test.py entertrack pcum_real_target_film --dataset threemdot_test --threads 8 --num_gpus 1
python tracking/analysis_results.py --tracker_param pcum_real_target_film --dataset_name threemdot_test
```

## Smoke Test

```bash
python -m unittest tests.test_pcum
python -m py_compile lib/models/entertrack/entertrack.py lib/models/entertrack/pcum.py lib/train/actors/entertrack_threemdot.py lib/train/data/sampler_threemdot.py lib/train/train_script.py
```
