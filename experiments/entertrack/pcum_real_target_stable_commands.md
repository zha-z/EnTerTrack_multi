# PCUM Real Target Stable

This run is the next recommended PCUM experiment after the overfitting checks.
It keeps the original tracker path intact and only enables the stricter PCUM
multi-view path through config.

## Main Changes

- Require synchronized ThreeMDOT A/B/C frames to be visible when training PCUM.
- Mask remote prompts with per-view visibility before fusion.
- Use `cosine_confidence` alignment so invalid remote prompts get low confidence.
- Match train/test search crop with `SEARCH.FACTOR=4.0`.
- Enable ARP from the first epoch with `CE_START_EPOCH=0` and `CE_WARM_EPOCH=0`.
- Disable the weak prompt alignment loss for this run.
- Cap the effective PCUM fusion residual scale at `0.03`.

## Train

Single GPU smoke-scale run:

```bash
python tracking/train.py --script entertrack --config pcum_real_target_stable --save_dir output/pcum_real_target_stable --use_wandb 0
```

Six GPUs:

```bash
python tracking/train.py --script entertrack --config pcum_real_target_stable --save_dir output/pcum_real_target_stable --mode multiple --nproc_per_node 6 --use_wandb 0
```

## Test

```bash
python tracking/test.py --tracker_name entertrack --tracker_param pcum_real_target_stable --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
```

## Smoke Test

```bash
python -m unittest tests.test_pcum
python -m py_compile lib/models/entertrack/entertrack.py lib/models/entertrack/pcum.py lib/train/actors/entertrack_threemdot.py lib/train/data/sampler_threemdot.py lib/train/base_functions.py lib/train/train_script.py
```

## Fallback

If the sampler raises `TrackingSamplerThreeMDOT failed after ... retries`, there
are too few frames where all three views are visible under the current split.
Keep `USE_REMOTE_VISIBLE_MASK: True`, but change only:

```yaml
TRAIN:
  PCUM:
    REQUIRE_ALL_VIEWS_VISIBLE: False
```
