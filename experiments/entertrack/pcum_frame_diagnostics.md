# PCUM Frame Diagnostics

## Semantics

Both candidates start from the same committed tracker state. `local_bbox` is
predicted without remote prompts, while `raw_collaborative_bbox` is predicted
with the available remote prompts. Therefore:

- `instant_delta_iou = raw_collaborative_iou - local_iou` measures the immediate
  one-frame prompt effect under the same history.
- `fallback_delta_iou = final_iou - raw_collaborative_iou` measures the recovery
  introduced by the existing fallback decision.
- `final_delta_iou = final_iou - local_iou` measures the deployed one-frame effect.

These values do not compare two independently evolving long-term tracker
trajectories.

## Configurations

- `pcum_diagnostic_reproduction_oracle_mask`: film checkpoint reproduction with
  the GT visibility mask. CSV rows are marked `reproduction_oracle_gt_visible_mask`.
  This is oracle analysis and must not be reported as a formal result.
- `pcum_diagnostic_formal_no_gt_mask`: identical configuration except
  `TEST.PCUM.USE_REMOTE_VISIBLE_MASK=False`. Visibility annotations are read only
  after inference for grouped analysis.

Both use `CHECKPOINT_NAME: pcum_ablation_current_full` and `FUSION: film`.
Film has no gated-add fusion gate, so `fusion_gate_*` columns are empty/NaN.

## Commands

```bash
python tracking/test.py --tracker_name entertrack --tracker_param pcum_diagnostic_reproduction_oracle_mask --dataset_name threemdot_test --runid 301 --threads 12 --num_gpus 6
python tracking/test.py --tracker_name entertrack --tracker_param pcum_diagnostic_formal_no_gt_mask --dataset_name threemdot_test --runid 302 --threads 12 --num_gpus 6
```

```bash
python tracking/analyze_pcum_frame_diagnostics.py --tracker_param pcum_diagnostic_reproduction_oracle_mask --runid 301
python tracking/analyze_pcum_frame_diagnostics.py --tracker_param pcum_diagnostic_formal_no_gt_mask --runid 302
```

Each worker writes one CSV per sequence and UAV. The analysis script merges the
files, writes UAV and sequence summaries, selects the ten largest positive and
negative sequences, and renders four-panel temporal plots.
