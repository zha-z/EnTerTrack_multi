# No-GT Inference Checklist

Use this checklist before accepting validation/test results or writing no-GT claims.

## Split Discipline

- `threemdot_val` is for checkpoint, threshold, epoch, and hyperparameter selection.
- `threemdot_test` is for final measurement only after validation passes and the user confirms.
- Never use test outcomes to choose checkpoints, thresholds, epochs, margins, temperatures, or selector settings.
- Never copy validation metrics into a test table or describe validation as test.

## Required Inference Settings

For raw/zero/delay collaborative inference:

```yaml
TEST:
  PCUM:
    REMOTE_STATE_SOURCE: tracker
    USE_REMOTE_VISIBLE_MASK: false
MODEL:
  PCUM:
    REMOTE_AGGREGATION: confidence_softmax
    REMOTE_WEIGHT_TEMPERATURE: 0.10
```

For `none` ablation:

```yaml
TEST:
  PCUM:
    REMOTE_STATE_SOURCE: none
    USE_REMOTE_VISIBLE_MASK: false
```

## Forbidden Inference Inputs

Do not use any of the following during validation/test inference:

- `target_visible`
- GT visibility
- annotation visibility
- oracle mask
- test IoU
- any GT-derived feature, selector input, or fallback mask

## Training-Only Exception

GT visibility may be used for training loss masks, including visible-only ranking, only when:

- It is used only inside the training actor/loss.
- It is not saved as an inference feature.
- Validation/test configs still use no-GT tracker remote state.
- Reports say "visible mask used for training supervision only".

## Result Acceptance Checklist

Record these before accepting a result:

- config name
- checkpoint path
- checkpoint `stored_epoch`
- runid
- split
- no-GT log line, including `uses_gt_visibility=false`
- bbox count and expected sequence count
- prediction length consistency with GT
- remote weight diagnostics count for raw/zero/delay/none
- confirmation that T1 local-only may have zero remote-weight files
- absence of Traceback, RuntimeError, CUDA error, NaN/Inf, OOM, and DDP ready-twice
