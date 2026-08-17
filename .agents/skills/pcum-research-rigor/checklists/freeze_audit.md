# Freeze Audit Checklist

Use this for PCUM-only fine-tuning, ranking-loss experiments, smoke tests, and attribution claims.

## Required Strict-Freeze Conditions

A strict PCUM-only checkpoint must satisfy:

- backbone changed keys = 0
- box_head/head changed keys = 0
- box_head BatchNorm `running_mean` changed keys = 0
- box_head BatchNorm `running_var` changed keys = 0
- box_head BatchNorm `num_batches_tracked` changed keys = 0
- optimizer contains no backbone parameters
- optimizer contains no box_head/head parameters
- only PCUM/fusion/prompt/residual parameters changed

## Training-Time Assertions

Training startup should print or assert:

- trainable parameter count
- frozen backbone count
- frozen head count
- PCUM trainable parameter count
- optimizer parameter names or grouped names
- frozen BatchNorm modules are in eval mode
- no backbone/head key appears in optimizer groups

If any backbone/head optimizer key appears, stop the run.

## BatchNorm Buffer Protection

`requires_grad=false` is not enough for frozen heads with BatchNorm. Confirm:

- frozen backbone/head modules are set to eval mode
- they remain eval after every `model.train()` call
- buffers do not change in checkpoint diff

## Attribution Language

Use these verdicts:

- If backbone/head parameters changed: not strict PCUM-only; do not use as clean PCUM ranking evidence.
- If only PCUM/fusion/prompt changed but T1 changes: T1 path likely still uses PCUM/fusion; clarify local-only definition.
- If only PCUM/fusion/prompt changed and T1 should not depend on them: audit config, tracker_param, and checkpoint path.

## Report Fields

Record:

- base checkpoint path and stored_epoch
- fine-tuned checkpoint path and stored_epoch
- state_dict key count
- identical tensor count
- changed tensor count
- changed keys by group
- L2 diff, max abs diff, relative diff by group
- top changed tensors
- optimizer group summary
- final verdict
