# Result Claim Review Checklist

Use this before writing paper text, final result indexes, LaTeX tables, or experiment conclusions.

## Claim Labels

Every result must be labeled as one of:

- test result
- validation result
- diagnostic result
- failed/negative result

Do not mix labels in the same sentence without explicit wording.

## Current Main Result Boundary

PCUM-v2A A0 weighted raw is the current inference-only main result candidate:

- checkpoint: E4 epoch15
- aggregation: `confidence_softmax`
- temperature: `0.10`
- test runid: `18552`
- test metric: `48.566 / 64.332 / 77.850`

State that A0 weighted is inference-only and did not retrain PCUM.

## Required Comparisons

For PCUM-v2A paper claims, include:

- weighted raw - original EnTeR-Track
- weighted raw - independent non-PCUM baseline
- weighted raw - E4 mean raw
- weighted raw - E4 T1
- weighted raw - weighted zero
- weighted raw - weighted delay
- weighted raw - weighted none

For validation-stage D0/D1 claims, include:

- raw - same-epoch T1
- raw - A0 weighted raw validation
- raw - zero
- raw - none
- raw - delay only as diagnostic when applicable
- negative transfer rate
- A/B/C view metrics

## Negative Transfer

Definition:

```text
negative transfer rate = count(sequence AUC_raw < sequence AUC_T1) / sequence_count
```

If negative transfer is higher than the baseline or exceeds the declared criterion, do not claim the method solves negative transfer.

## Ablation Interpretation

- `zero > raw` means remote prompt semantics or aggregation may be harmful.
- `raw > zero` supports that real remote information is useful.
- `raw > none` supports that collaborative remote prompts add value beyond no remote input.
- `raw > delay` supports synchronization only if train and test delay definitions are aligned.
- In D0/D1 with `LAMBDA_DELAY=0.0`, delay is diagnostic only.

## Negative Diagnostics

Write these plainly:

- B0 deterministic selector failed validation and did not enter test.
- B1 offline selector failed because current prediction-only features were not separable enough.
- C0 enhanced selector failed validation threshold rules.
- D0 original was not strict PCUM-only because head parameters changed.
- D0-fixed-lite epoch2/epoch4 improved raw AUC but failed negative-transfer criteria.
- D1 may proceed only if its validation criteria pass; otherwise no test.

Never turn failed diagnostics into paper main results.
