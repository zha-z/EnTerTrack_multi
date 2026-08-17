# Experiment Report Template

# <Experiment Name> Report

## 1. Summary

- Result label: `<validation result | test result | diagnostic result | failed/negative result>`
- Split: `<threemdot_val | threemdot_test>`
- Main checkpoint: `<path>`
- Main config: `<config name>`
- Main runid: `<runid>`
- No-GT inference: `<source=tracker/none, uses_gt_visibility=false>`
- Decision: `<pass/fail/diagnostic only>`

## 2. Integrity Checks

| Item | Value |
|---|---|
| checkpoint path | `<path>` |
| stored_epoch | `<epoch>` |
| config | `<config>` |
| runid | `<runid>` |
| split | `<split>` |
| bbox completeness | `<N/N>` |
| prediction length consistency | `<ok/fail>` |
| no-GT log | `<source=..., uses_gt_visibility=false>` |
| remote weight diagnostics | `<N/N or N/A>` |
| runtime errors | `<none or details>` |

## 3. Metrics

| Setting | AUC | Precision | Norm Precision |
|---|---:|---:|---:|
| T1 local-only |  |  |  |
| raw weighted |  |  |  |
| zero |  |  |  |
| none |  |  |  |
| delay diagnostic |  |  |  |

## 4. Key Comparisons

| Comparison | Delta AUC | Delta Precision | Delta Norm Precision |
|---|---:|---:|---:|
| raw - T1 |  |  |  |
| raw - A0 weighted raw |  |  |  |
| raw - zero |  |  |  |
| raw - none |  |  |  |
| raw - delay diagnostic |  |  |  |

## 5. Negative Transfer

Definition: sequence-level raw AUC lower than same-setting T1 AUC.

| Setting | Positive/Negative/No-change | Negative transfer rate | Top positive | Top negative |
|---|---:|---:|---|---|
| raw weighted |  |  |  |  |

## 6. A/B/C Views

| Setting | Drone A | Drone B | Drone C |
|---|---|---|---|
| T1 local-only |  |  |  |
| raw weighted |  |  |  |
| zero |  |  |  |
| none |  |  |  |
| delay diagnostic |  |  |  |

## 7. Remote Weight Diagnostics

| Setting | entropy mean/std | max weight mean/p90 | valid count | quality mean/min/max | fallback | selected A/B/C |
|---|---:|---:|---:|---:|---:|---:|
| raw weighted |  |  |  |  |  |  |
| zero |  |  |  |  |  |  |
| none |  |  |  |  |  |  |
| delay diagnostic |  |  |  |  |  |  |

## 8. Freeze Audit Summary

| Check | Result |
|---|---|
| backbone changed keys |  |
| box_head changed keys |  |
| box_head BN buffers changed keys |  |
| optimizer has backbone/box_head |  |
| only PCUM/fusion/prompt changed |  |

## 9. Decision

- Validation/test eligibility: `<eligible/not eligible>`
- Failed criteria: `<list or none>`
- Delay interpretation: diagnostic only unless train/test delay definitions are aligned.
- Next action: `<stop | request user confirmation | run specified validation only>`
