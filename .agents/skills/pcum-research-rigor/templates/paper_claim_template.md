# Paper Claim Template

Use this template to keep claims split-safe and attribution-safe.

## Main Test Claim

`PCUM-v2A A0 weighted raw is an inference-only modification using the E4 epoch15 checkpoint with confidence_softmax aggregation at temperature 0.10. On threemdot_test, runid <runid>, it achieves <AUC>/<Precision>/<Norm Precision>.`

Follow with explicit comparisons:

- vs Original EnTeR-Track epoch21: `<delta>`
- vs independent non-PCUM baseline epoch25: `<delta>`
- vs E4 mean raw: `<delta>`
- vs zero/delay/none ablations: `<delta>`

## Validation Claim

`On threemdot_val, <method> was used only for checkpoint/epoch/threshold selection. It achieved <metrics>, but this is not a test result.`

If it failed a criterion:

`Despite improving <metric>, it did not pass the declared validation gate because <criterion>. Therefore it was not evaluated on threemdot_test and is reported as a diagnostic/negative result.`

## No-GT Claim

`All validation/test inference used tracker-derived remote state with USE_REMOTE_VISIBLE_MASK=false and logs showing uses_gt_visibility=false. GT visibility, target_visible, annotation visibility, oracle masks, and test IoU were not used during inference.`

If training used visibility:

`GT visibility was used only as a training loss mask and was not exposed to validation/test inference.`

## Negative Transfer Claim

`Negative transfer is measured as the fraction of sequences where raw collaborative AUC is below same-setting T1 local-only AUC. <Method> has a negative transfer rate of <rate>, so <it does/does not> support a claim of reduced negative transfer.`

## Ablation Claim

`The zero, none, and delay settings are diagnostic ablations. Raw > zero supports useful non-zero remote prompt semantics; raw > none supports value from remote collaboration; raw > delay is only synchronization evidence when train and test delay definitions are aligned. For D0/D1 runs with LAMBDA_DELAY=0.0, delay is reported only as a diagnostic.`

## Failed Selector Claim

`B1/C0 selector experiments are negative diagnostics: the oracle upper bound was high, but prediction-only selector features were not separable enough, and threshold sweeps did not pass validation gates. They are not used as main results.`
