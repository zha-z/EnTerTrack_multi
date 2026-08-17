---
name: pcum-research-rigor
description: Project-local rigor guardrails for EnTeR-Track PCUM multi-UAV tracking experiments. Use when preparing paper results, reviewing claims, generating plots/tables, deciding validation-to-test progression, auditing no-GT inference, managing runids, checking freeze audits, interpreting negative transfer, or documenting PCUM-v2A/B/C/D results and diagnostics.
---

# PCUM Research Rigor

Use this skill before writing claims, running or recommending evaluation, preparing tables/figures, or deciding whether a PCUM experiment can move from validation to test.

## Core Rules

- Keep `threemdot_val` and `threemdot_test` separate.
- Use `threemdot_val` only for checkpoint, threshold, epoch, and hyperparameter selection.
- Use `threemdot_test` only after validation passes the declared criteria and the user explicitly confirms the test run.
- Preserve no-GT inference: `TEST.PCUM.REMOTE_STATE_SOURCE=tracker`, `TEST.PCUM.USE_REMOTE_VISIBLE_MASK=false`, and no `target_visible`, GT visibility, annotation visibility, oracle mask, or test IoU in inference.
- Training losses may use GT visibility masks only as training supervision; never describe that as an inference feature.
- Never overwrite existing runids. Update the registry before proposing or running new evaluations.
- Verify `stored_epoch`, config, checkpoint path, runid, split, no-GT logs, bbox completeness, and remote weight diagnostics before accepting a result.
- Treat negative diagnostics as evidence, not as main results.

## Required References

Read only the relevant checklist/template for the task:

- No-GT inference or validation/test eligibility: [checklists/no_gt_inference.md](checklists/no_gt_inference.md)
- Paper claims and result wording: [checklists/result_claim_review.md](checklists/result_claim_review.md)
- Freeze and attribution audits: [checklists/freeze_audit.md](checklists/freeze_audit.md)
- Runid planning: [checklists/runid_registry.md](checklists/runid_registry.md)
- Experiment report drafting: [templates/experiment_report_template.md](templates/experiment_report_template.md)
- Paper paragraph/table claim drafting: [templates/paper_claim_template.md](templates/paper_claim_template.md)

## PCUM-Specific Interpretation

- Negative transfer rate is the sequence-level proportion where raw collaborative AUC is lower than the same-setting T1 local-only AUC.
- `zero`, `delay`, and `none` are causal/diagnostic ablations. State their exact split, checkpoint, and ablation definition.
- D0/D1 delay is diagnostic only when `LAMBDA_DELAY=0.0` or when training delay and test delay are not strictly aligned.
- B1/C0 selector failures are negative diagnostics. Do not repackage them as a main contribution.
- D0-fixed-lite and D1-visible-safe may be considered for test only after validation success criteria pass and freeze audit confirms only PCUM/fusion/prompt changes.

## Output Discipline

When writing a report or claim, explicitly label each result as one of:

- `test result`
- `validation result`
- `diagnostic result`
- `failed/negative result`

If the result is validation-only, do not phrase it as a final test or benchmark result.
