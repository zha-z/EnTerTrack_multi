"""Strict CSV schema for E3 prediction-only runtime diagnostics."""

import csv


TARGET_PROMPT_DIAGNOSTIC_COLUMNS = (
    "frame_id",
    "target_id",
    "receiver_view",
    "prompt_k",
    "sender_view_0",
    "sender_view_1",
    "sender_0_prompt_norm",
    "sender_1_prompt_norm",
    "sender_0_topk_score_mean",
    "sender_0_topk_score_min",
    "sender_0_topk_score_max",
    "sender_1_topk_score_mean",
    "sender_1_topk_score_min",
    "sender_1_topk_score_max",
    "residual_norm",
    "relative_residual_norm",
    "residual_scale",
    "valid_remote_count",
    "used_remote",
    "reported_output_source",
    "state_output_source",
    "sender_prompt_source",
    "payload_fp32_bytes_per_sender",
    "payload_fp16_bytes_per_sender",
    "uses_gt",
    "persistent_state_digest_before",
    "persistent_state_digest_after_collaboration",
    "persistent_state_digest_after_commit",
    "next_crop_state_digest",
)


def save_target_prompt_diagnostics(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TARGET_PROMPT_DIAGNOSTIC_COLUMNS,
            extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def target_prompt_diagnostic_file(base_results_path):
    return "{}_target_prompt_e3.csv".format(base_results_path)
