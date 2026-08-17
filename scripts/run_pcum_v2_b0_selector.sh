#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
LOG_DIR="$ROOT/output/pcum_v2_b_selector/logs"
CKPT="$ROOT/output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/train/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/EnTeRTrack_ep0015.pth.tar"
RESULT_ROOT="$ROOT/output/test/tracking_results/entertrack"
REFERENCE_DIR="$RESULT_ROOT/pcum_v2_a0_weighted_softmax_t010_ep15_t2_raw_test_18552"
REPORT="$ROOT/output/pcum_v2_b_selector/b0_selector_validation_and_test_report.md"

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PATH="/home/user/.conda/envs/zjy/bin:$PATH"

"$PYTHON" tracking/generate_pcum_v2_b0_configs.py

run_eval() {
    local config="$1"
    local dataset="$2"
    local runid="$3"
    local source="$4"
    local label="$5"
    local log="$LOG_DIR/${label}_run${runid}.log"

    echo "[TEST] config=$config dataset=$dataset runid=$runid source=$source"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" tracking/test.py \
        --tracker_name entertrack \
        --tracker_param "$config" \
        --dataset_name "$dataset" \
        --runid "$runid" \
        --threads 12 \
        --num_gpus 6 \
        2>&1 | tee "$log"

    "$PYTHON" tracking/verify_pcum_test_run.py \
        --config "$config" \
        --dataset "$dataset" \
        --runid "$runid" \
        --log "$log" \
        --expected-source "$source" \
        --expected-checkpoint "$CKPT" \
        --expected-epoch 15 \
        --expected-aggregation confidence_softmax \
        --expected-temperature 0.1
}

run_eval pcum_v2_b0_selector_none_compat_ep15_test threemdot_test 18650 tracker compat_none

"$PYTHON" tracking/compare_tracking_bboxes.py \
    --dataset threemdot_test \
    --reference-dir "$REFERENCE_DIR" \
    --candidate-dir "$RESULT_ROOT/pcum_v2_b0_selector_none_compat_ep15_test_18650"

run_eval pcum_v2_b0_selector_m000_ep15_val threemdot_val 18600 tracker selector_m000_val
run_eval pcum_v2_b0_selector_m002_ep15_val threemdot_val 18602 tracker selector_m002_val
run_eval pcum_v2_b0_selector_m005_ep15_val threemdot_val 18605 tracker selector_m005_val
run_eval pcum_v2_b0_selector_m010_ep15_val threemdot_val 18610 tracker selector_m010_val
run_eval pcum_v2_b0_selector_m015_ep15_val threemdot_val 18615 tracker selector_m015_val
run_eval pcum_v2_b0_selector_m020_ep15_val threemdot_val 18620 tracker selector_m020_val

analysis_output="$("$PYTHON" tracking/analyze_pcum_v2_b0_selector.py \
    --compatibility-ok \
    --output "$REPORT")"
printf '%s\n' "$analysis_output"

selected="$(printf '%s\n' "$analysis_output" | awk -F= '/^SELECTED_MARGIN=/{print $2}' | tail -n 1)"
if [[ -z "$selected" || "$selected" == "none" ]]; then
    echo "[STOP] No selector margin satisfied validation rules. Report written: $REPORT"
    exit 0
fi

case "$selected" in
    0.00) selected_config="pcum_v2_b0_selector_m000_ep15_test" ;;
    0.02) selected_config="pcum_v2_b0_selector_m002_ep15_test" ;;
    0.05) selected_config="pcum_v2_b0_selector_m005_ep15_test" ;;
    0.10) selected_config="pcum_v2_b0_selector_m010_ep15_test" ;;
    0.15) selected_config="pcum_v2_b0_selector_m015_ep15_test" ;;
    0.20) selected_config="pcum_v2_b0_selector_m020_ep15_test" ;;
    *)
        echo "Unsupported selected margin: $selected" >&2
        exit 2
        ;;
esac

run_eval "$selected_config" threemdot_test 18652 tracker selector_selected_test

"$PYTHON" tracking/analyze_pcum_v2_b0_selector.py \
    --compatibility-ok \
    --selected-margin "$selected" \
    --output "$REPORT"

echo "[DONE] B0 selector validation/test pipeline completed. Report: $REPORT"
