#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
LOG_DIR="$ROOT/output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/logs/no_gt_pipeline"

mkdir -p "$LOG_DIR"
cd "$ROOT"

run_test() {
    local config="$1"
    local dataset="$2"
    local runid="$3"
    local label="$4"
    local expected_source="$5"
    local log_file="$LOG_DIR/${label}_run${runid}.log"

    echo "[START] $label config=$config dataset=$dataset runid=$runid"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" tracking/test.py \
        --tracker_name entertrack \
        --tracker_param "$config" \
        --dataset_name "$dataset" \
        --runid "$runid" \
        --threads 12 \
        --num_gpus 6 \
        > "$log_file" 2>&1
    "$PYTHON" tracking/verify_pcum_test_run.py \
        --config "$config" \
        --dataset "$dataset" \
        --runid "$runid" \
        --log "$log_file" \
        --expected-source "$expected_source"
    echo "[DONE]  $label runid=$runid verified=true"
}

run_test \
    pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t1_local_only \
    threemdot_test 12551 ep15_t1 tracker
run_test \
    pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_no_gt_normal \
    threemdot_test 12552 ep15_t2_raw tracker
run_test \
    pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_safe \
    threemdot_test 12553 ep15_t2_safe tracker
run_test \
    pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_zero \
    threemdot_test 12554 ep15_t2_zero tracker
run_test \
    pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_temporal_shuffle \
    threemdot_test 12555 ep15_t2_delay tracker
run_test \
    pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_none \
    threemdot_test 12556 ep15_t2_none none

for epoch in 5 10 15 20 25 30 35 40; do
    printf -v epoch4 "%04d" "$epoch"
    t1_runid=$((16000 + epoch * 10 + 1))
    t2_runid=$((16000 + epoch * 10 + 2))
    run_test \
        "pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep${epoch4}_t1_local_only" \
        threemdot_val "$t1_runid" "val_ep${epoch4}_t1" tracker
    run_test \
        "pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep${epoch4}_t2_raw" \
        threemdot_val "$t2_runid" "val_ep${epoch4}_t2_raw" tracker
done

echo "All no-GT stage-2 and validation tests completed."
