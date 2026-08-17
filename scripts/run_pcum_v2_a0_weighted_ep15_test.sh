#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
LOG_DIR="$ROOT/output/pcum_v2_a_weighted/logs"
CKPT="$ROOT/output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/train/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/EnTeRTrack_ep0015.pth.tar"

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PATH="/home/user/.conda/envs/zjy/bin:$PATH"

run_eval() {
    local config="$1"
    local runid="$2"
    local source="$3"
    local log="$LOG_DIR/${config}_run${runid}.log"
    echo "[TEST] config=$config runid=$runid source=$source aggregation=confidence_softmax"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" tracking/test.py \
        --tracker_name entertrack \
        --tracker_param "$config" \
        --dataset_name threemdot_test \
        --runid "$runid" \
        --threads 12 \
        --num_gpus 6 \
        2>&1 | tee "$log"

    "$PYTHON" tracking/verify_pcum_test_run.py \
        --config "$config" \
        --dataset threemdot_test \
        --runid "$runid" \
        --log "$log" \
        --expected-source "$source" \
        --expected-checkpoint "$CKPT" \
        --expected-epoch 15 \
        --expected-aggregation confidence_softmax \
        --expected-temperature 0.1
}

run_eval pcum_v2_a0_weighted_softmax_t010_ep15_t2_raw_test 18552 tracker
run_eval pcum_v2_a0_weighted_softmax_t010_ep15_t2_zero_test 18554 tracker
run_eval pcum_v2_a0_weighted_softmax_t010_ep15_t2_delay_test 18555 tracker
run_eval pcum_v2_a0_weighted_softmax_t010_ep15_t2_none_test 18556 none
