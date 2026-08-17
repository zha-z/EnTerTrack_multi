#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
LOG_DIR="$ROOT/output/pcum_v2_a_weighted/logs"
CKPT="$ROOT/output/pcum_v2_a_weighted/checkpoints/train/entertrack/pcum_v2_a_weighted_softmax_t010_ep5/EnTeRTrack_ep0005.pth.tar"

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PATH="/home/user/.conda/envs/zjy/bin:$PATH"

run_eval() {
    local config="$1"
    local runid="$2"
    local source="$3"
    local aggregation="${4:-}"
    local log="$LOG_DIR/${config}_run${runid}.log"
    echo "[VAL] config=$config runid=$runid source=$source aggregation=${aggregation:-none}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" tracking/test.py \
        --tracker_name entertrack \
        --tracker_param "$config" \
        --dataset_name threemdot_val \
        --runid "$runid" \
        --threads 12 \
        --num_gpus 6 \
        2>&1 | tee "$log"

    if [[ -n "$aggregation" ]]; then
        "$PYTHON" tracking/verify_pcum_test_run.py \
            --config "$config" \
            --dataset threemdot_val \
            --runid "$runid" \
            --log "$log" \
            --expected-source "$source" \
            --expected-checkpoint "$CKPT" \
            --expected-epoch 5 \
            --expected-aggregation "$aggregation" \
            --expected-temperature 0.1
    else
        "$PYTHON" tracking/verify_pcum_test_run.py \
            --config "$config" \
            --dataset threemdot_val \
            --runid "$runid" \
            --log "$log" \
            --expected-source "$source" \
            --expected-checkpoint "$CKPT" \
            --expected-epoch 5
    fi
}

run_eval pcum_v2_a_weighted_softmax_t010_ep5_t1_local_only 19051 tracker
run_eval pcum_v2_a_weighted_softmax_t010_ep5_t2_raw 19052 tracker confidence_softmax
run_eval pcum_v2_a_weighted_softmax_t010_ep5_t2_zero 19054 tracker confidence_softmax
run_eval pcum_v2_a_weighted_softmax_t010_ep5_t2_delay 19055 tracker confidence_softmax
run_eval pcum_v2_a_weighted_softmax_t010_ep5_t2_none 19056 none confidence_softmax
