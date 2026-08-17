#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
LOG_DIR="$ROOT/output/pcum_v2_a_weighted/logs"
CHECKPOINT="$ROOT/output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/train/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/EnTeRTrack_ep0015.pth.tar"

mkdir -p "$LOG_DIR"
cd "$ROOT"
"$PYTHON" tracking/generate_pcum_v2_a0_configs.py

run_test() {
    local config="$1"
    local runid="$2"
    local label="$3"
    local log_file="$LOG_DIR/${label}_run${runid}.log"

    echo "[START] $label config=$config dataset=threemdot_val runid=$runid"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" tracking/test.py \
        --tracker_name entertrack \
        --tracker_param "$config" \
        --dataset_name threemdot_val \
        --runid "$runid" \
        --threads 12 \
        --num_gpus 6 \
        > "$log_file" 2>&1

    "$PYTHON" tracking/verify_pcum_test_run.py \
        --config "$config" \
        --dataset threemdot_val \
        --runid "$runid" \
        --log "$log_file" \
        --expected-source tracker \
        --expected-checkpoint "$CHECKPOINT" \
        --expected-epoch 15 \
        --expected-aggregation confidence_softmax \
        --expected-temperature 0.1
    echo "[DONE] $label"
}

run_test pcum_v2_a0_softmax_t010_ep0015_zero 18301 softmax_t010_zero
run_test pcum_v2_a0_softmax_t010_ep0015_delay 18302 softmax_t010_delay

echo "A0 best weighted zero/delay validation completed."
