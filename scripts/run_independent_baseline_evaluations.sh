#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
LOG_DIR="$ROOT/output/baseline_eval/logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

run_test() {
    local config="$1"
    local dataset="$2"
    local runid="$3"
    local label="$4"
    local checkpoint="$5"
    local epoch="$6"
    local log_file="$LOG_DIR/${label}.log"

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
        --expected-source tracker \
        --expected-checkpoint "$checkpoint" \
        --expected-epoch "$epoch" \
        --require-no-pcum
    echo "[DONE]  $label verified=true"
}

run_test \
    entertrack_threemdot_original_ep21_eval \
    threemdot_test 15021 test_original_entertrack_ep21_run15021 \
    /data/zjy/multi/output/entertrack_single_lasot/checkpoints/train/entertrack/entertrack_threemdot/EnTeRTrack_ep0021.pth.tar \
    21

for epoch in 5 10 15 20 25 30 35 40; do
    printf -v epoch4 "%04d" "$epoch"
    config="pcum_ablation_current_baseline_ep${epoch4}_eval"
    checkpoint="$ROOT/output/pcum_ablation_current/checkpoints/train/entertrack/pcum_ablation_current_baseline/EnTeRTrack_ep${epoch4}.pth.tar"
    runid=$((17000 + epoch))
    run_test \
        "$config" threemdot_val "$runid" \
        "val_current_baseline_ep${epoch4}_run${runid}" \
        "$checkpoint" "$epoch"
done

echo "Original test and independent baseline validation tests completed."
