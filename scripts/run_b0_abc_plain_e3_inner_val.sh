#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
CONFIG="b0_abc_plain_ind_sampler_4gpu_ep50"
RUN_DIR="$ROOT/output/diagnostics/b0_abc_plain_ind_sampler_long/e3_run_20260818_seed42_4gpu_r001"
CKPT_DIR="$RUN_DIR/checkpoints/train/entertrack/$CONFIG"
LOG_DIR="$RUN_DIR/inner_val_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

for epoch in 30 35 40 45 50; do
    printf -v epoch4 "%04d" "$epoch"
    runid=$((28300 + epoch))
    checkpoint="$CKPT_DIR/EnTeRTrack_ep${epoch4}.pth.tar"
    log="$LOG_DIR/ep${epoch4}_run${runid}.log"

    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
        MPLCONFIGDIR="/tmp/mpl-e3-val-${runid}" \
        "$PYTHON" tracking/test.py \
        --tracker_name entertrack \
        --tracker_param "$CONFIG" \
        --dataset_name threemdot_val \
        --runid "$runid" \
        --threads 12 \
        --num_gpus 4 \
        --checkpoint "$checkpoint" \
        --fail_if_results_exist 1 \
        --no_gt_inference 1 \
        >"$log" 2>&1

    PYTHONPATH=. MPLCONFIGDIR="/tmp/mpl-e3-verify-${runid}" \
        "$PYTHON" tracking/verify_pcum_test_run.py \
        --config "$CONFIG" \
        --dataset threemdot_val \
        --runid "$runid" \
        --log "$log" \
        --expected-source none \
        --expected-checkpoint "$checkpoint" \
        --expected-epoch "$epoch" \
        --require-no-pcum
done
