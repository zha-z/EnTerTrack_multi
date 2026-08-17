#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
CONFIG="pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t0_disabled"
RUNID=12550
LOG="$ROOT/output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/logs/no_gt_pipeline/ep15_t0_run12550.log"

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" tracking/test.py \
    --tracker_name entertrack \
    --tracker_param "$CONFIG" \
    --dataset_name threemdot_test \
    --runid "$RUNID" \
    --threads 12 \
    --num_gpus 6 \
    > "$LOG" 2>&1

"$PYTHON" tracking/verify_pcum_test_run.py \
    --config "$CONFIG" \
    --dataset threemdot_test \
    --runid "$RUNID" \
    --log "$LOG" \
    --expected-source tracker
