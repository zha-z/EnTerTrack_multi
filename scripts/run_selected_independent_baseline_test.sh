#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
CONFIG="pcum_ablation_current_baseline_ep0025_eval"
RUNID=16025
CHECKPOINT="$ROOT/output/pcum_ablation_current/checkpoints/train/entertrack/pcum_ablation_current_baseline/EnTeRTrack_ep0025.pth.tar"
LOG="$ROOT/output/baseline_eval/logs/test_current_baseline_ep0025_run16025.log"

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
    --expected-source tracker \
    --expected-checkpoint "$CHECKPOINT" \
    --expected-epoch 25 \
    --require-no-pcum
