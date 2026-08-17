#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
LOG_DIR="$ROOT/output/pcum_v2_a_weighted/logs"
LOG_FILE="$LOG_DIR/train_ddp6_ep5.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PATH="/home/user/.conda/envs/zjy/bin:$PATH"

echo "[TRAIN] config=pcum_v2_a_weighted_softmax_t010_ep5 save_dir=output/pcum_v2_a_weighted gpus=0,1,2,3,4,5"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" tracking/train.py \
    --script entertrack \
    --config pcum_v2_a_weighted_softmax_t010_ep5 \
    --save_dir output/pcum_v2_a_weighted \
    --mode multiple \
    --nproc_per_node 6 \
    --use_wandb 0 \
    2>&1 | tee "$LOG_FILE"
