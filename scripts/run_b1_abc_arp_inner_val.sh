#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
CONFIG="b1_abc_arp_4gpu"
RUN_DIR="$ROOT/output/diagnostics/b1_abc_arp/formal_20260819_seed42_4gpu_r001"
CHECKPOINT_DIR="$RUN_DIR/checkpoints/train/entertrack/$CONFIG"
RESULT_DIR="$ROOT/docs/results/b1_abc_arp_controlled_20260819/B1"

cd "$ROOT"

for epoch in 15 20 25; do
    printf -v epoch4 "%04d" "$epoch"
    runid=$((29400 + epoch))
    checkpoint="$CHECKPOINT_DIR/EnTeRTrack_ep${epoch4}.pth.tar"
    output_dir="$RESULT_DIR/inner_val_ep${epoch4}"
    log="$output_dir/run.log"

    mkdir -p "$output_dir"
    test -f "$checkpoint"

    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
        MPLCONFIGDIR="/tmp/mpl-b1-arp-val-${runid}" \
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

    PYTHONPATH=. MPLCONFIGDIR="/tmp/mpl-b1-arp-verify-${runid}" \
        "$PYTHON" tracking/verify_pcum_test_run.py \
        --config "$CONFIG" \
        --dataset threemdot_val \
        --runid "$runid" \
        --log "$log" \
        --expected-source none \
        --expected-checkpoint "$checkpoint" \
        --expected-epoch "$epoch" \
        --require-no-pcum \
        >"$output_dir/verification.txt" 2>&1

    PYTHONPATH=. MPLCONFIGDIR="/tmp/mpl-b1-arp-analysis-${runid}" \
        "$PYTHON" tracking/analysis_results.py \
        --tracker_name entertrack \
        --tracker_param "$CONFIG" \
        --dataset_name threemdot_val \
        --runid "$runid" \
        --output_dir "$output_dir" \
        >"$output_dir/analysis.log" 2>&1
done
