#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
RESULT_DIR="$ROOT/docs/results/b1_abc_arp_controlled_20260819/stage_a"

run_sweep() {
    local experiment="$1"
    local config="$2"
    local checkpoint_dir="$3"
    local runid_prefix="$4"

    for epoch in 26 27 28 29; do
        printf -v epoch4 "%04d" "$epoch"
        local runid="${runid_prefix}${epoch}"
        local checkpoint="$checkpoint_dir/EnTeRTrack_ep${epoch4}.pth.tar"
        local output_dir="$RESULT_DIR/${experiment}/ep${epoch4}"
        local log="$output_dir/run.log"

        mkdir -p "$output_dir"
        test -f "$checkpoint"

        CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
            MPLCONFIGDIR="/tmp/mpl-b1-stage-a-${runid}" \
            "$PYTHON" tracking/test.py \
            --tracker_name entertrack \
            --tracker_param "$config" \
            --dataset_name threemdot_val \
            --runid "$runid" \
            --threads 12 \
            --num_gpus 4 \
            --checkpoint "$checkpoint" \
            --fail_if_results_exist 1 \
            --no_gt_inference 1 \
            >"$log" 2>&1

        PYTHONPATH=. MPLCONFIGDIR="/tmp/mpl-b1-stage-a-verify-${runid}" \
            "$PYTHON" tracking/verify_pcum_test_run.py \
            --config "$config" \
            --dataset threemdot_val \
            --runid "$runid" \
            --log "$log" \
            --expected-source none \
            --expected-checkpoint "$checkpoint" \
            --expected-epoch "$epoch" \
            --require-no-pcum \
            >"$output_dir/verification.txt" 2>&1

        PYTHONPATH=. MPLCONFIGDIR="/tmp/mpl-b1-stage-a-analysis-${runid}" \
            "$PYTHON" tracking/analysis_results.py \
            --tracker_name entertrack \
            --tracker_param "$config" \
            --dataset_name threemdot_val \
            --runid "$runid" \
            --output_dir "$output_dir" \
            >"$output_dir/analysis.log" 2>&1
    done
}

cd "$ROOT"

run_sweep \
    E1 \
    b0_abc_plain_4gpu_ep50 \
    "$ROOT/output/diagnostics/b0_abc_plain_long/e1_run_20260818_seed42_4gpu_r002/checkpoints/train/entertrack/b0_abc_plain_4gpu_ep50" \
    291

run_sweep \
    E3 \
    b0_abc_plain_ind_sampler_4gpu_ep50 \
    "$ROOT/output/diagnostics/b0_abc_plain_ind_sampler_long/e3_run_20260818_seed42_4gpu_r001/checkpoints/train/entertrack/b0_abc_plain_ind_sampler_4gpu_ep50" \
    293
