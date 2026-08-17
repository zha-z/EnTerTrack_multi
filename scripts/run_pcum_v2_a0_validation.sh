#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zjy/EnTeR-Track-main"
PYTHON="/home/user/.conda/envs/zjy/bin/python"
LOG_DIR="$ROOT/output/pcum_v2_a_weighted/logs"
CHECKPOINT="$ROOT/output/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/checkpoints/train/entertrack/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40/EnTeRTrack_ep0015.pth.tar"
RESULT_ROOT="$ROOT/output/test/tracking_results/entertrack"
REFERENCE_DIR="$RESULT_ROOT/pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40_ep0015_t2_no_gt_normal_12552"
SKIP_MEAN=0
if [[ "${1:-}" == "--skip-mean" ]]; then
    SKIP_MEAN=1
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--skip-mean]" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"
"$PYTHON" tracking/generate_pcum_v2_a0_configs.py

run_test() {
    local config="$1"
    local dataset="$2"
    local runid="$3"
    local label="$4"
    local aggregation="${5:-}"
    local temperature="${6:-}"
    local log_file="$LOG_DIR/${label}_run${runid}.log"

    echo "[START] $label config=$config dataset=$dataset runid=$runid"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" tracking/test.py \
        --tracker_name entertrack \
        --tracker_param "$config" \
        --dataset_name "$dataset" \
        --runid "$runid" \
        --threads 12 \
        --num_gpus 6 \
        > "$log_file" 2>&1

    verify_args=(
        --config "$config"
        --dataset "$dataset"
        --runid "$runid"
        --log "$log_file"
        --expected-source tracker
        --expected-checkpoint "$CHECKPOINT"
        --expected-epoch 15
    )
    if [[ -n "$aggregation" ]]; then
        verify_args+=(
            --expected-aggregation "$aggregation"
            --expected-temperature "$temperature"
        )
    fi
    "$PYTHON" tracking/verify_pcum_test_run.py "${verify_args[@]}"
    echo "[DONE] $label"
}

# Stage A: default mean must reproduce the official no-GT raw trajectories.
if [[ "$SKIP_MEAN" != "1" ]]; then
    run_test pcum_v2_a0_mean_ep0015 threemdot_test 18052 mean_compatibility
fi
CANDIDATE_DIR="$RESULT_ROOT/pcum_v2_a0_mean_ep0015_18052"
"$PYTHON" tracking/compare_tracking_bboxes.py \
    --dataset threemdot_test \
    --reference-dir "$REFERENCE_DIR" \
    --candidate-dir "$CANDIDATE_DIR"

# Stage B: validation-only raw weighted sweep.
run_test pcum_v2_a0_softmax_t010_ep0015 threemdot_val 18101 \
    softmax_t010 confidence_softmax 0.1
run_test pcum_v2_a0_softmax_t025_ep0015 threemdot_val 18125 \
    softmax_t025 confidence_softmax 0.25
run_test pcum_v2_a0_softmax_t050_ep0015 threemdot_val 18150 \
    softmax_t050 confidence_softmax 0.5
run_test pcum_v2_a0_softmax_t100_ep0015 threemdot_val 18200 \
    softmax_t100 confidence_softmax 1.0
run_test pcum_v2_a0_sigmoid_t025_ep0015 threemdot_val 18225 \
    sigmoid_t025 confidence_sigmoid 0.25

echo "A0 mean compatibility and raw validation sweep completed."
