#!/usr/bin/env bash
set -euo pipefail

# PCUM ablation launcher.
# Usage:
#   bash experiments.sh test_local
#   bash experiments.sh test_remote
#   bash experiments.sh test_all
#   bash experiments.sh analysis
#   bash experiments.sh train
#   bash experiments.sh smoke
#
# Override defaults when needed:
#   RUNID_LOCAL=200 RUNID_REMOTE=201 THREADS=12 NUM_GPUS=6 bash experiments.sh test_all

MODE="${1:-test_all}"
DATASET="${DATASET:-threemdot_test}"
THREADS="${THREADS:-12}"
NUM_GPUS="${NUM_GPUS:-6}"
RUNID_LOCAL="${RUNID_LOCAL:-200}"
RUNID_REMOTE="${RUNID_REMOTE:-201}"
RUNID_SMOKE="${RUNID_SMOKE:-299}"
SAVE_DIR="${SAVE_DIR:-output/pcum_ablation_current}"

TRAIN_CONFIGS=(
  pcum_ablation_current_baseline
  pcum_ablation_current_local_view0
  pcum_ablation_current_local_allviews
  pcum_ablation_current_real_target
  pcum_ablation_current_allviews_equal
  pcum_ablation_current_a_weight
  pcum_ablation_current_dropout
  pcum_ablation_current_full
  pcum_ablation_current_full_crosslayer
)

# Local-only evaluation. These configs keep TEST.PCUM.USE_REMOTE disabled.
TEST_LOCAL_CONFIGS=(
  pcum_ablation_current_baseline
  pcum_ablation_current_local_view0
  pcum_ablation_current_local_allviews
  pcum_ablation_current_real_target
  pcum_ablation_current_allviews_equal
  pcum_ablation_current_a_weight
  pcum_ablation_current_dropout
  pcum_ablation_current_full
  pcum_ablation_current_full_crosslayer
)

# Test-time remote-prompt evaluation. These configs load the same checkpoints as
# their non-remote counterparts, but enable A/B/C prompt exchange in testing.
TEST_REMOTE_CONFIGS=(
  pcum_ablation_current_real_target_remote
  pcum_ablation_current_allviews_equal_remote
  pcum_ablation_current_a_weight_remote
  pcum_ablation_current_dropout_remote
  pcum_ablation_current_full_remote
  pcum_ablation_current_full_crosslayer_remote
)

run_train() {
  for cfg in "${TRAIN_CONFIGS[@]}"; do
    python tracking/train.py \
      --script entertrack \
      --config "${cfg}" \
      --save_dir "${SAVE_DIR}" \
      --mode multiple \
      --nproc_per_node "${NUM_GPUS}" \
      --use_wandb 0
  done
}

run_test_local() {
  for cfg in "${TEST_LOCAL_CONFIGS[@]}"; do
    python tracking/test.py \
      --tracker_name entertrack \
      --tracker_param "${cfg}" \
      --dataset_name "${DATASET}" \
      --runid "${RUNID_LOCAL}" \
      --threads "${THREADS}" \
      --num_gpus "${NUM_GPUS}"
  done
}

run_test_remote() {
  for cfg in "${TEST_REMOTE_CONFIGS[@]}"; do
    python tracking/test.py \
      --tracker_name entertrack \
      --tracker_param "${cfg}" \
      --dataset_name "${DATASET}" \
      --runid "${RUNID_REMOTE}" \
      --threads "${THREADS}" \
      --num_gpus "${NUM_GPUS}"
  done
}

run_smoke() {
  python tracking/test.py \
    --tracker_name entertrack \
    --tracker_param pcum_ablation_current_full_remote \
    --dataset_name "${DATASET}" \
    --sequence 0 \
    --runid "${RUNID_SMOKE}" \
    --threads 0 \
    --num_gpus 1
}

run_analysis() {
  python tracking/analysis_pcum_ablation_current.py \
    --dataset_name "${DATASET}" \
    --runid_local "${RUNID_LOCAL}" \
    --runid_remote "${RUNID_REMOTE}"
}

case "${MODE}" in
  train)
    run_train
    ;;
  test_local)
    run_test_local
    ;;
  test_remote)
    run_test_remote
    ;;
  test_all)
    run_test_local
    run_test_remote
    ;;
  smoke)
    run_smoke
    ;;
  analysis)
    run_analysis
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Valid modes: train, test_local, test_remote, test_all, smoke, analysis" >&2
    exit 2
    ;;
esac
