#!/bin/bash
set -euo pipefail

repo=/data/zjy/EnTeR-Track-main
python=/home/user/.conda/envs/zjy/bin/python
checkpoint="$repo/output/c3r_formal/c1/f1/checkpoints/train/entertrack/entertrack_c3r_c1_f1/EnTeRTrack_ep0006.pth.tar"
output="$repo/output/multi_agent_collaboration_clean/remote_information_sufficiency/raw_rich"

mkdir -p "$output"
cd "$repo"

export PYTHONPATH=.
export TEMPORAL_GATE_REAL_ROLLOUT_AUTHORIZED=1

CUDA_VISIBLE_DEVICES=0 "$python" tracking/generate_c3r_temporal_gate_rollouts.py \
  --execute-authorized-rollout \
  --config entertrack_c3r_temporal_gate_v2_f1 \
  --runid remote_info_train_20260719 \
  --checkpoint "$checkpoint" \
  --split lib/train/data_specs/threemdot/c3r_f1_temporal_v2_inner_train.txt \
  --output "$output/inner_train_prediction_features.jsonl" \
  --counterfactual-diagnostics \
  --remote-information-diagnostics \
  > "$output/inner_train_rollout.log" 2>&1 &
train_pid=$!

CUDA_VISIBLE_DEVICES=1 "$python" tracking/generate_c3r_temporal_gate_rollouts.py \
  --execute-authorized-rollout \
  --config entertrack_c3r_temporal_gate_v2_f1 \
  --runid remote_info_dev_20260719 \
  --checkpoint "$checkpoint" \
  --split lib/train/data_specs/threemdot/c3r_f1_temporal_v2_inner_dev.txt \
  --output "$output/inner_dev_prediction_features.jsonl" \
  --counterfactual-diagnostics \
  --remote-information-diagnostics \
  > "$output/inner_dev_rollout.log" 2>&1 &
dev_pid=$!

status=0
wait "$train_pid" || status=1
wait "$dev_pid" || status=1
exit "$status"
