#!/usr/bin/env bash
set -Eeuo pipefail

cd /data/zjy/EnTeR-Track-main

NAME="pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40"
BASE_CFG="experiments/entertrack/${NAME}.yaml"
LOG_DIR="output/${NAME}/logs"
DATASET_NAME="${DATASET_NAME:-threemdot_test}"

GPUS="${GPUS:-0,1,2,3,4,5}"
NUM_GPUS="${NUM_GPUS:-6}"
THREADS="${THREADS:-12}"

# 重点测试这些 epoch
EPOCHS="${EPOCHS:-5 10 15 20 25 30 35 40}"

mkdir -p "${LOG_DIR}"

python - <<'PY'
from pathlib import Path
import yaml

name = "pcum_supervision_e4_safe_m0_lr8e5_ddp6_ep40"
base = Path(f"experiments/entertrack/{name}.yaml")

with base.open("r", encoding="utf-8") as f:
    cfg0 = yaml.safe_load(f)

epochs = [5, 10, 15, 20, 25, 30, 35, 40]

for ep in epochs:
    for mode in ["t1_local_only", "t2_raw"]:
        cfg = yaml.safe_load(yaml.safe_dump(cfg0, allow_unicode=True))

        cfg["MODEL"]["PCUM"]["ENABLED"] = True
        cfg["TEST"]["SAVE_DIR"] = f"output/{name}"
        cfg["TEST"]["CHECKPOINT_NAME"] = name
        cfg["TEST"]["EPOCH"] = ep

        cfg["TEST"]["PCUM"]["USE_REMOTE_VISIBLE_MASK"] = False

        if mode == "t1_local_only":
            cfg["TEST"]["PCUM"]["USE_REMOTE"] = False
            cfg["TEST"]["PCUM"]["KEEP_LOCAL_IF_REMOTE_WORSE"] = True
        elif mode == "t2_raw":
            cfg["TEST"]["PCUM"]["USE_REMOTE"] = True
            cfg["TEST"]["PCUM"]["KEEP_LOCAL_IF_REMOTE_WORSE"] = False

        out = Path(f"experiments/entertrack/{name}_ep{ep:04d}_{mode}.yaml")
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        print(f"written: {out}")
PY

for ep in ${EPOCHS}; do
  printf -v ep4 "%04d" "${ep}"

  # runid 规则：
  # epoch 5:  12051 / 12052
  # epoch 10: 12101 / 12102
  # ...
  t1_runid=$((12000 + ep * 10 + 1))
  t2_runid=$((12000 + ep * 10 + 2))

  echo
  echo "============================================================"
  echo "[EVAL] epoch ${ep4} T1 local-only"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" \
  PYTHONPATH=. \
  python tracking/test.py \
    --tracker_name entertrack \
    --tracker_param "${NAME}_ep${ep4}_t1_local_only" \
    --dataset_name "${DATASET_NAME}" \
    --runid "${t1_runid}" \
    --threads "${THREADS}" \
    --num_gpus "${NUM_GPUS}" \
    2>&1 | tee "${LOG_DIR}/test_ep${ep4}_t1_run${t1_runid}.log"

  echo
  echo "============================================================"
  echo "[EVAL] epoch ${ep4} T2 raw"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" \
  PYTHONPATH=. \
  python tracking/test.py \
    --tracker_name entertrack \
    --tracker_param "${NAME}_ep${ep4}_t2_raw" \
    --dataset_name "${DATASET_NAME}" \
    --runid "${t2_runid}" \
    --threads "${THREADS}" \
    --num_gpus "${NUM_GPUS}" \
    2>&1 | tee "${LOG_DIR}/test_ep${ep4}_t2raw_run${t2_runid}.log"
done

echo
echo "All checkpoint evaluations finished."