#!/usr/bin/env bash
set -Eeuo pipefail


###############################################################################
# PCUM supervision experiments:
#
#   train E1 -> test T1/T2
#   train E2a -> test T1/T2
#   train E2b -> test T1/T2
#   train E3 -> test T1/T2
#   train E4 -> test T1/T2
#
# T1:
#   MODEL.PCUM.ENABLED=True
#   TEST.PCUM.USE_REMOTE=False
#
# T2:
#   MODEL.PCUM.ENABLED=True
#   TEST.PCUM.USE_REMOTE=True
#   TEST.PCUM.USE_REMOTE_VISIBLE_MASK=False
#
# 默认做5 epoch短训：
#   bash scripts/run_pcum_supervision_all.sh
#
# 使用原YAML完整40 epoch：
#   EPOCH_OVERRIDE=0 bash scripts/run_pcum_supervision_all.sh
#
# 只跑指定实验：
#   ONLY=e1,e2a,e3 bash scripts/run_pcum_supervision_all.sh
#
# 跳过训练，只执行测试：
#   SKIP_TRAIN=1 bash scripts/run_pcum_supervision_all.sh
###############################################################################


############################
# 1. 全局配置
############################

PROJECT_ROOT="/data/zjy/EnTeR-Track-main"
CONFIG_DIR="${PROJECT_ROOT}/experiments/entertrack"

# 默认5 epoch短训。
# 设为0时，使用各原始YAML中的TRAIN.EPOCH。
EPOCH_OVERRIDE="${EPOCH_OVERRIDE:-5}"

# all 或 e1,e2a,e2b,e3,e4
ONLY="${ONLY:-all}"

# GPU设置
GPUS="${GPUS:-0,1,2,3,4,5}"
NUM_GPUS="${NUM_GPUS:-6}"

# 测试参数
DATASET_NAME="${DATASET_NAME:-threemdot_test}"
TEST_THREADS="${TEST_THREADS:-12}"

# 是否使用 wandb
USE_WANDB="${USE_WANDB:-0}"

# 跳过开关
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"

# T2是否关闭“remote较差时保留local”的安全机制。
#
# 0：保留原YAML设置，属于正式安全推理结果。
# 1：强制KEEP_LOCAL_IF_REMOTE_WORSE=False，
#    更适合观察未经fallback掩盖的纯remote作用。
PURE_REMOTE="${PURE_REMOTE:-0}"

# 测试runid起点
RUN_BASE="${RUN_BASE:-9200}"

# 若你的训练命令还需要类似：
#   --mode multiple --nproc_per_node 6
# 可以在启动脚本时传入：
#
# TRAIN_EXTRA_ARGS="--mode multiple --nproc_per_node 6" bash ...
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"


############################
# 2. 五组实验
############################

CONFIGS=(
    "pcum_supervision_e1_paired_lr8e6"
    "pcum_supervision_e2a_safe_m0_lr8e6"
    "pcum_supervision_e2b_safe_m0001_lr8e6"
    "pcum_supervision_e3_safe_m0_lr4e5"
    "pcum_supervision_e4_safe_m0_lr8e5"
)

LABELS=(
    "e1"
    "e2a"
    "e2b"
    "e3"
    "e4"
)


############################
# 3. 初始化
############################

cd "${PROJECT_ROOT}"

MASTER_OUTPUT="${PROJECT_ROOT}/output/pcum_supervision_pipeline"
LOG_DIR="${MASTER_OUTPUT}/logs"
META_DIR="${MASTER_OUTPUT}/metadata"

mkdir -p "${LOG_DIR}" "${META_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "zjy" ]]; then
    echo "[ERROR] 当前环境不是 zjy。"
    echo "请先运行：conda activate zjy"
    exit 1
fi

python - <<'PY'
import torch
import yaml

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA不可用")
PY

echo
echo "============================================================"
echo "PCUM train + test pipeline"
echo "PROJECT_ROOT:    ${PROJECT_ROOT}"
echo "EPOCH_OVERRIDE:  ${EPOCH_OVERRIDE}"
echo "ONLY:            ${ONLY}"
echo "GPUS:            ${GPUS}"
echo "NUM_GPUS:        ${NUM_GPUS}"
echo "DATASET:         ${DATASET_NAME}"
echo "PURE_REMOTE:     ${PURE_REMOTE}"
echo "SKIP_TRAIN:      ${SKIP_TRAIN}"
echo "SKIP_TEST:       ${SKIP_TEST}"
echo "============================================================"
echo


############################
# 4. 错误处理
############################

CURRENT_STAGE="initialization"

error_handler() {
    local exit_code=$?

    echo
    echo "============================================================"
    echo "[FAILED]"
    echo "Stage: ${CURRENT_STAGE}"
    echo "Exit code: ${exit_code}"
    echo "============================================================"

    exit "${exit_code}"
}

trap error_handler ERR


############################
# 5. 工具函数
############################

should_run() {
    local label="$1"

    if [[ "${ONLY}" == "all" ]]; then
        return 0
    fi

    [[ ",${ONLY}," == *",${label},"* ]]
}


to_absolute_path() {
    local path="$1"

    if [[ "${path}" = /* ]]; then
        echo "${path}"
    else
        echo "${PROJECT_ROOT}/${path}"
    fi
}


prepare_train_config() {
    local base_stem="$1"
    local source_yaml="${CONFIG_DIR}/${base_stem}.yaml"

    if [[ ! -f "${source_yaml}" ]]; then
        echo "[ERROR] 找不到配置：${source_yaml}"
        exit 1
    fi

    local result

    result="$(
        python - \
            "${source_yaml}" \
            "${CONFIG_DIR}" \
            "${base_stem}" \
            "${EPOCH_OVERRIDE}" <<'PY'
import sys
from pathlib import Path

import yaml


source_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
base_stem = sys.argv[3]
epoch_override = int(sys.argv[4])

with source_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise RuntimeError(f"无效YAML：{source_path}")

train_cfg = cfg.setdefault("TRAIN", {})
test_cfg = cfg.setdefault("TEST", {})

if epoch_override <= 0:
    run_stem = base_stem

    train_epoch = int(train_cfg["EPOCH"])
    test_cfg.setdefault("EPOCH", train_epoch)

    save_dir = str(
        test_cfg.get(
            "SAVE_DIR",
            f"output/{base_stem}",
        )
    )

    checkpoint_name = str(
        test_cfg.get(
            "CHECKPOINT_NAME",
            base_stem,
        )
    )

else:
    run_stem = f"_short_{base_stem}_ep{epoch_override}"

    train_cfg["EPOCH"] = epoch_override
    train_cfg["RESUME"] = False

    test_cfg["EPOCH"] = epoch_override
    test_cfg["CHECKPOINT_NAME"] = run_stem
    test_cfg["SAVE_DIR"] = f"output/{run_stem}"

    save_dir = test_cfg["SAVE_DIR"]
    checkpoint_name = run_stem

    target_path = config_dir / f"{run_stem}.yaml"

    with target_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    print(
        f"[CONFIG] generated: {target_path}",
        file=sys.stderr,
    )

train_epoch = int(train_cfg["EPOCH"])

print(
    "|".join([
        run_stem,
        save_dir,
        checkpoint_name,
        str(train_epoch),
    ])
)
PY
    )"

    IFS='|' read -r \
        TRAIN_CONFIG_STEM \
        TRAIN_SAVE_DIR_REL \
        TRAIN_CHECKPOINT_NAME \
        TRAIN_EPOCH \
        <<< "${result}"

    TRAIN_SAVE_DIR_ABS="$(to_absolute_path "${TRAIN_SAVE_DIR_REL}")"

    mkdir -p "${TRAIN_SAVE_DIR_ABS}"
}


prepare_eval_config() {
    local train_stem="$1"
    local mode="$2"
    local use_remote="$3"

    local train_yaml="${CONFIG_DIR}/${train_stem}.yaml"

    if [[ ! -f "${train_yaml}" ]]; then
        echo "[ERROR] 训练配置不存在：${train_yaml}"
        exit 1
    fi

    local result

    result="$(
        python - \
            "${train_yaml}" \
            "${CONFIG_DIR}" \
            "${mode}" \
            "${use_remote}" \
            "${PURE_REMOTE}" <<'PY'
import sys
from pathlib import Path

import yaml


source_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
mode = sys.argv[3]
use_remote = sys.argv[4].lower() == "true"
pure_remote = int(sys.argv[5]) == 1

with source_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise RuntimeError(f"无效YAML：{source_path}")

source_stem = source_path.stem
eval_stem = f"_eval_{source_stem}_{mode}"

model_pcum = (
    cfg
    .setdefault("MODEL", {})
    .setdefault("PCUM", {})
)
model_pcum["ENABLED"] = True

test_pcum = (
    cfg
    .setdefault("TEST", {})
    .setdefault("PCUM", {})
)

test_pcum["USE_REMOTE"] = use_remote

# 正式测试禁止使用GT可见性筛选。
test_pcum["USE_REMOTE_VISIBLE_MASK"] = False

# T2原始remote因果测试，可以关闭fallback。
if use_remote and pure_remote:
    test_pcum["KEEP_LOCAL_IF_REMOTE_WORSE"] = False

target_path = config_dir / f"{eval_stem}.yaml"

with target_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg,
        f,
        allow_unicode=True,
        sort_keys=False,
    )

print(
    f"[CONFIG] generated eval: {target_path}",
    file=sys.stderr,
)

print(eval_stem)
PY
    )"

    EVAL_CONFIG_STEM="${result}"
}


save_metadata() {
    local label="$1"
    local config_stem="$2"
    local output_dir="${META_DIR}/${label}"

    mkdir -p "${output_dir}"

    cp \
        "${CONFIG_DIR}/${config_stem}.yaml" \
        "${output_dir}/train_config.yaml"

    git rev-parse HEAD \
        > "${output_dir}/git_commit.txt"

    git status --short \
        > "${output_dir}/git_status.txt"

    git diff \
        > "${output_dir}/worktree.diff"

    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "label=${label}"
        echo "config=${config_stem}"
        echo "checkpoint_name=${TRAIN_CHECKPOINT_NAME}"
        echo "save_dir=${TRAIN_SAVE_DIR_REL}"
        echo "epoch=${TRAIN_EPOCH}"
        echo "gpus=${GPUS}"
        echo "num_gpus=${NUM_GPUS}"
        echo "dataset=${DATASET_NAME}"
        echo "pure_remote=${PURE_REMOTE}"
    } > "${output_dir}/run_info.txt"
}


print_config_summary() {
    local config_stem="$1"

    python - \
        "${CONFIG_DIR}/${config_stem}.yaml" <<'PY'
import sys
from pathlib import Path

import yaml


path = Path(sys.argv[1])

with path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

train = cfg["TRAIN"]
pcum_train = train["PCUM"]
test = cfg["TEST"]

print("Configuration summary:")
print("  file:", path.name)
print("  epoch:", train["EPOCH"])
print("  PCUM LR:", train.get("PCUM_LR"))
print("  paired:", pcum_train.get("PAIRED_SUPERVISION"))
print("  local weight:", pcum_train.get("LOCAL_LOSS_WEIGHT"))
print("  collaborative weight:", pcum_train.get("COLLAB_LOSS_WEIGHT"))
print("  safe weight:", pcum_train.get("SAFE_LOSS_WEIGHT"))
print("  safe margin:", pcum_train.get("SAFE_MARGIN"))
print("  diagnostics:", pcum_train.get("DIAGNOSTICS_ENABLED"))
print("  test save dir:", test.get("SAVE_DIR"))
print("  checkpoint name:", test.get("CHECKPOINT_NAME"))
print("  test epoch:", test.get("EPOCH"))
PY
}


checkpoint_exists() {
    local save_dir="$1"

    if [[ ! -d "${save_dir}" ]]; then
        return 1
    fi

    find "${save_dir}" \
        -type f \
        \( \
            -name "*.pth.tar" \
            -o -name "*.pth" \
            -o -name "*.pt" \
        \) \
        -print -quit \
        | grep -q .
}


run_training() {
    local label="$1"
    local config_stem="$2"
    local save_dir="$3"

    local log_file="${LOG_DIR}/${label}_train.log"

    CURRENT_STAGE="${label}: training"

    echo
    echo "============================================================"
    echo "[TRAIN] ${label}"
    echo "config:   ${config_stem}"
    echo "save_dir: ${save_dir}"
    echo "log:      ${log_file}"
    echo "============================================================"

    local extra_args=()

    if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
        read -r -a extra_args <<< "${TRAIN_EXTRA_ARGS}"
    fi

    CUDA_VISIBLE_DEVICES="${GPUS}" \
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python tracking/train.py \
        --script entertrack \
        --config "${config_stem}" \
        --save_dir "${save_dir}" \
        --use_wandb "${USE_WANDB}" \
        "${extra_args[@]}" \
        2>&1 | tee "${log_file}"

    if ! checkpoint_exists "${save_dir}"; then
        echo "[ERROR] 训练结束，但在以下目录未发现checkpoint："
        echo "  ${save_dir}"
        echo
        echo "请检查训练日志："
        echo "  ${log_file}"
        exit 1
    fi
}


run_test() {
    local label="$1"
    local mode="$2"
    local config_stem="$3"
    local runid="$4"

    local log_file="${LOG_DIR}/${label}_${mode}_run${runid}.log"

    CURRENT_STAGE="${label}: test ${mode}"

    echo
    echo "------------------------------------------------------------"
    echo "[TEST] ${label} / ${mode}"
    echo "config:  ${config_stem}"
    echo "runid:   ${runid}"
    echo "dataset: ${DATASET_NAME}"
    echo "log:     ${log_file}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="${GPUS}" \
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    python tracking/test.py \
        --tracker_name entertrack \
        --tracker_param "${config_stem}" \
        --dataset_name "${DATASET_NAME}" \
        --runid "${runid}" \
        --threads "${TEST_THREADS}" \
        --num_gpus "${NUM_GPUS}" \
        2>&1 | tee "${log_file}"
}


############################
# 6. 配置预检查
############################

echo "[CHECK] 检查五个原始配置文件"

for config in "${CONFIGS[@]}"; do
    yaml_path="${CONFIG_DIR}/${config}.yaml"

    if [[ ! -f "${yaml_path}" ]]; then
        echo "[ERROR] 缺少配置：${yaml_path}"
        exit 1
    fi
done

echo "[CHECK] 所有配置文件存在"
echo


############################
# 7. 主循环
############################

SUMMARY_FILE="${MASTER_OUTPUT}/summary.tsv"

if [[ ! -f "${SUMMARY_FILE}" ]]; then
    printf \
        "label\ttrain_config\tsave_dir\tcheckpoint_name\tepoch\tt1_config\tt1_runid\tt2_config\tt2_runid\tfinished_at\n" \
        > "${SUMMARY_FILE}"
fi

for index in "${!CONFIGS[@]}"; do
    base_config="${CONFIGS[$index]}"
    label="${LABELS[$index]}"

    if ! should_run "${label}"; then
        echo "[SKIP] ${label} 不在 ONLY=${ONLY} 中"
        continue
    fi

    CURRENT_STAGE="${label}: prepare config"

    prepare_train_config "${base_config}"

    echo
    echo "############################################################"
    echo "Experiment: ${label}"
    echo "Base config: ${base_config}"
    echo "Run config:  ${TRAIN_CONFIG_STEM}"
    echo "Save dir:    ${TRAIN_SAVE_DIR_ABS}"
    echo "Epoch:       ${TRAIN_EPOCH}"
    echo "############################################################"

    print_config_summary "${TRAIN_CONFIG_STEM}"
    save_metadata "${label}" "${TRAIN_CONFIG_STEM}"

    if [[ "${SKIP_TRAIN}" == "1" ]]; then
        echo "[SKIP TRAIN] ${label}"
    else
        run_training \
            "${label}" \
            "${TRAIN_CONFIG_STEM}" \
            "${TRAIN_SAVE_DIR_ABS}"
    fi

    if [[ "${SKIP_TEST}" == "1" ]]; then
        echo "[SKIP TEST] ${label}"
        continue
    fi

    if ! checkpoint_exists "${TRAIN_SAVE_DIR_ABS}"; then
        echo "[ERROR] 测试前未找到checkpoint：${TRAIN_SAVE_DIR_ABS}"
        exit 1
    fi

    # 每组实验使用不同runid。
    #
    # e1:  9211 / 9212
    # e2a: 9221 / 9222
    # e2b: 9231 / 9232
    # e3:  9241 / 9242
    # e4:  9251 / 9252
    t1_runid=$((RUN_BASE + (index + 1) * 10 + 1))
    t2_runid=$((RUN_BASE + (index + 1) * 10 + 2))

    # T1：只使用local prompt。
    prepare_eval_config \
        "${TRAIN_CONFIG_STEM}" \
        "t1_local_only" \
        "false"

    t1_config="${EVAL_CONFIG_STEM}"

    run_test \
        "${label}" \
        "t1_local_only" \
        "${t1_config}" \
        "${t1_runid}"

    # T2：使用真实remote prompt。
    prepare_eval_config \
        "${TRAIN_CONFIG_STEM}" \
        "t2_real_remote" \
        "true"

    t2_config="${EVAL_CONFIG_STEM}"

    run_test \
        "${label}" \
        "t2_real_remote" \
        "${t2_config}" \
        "${t2_runid}"

    printf \
        "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${label}" \
        "${TRAIN_CONFIG_STEM}" \
        "${TRAIN_SAVE_DIR_REL}" \
        "${TRAIN_CHECKPOINT_NAME}" \
        "${TRAIN_EPOCH}" \
        "${t1_config}" \
        "${t1_runid}" \
        "${t2_config}" \
        "${t2_runid}" \
        "$(date --iso-8601=seconds)" \
        >> "${SUMMARY_FILE}"

    echo
    echo "[DONE] ${label} training + T1 + T2"
done


############################
# 8. 完成
############################

CURRENT_STAGE="finished"

echo
echo "============================================================"
echo "全部指定实验已完成"
echo
echo "日志目录："
echo "  ${LOG_DIR}"
echo
echo "实验元数据："
echo "  ${META_DIR}"
echo
echo "汇总文件："
echo "  ${SUMMARY_FILE}"
echo "============================================================"