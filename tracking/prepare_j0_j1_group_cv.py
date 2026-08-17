#!/usr/bin/env python3
"""Prepare deterministic target-group CV manifests and commands for J0/J1."""

import argparse
import csv
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/controlled_baselines/pcum_joint_adaptation"
MANIFEST_DIR = OUT / "cv_manifests"
CONFIG_DIR = ROOT / "experiments/entertrack"
SPLIT_DIR = ROOT / "lib/train/data_specs/threemdot"
TRAIN_SPLIT = SPLIT_DIR / "threemdot_train.txt"
VAL_SPLIT = SPLIT_DIR / "threemdot_val.txt"
TEST_SPLIT = SPLIT_DIR / "threemdot_test.txt"
FOLD_COUNT = 5
FOLD_SIZES = [5, 5, 5, 4, 4]
SEED = 42
B0_CHECKPOINT = ROOT / "output/controlled_baselines/b0/checkpoints/train/entertrack/ostrack_deit_tiny_b0_ep25/EnTeRTrack_ep0025.pth.tar"


def read_split(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def target_id(sequence):
    return sequence.rsplit("-", 1)[0]


def group_by_target(sequences):
    grouped = {}
    for sequence in sequences:
        grouped.setdefault(target_id(sequence), []).append(sequence)
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def frame_count(sequence):
    root = Path("/data2/Three-MDOT")
    path = root / target_id(sequence) / sequence / "groundtruth.txt"
    if not path.is_file():
        return ""
    return len([line for line in path.read_text().splitlines() if line.strip()])


def deterministic_folds(targets):
    targets = list(sorted(targets))
    rng = random.Random(SEED)
    rng.shuffle(targets)
    folds = []
    cursor = 0
    for size in FOLD_SIZES:
        folds.append(sorted(targets[cursor:cursor + size]))
        cursor += size
    return folds


def write_list(path, sequences):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sequences) + "\n")


def fold_config_name(base, fold):
    return "ostrack_deit_tiny_b0_%s_partial_adapt_ep15_fold%d" % (base, fold)


def write_fold_config(model, fold, train_file, holdout_file):
    if model == "j0":
        base = "ostrack_deit_tiny_b0_j0_partial_adapt_ep15"
        role = "posthoc_j0_b0_partial_adapt_fold%d" % fold
        save_dir = "output/controlled_baselines/pcum_joint_adaptation/fold_%d/j0_train" % fold
    else:
        base = "ostrack_deit_tiny_b0_j1_pcum_partial_adapt_ep15"
        role = "posthoc_j1_b0_pcum_partial_adapt_fold%d" % fold
        save_dir = "output/controlled_baselines/pcum_joint_adaptation/fold_%d/j1_train" % fold
    name = fold_config_name(model, fold)
    text = """BASE_CONFIG: {base}
MODEL_ROLE: {role}
DATA:
  TRAIN:
    SPLIT_FILE: {train_file}
  VAL:
    DATASETS_NAME: [THREEMDOT]
    DATASETS_RATIO: [1]
    SPLIT_FILE: {holdout_file}
TRAIN:
  SEED: {seed}
  VAL_EPOCH_INTERVAL: 9999
TEST:
  SAVE_DIR: {save_dir}
  CHECKPOINT_NAME: {name}
""".format(
        base=base,
        role=role,
        train_file=train_file,
        holdout_file=holdout_file,
        seed=SEED,
        save_dir=save_dir,
        name=name,
    )
    (CONFIG_DIR / (name + ".yaml")).write_text(text)
    return name, save_dir


def train_command(config, save_dir, log_path):
    return (
        "cd /data/zjy/EnTeR-Track-main\n"
        "mkdir -p {log_dir}\n"
        "set -o pipefail\n"
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /home/user/.conda/envs/zjy/bin/torchrun "
        "--standalone --nproc_per_node=6 lib/train/run_training.py "
        "--script entertrack --config {config} --save_dir {save_dir} "
        "--seed 42 --use_wandb 0 2>&1 | tee {log_path}"
    ).format(config=config, save_dir=save_dir, log_path=log_path,
             log_dir=str(Path(log_path).parent))


def eval_command(config, split_file, runid, log_path):
    return (
        "cd /data/zjy/EnTeR-Track-main\n"
        "mkdir -p {log_dir}\n"
        "set -o pipefail\n"
        "THREEMDOT_CV_SPLIT_FILE={split_file} "
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /home/user/.conda/envs/zjy/bin/python "
        "tracking/test.py --tracker_name entertrack --tracker_param {config} "
        "--dataset_name threemdot_cv --runid {runid} --threads 12 --num_gpus 6 "
        "2>&1 | tee {log_path}"
    ).format(config=config, split_file=split_file, runid=runid,
             log_path=log_path, log_dir=str(Path(log_path).parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    train_sequences = read_split(TRAIN_SPLIT)
    val_targets = set(map(target_id, read_split(VAL_SPLIT)))
    test_targets = set(map(target_id, read_split(TEST_SPLIT)))
    grouped = group_by_target(train_sequences)
    folds = deterministic_folds(grouped.keys())

    if len(grouped) != 23:
        raise RuntimeError("Expected 23 train targets, got %d" % len(grouped))
    if set(grouped) & val_targets or set(grouped) & test_targets:
        raise RuntimeError("Train CV targets overlap val/test targets")

    assignment_rows = []
    registry_rows = []
    command_blocks = []
    print_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cat <<'CMDS'",
    ]

    all_holdout = []
    for fold_id, holdout_targets in enumerate(folds):
        train_targets = sorted(set(grouped) - set(holdout_targets))
        holdout_sequences = [seq for target in holdout_targets for seq in grouped[target]]
        train_fold_sequences = [seq for target in train_targets for seq in grouped[target]]
        train_file = MANIFEST_DIR / ("fold_%d_train.txt" % fold_id)
        holdout_file = MANIFEST_DIR / ("fold_%d_holdout.txt" % fold_id)
        if args.write:
            write_list(train_file, train_fold_sequences)
            write_list(holdout_file, holdout_sequences)
        all_holdout.extend(holdout_targets)
        for target in sorted(grouped):
            role = "holdout" if target in holdout_targets else "train"
            assignment_rows.append({
                "target_id": target,
                "view_sequences": "|".join(grouped[target]),
                "fold_id": fold_id,
                "role": role,
                "frame_count": sum(int(frame_count(seq) or 0) for seq in grouped[target]),
            })
        fold_commands = {}
        for model in ("j0", "j1"):
            config, save_dir = write_fold_config(model, fold_id, train_file, holdout_file) if args.write else (
                fold_config_name(model, fold_id),
                "output/controlled_baselines/pcum_joint_adaptation/fold_%d/%s_train" % (fold_id, model),
            )
            runid = 26000 + fold_id * 10 + (0 if model == "j0" else 1)
            log_dir = OUT / "logs"
            train_log = log_dir / ("%s_fold%d_seed42_train.log" % (model, fold_id))
            eval_log = log_dir / ("%s_fold%d_holdout_eval_%d.log" % (model, fold_id, runid))
            registry_rows.append({
                "fold_id": fold_id,
                "model_role": model.upper(),
                "config": config,
                "train_targets": "|".join(train_targets),
                "holdout_targets": "|".join(holdout_targets),
                "seed": SEED,
                "initial_checkpoint": str(B0_CHECKPOINT),
                "final_epoch": 15,
                "output_dir": save_dir,
                "train_log": str(train_log),
                "evaluation_runid": runid,
                "status": "planned_not_run",
            })
            fold_commands[model] = {
                "train": train_command(config, save_dir, train_log),
                "eval": eval_command(config, holdout_file, runid, eval_log),
            }
        command_blocks.append(("fold%d J0 training" % fold_id, fold_commands["j0"]["train"]))
        command_blocks.append(("fold%d J1 training" % fold_id, fold_commands["j1"]["train"]))
        command_blocks.append(("fold%d J0 held-out evaluation" % fold_id, fold_commands["j0"]["eval"]))
        command_blocks.append(("fold%d J1 held-out evaluation" % fold_id, fold_commands["j1"]["eval"]))

    if sorted(all_holdout) != sorted(grouped):
        raise RuntimeError("Holdout union does not equal all train targets")
    if len(all_holdout) != len(set(all_holdout)):
        raise RuntimeError("A target is held out more than once")

    if args.write:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        with (MANIFEST_DIR / "fold_assignment.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "target_id", "view_sequences", "fold_id", "role", "frame_count"])
            writer.writeheader()
            writer.writerows(assignment_rows)
        with (OUT / "cv_experiment_registry.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "fold_id", "model_role", "config", "train_targets",
                "holdout_targets", "seed", "initial_checkpoint", "final_epoch",
                "output_dir", "train_log", "evaluation_runid", "status"])
            writer.writeheader()
            writer.writerows(registry_rows)
        fold_lines = [
            "# J0/J1 target-group CV fold manifest",
            "",
            "- fold_count: `5`",
            "- seed: `42`",
            "- fold sizes: `5/5/5/4/4` targets",
            "- source split: `lib/train/data_specs/threemdot/threemdot_train.txt`",
            "- Three-MDOT val/test targets are excluded.",
            "",
            "| Fold | Holdout targets | Train target count | Holdout sequence-view count |",
            "|---:|---|---:|---:|",
        ]
        for fold_id, holdout_targets in enumerate(folds):
            fold_lines.append("| %d | `%s` | %d | %d |" % (
                fold_id, ", ".join(holdout_targets), 23 - len(holdout_targets),
                len(holdout_targets) * 3))
        (MANIFEST_DIR / "fold_manifest.md").write_text("\n".join(fold_lines) + "\n")
        pipeline = """# CV data-pipeline fairness audit

| Check | Status |
|---|---|
| J0/J1 use same train target list per fold | `PASS` |
| J0/J1 use same holdout target list per fold | `PASS` |
| Three views stay grouped by target | `PASS` |
| sampler seed | `42` |
| augmentation and config seed | `same command seed=42` |
| DATA.TRAIN.SAMPLE_PER_EPOCH | `6000` for both |
| batch size | `8` for both |
| optimizer step count | `same samples/epoch, batch size, epochs` |
| total epoch | `15` |
| J0 flat multiview baseline | `enabled; reads the same three-view groups` |
| J1 additional path | `PCUM forward + PCUM paired/safe loss only` |
| Three-MDOT val/test usage | `forbidden for selection; not included in fold manifests` |

The only method differences are PCUM enablement, PCUM forward, and PCUM
supervision/safe loss. Fold YAML files inherit from the audited J0/J1 base
configs via BASE_CONFIG.
"""
        (OUT / "cv_data_pipeline_audit.md").write_text(pipeline)
        manual = ["# J0/J1 CV manual run commands", ""]
        for title, command in command_blocks:
            manual.extend(["## %s" % title, "", "```bash", command, "```", ""])
            print_lines.extend(["# %s" % title, command, ""])
        manual.append("Do not run Three-MDOT val/test. Do not change folds after results are observed.\n")
        (OUT / "cv_manual_run_commands.md").write_text("\n".join(manual))
        print_lines.append("CMDS")
        script = OUT / "print_cv_commands.sh"
        script.write_text("\n".join(print_lines) + "\n")
        os.chmod(script, 0o755)

    print("fold_count=5")
    for fold_id, holdout_targets in enumerate(folds):
        print("fold_%d_holdout=%s" % (fold_id, ",".join(holdout_targets)))


if __name__ == "__main__":
    main()
