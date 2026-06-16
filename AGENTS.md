# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python/PyTorch tracking project. Core source lives in `lib/`: models in `lib/models`, training actors and data loaders in `lib/train`, evaluation code in `lib/test`, and shared helpers in `lib/utils`. CLI entry points are under `tracking/`, including training, testing, profiling, and analysis scripts. Experiment YAML files and command notes are stored in `experiments/entertrack/` and `experiments/entertrack_teacher/`. Tests live in `tests/`, currently focused on PCUM behavior. Runtime artifacts such as checkpoints, tracking results, analysis reports, and TensorBoard logs are written to `output/`, `outputs/`, and `tensorboard/`; avoid committing generated artifacts unless explicitly needed.

## Build, Test, and Development Commands

Set up the environment with Python 3.8:

```bash
conda create -n entertrack python=3.8
conda activate entertrack
pip install -r requirements.txt
```

Run the PCUM regression tests:

```bash
python -m unittest tests.test_pcum
```

Run a quick syntax check after editing training/model code:

```bash
python -m py_compile lib/models/entertrack/entertrack.py lib/train/actors/entertrack_threemdot.py
```

Train and test an experiment:

```bash
python tracking/train.py --script entertrack --config pcum_ablation_current_full --save_dir output/pcum_ablation_current --mode multiple --nproc_per_node 6 --use_wandb 0
python tracking/test.py --tracker_name entertrack --tracker_param pcum_ablation_current_full --dataset_name threemdot_test --runid 0 --threads 12 --num_gpus 6
```

## Coding Style & Naming Conventions

Use 4-space indentation and keep Python code compatible with Python 3.8. Follow existing naming patterns: snake_case for functions and variables, CamelCase for classes, and uppercase YAML keys. Keep model/config additions scoped and mirrored between `lib/config/entertrack/config.py` defaults and experiment YAMLs. Prefer explicit tensor shape comments where tracker data switches between single-view and multi-view formats.

## Testing Guidelines

Use `unittest`; add new tests to `tests/test_*.py`. For PCUM, sampler, or actor changes, extend `tests/test_pcum.py` with shape, config-load, and loss-path checks. Before long training runs, at minimum run `python -m unittest tests.test_pcum` and a targeted `py_compile` command for edited modules.

## Commit & Pull Request Guidelines

Git history currently contains only an initial commit, so use concise imperative commit messages such as `Add PCUM dropout ablation config`. Pull requests should describe the experiment or code path changed, list validation commands and results, note affected YAML configs, and include metric tables or visualization links for tracking-quality changes.

## Security & Configuration Tips

Dataset and output paths are machine-specific, especially `lib/test/evaluation/local.py` and training environment files. Do not commit private absolute paths, large checkpoints, raw datasets, or generated tracking outputs unless the change is intentionally archival.

## Git checkpoint policy

每次完成用户要求的代码修改后：

1. 运行相关测试。
2. 运行 `git status --short` 检查修改。
3. 确认没有密钥、数据集、模型权重、日志或临时文件。
4. 执行 `git add -A`。
5. 创建一次 Git 提交，提交信息格式：
   `codex: <本次修改的简短说明>`
6. 除非用户明确要求，否则不要执行 `git push`。
7. 不要修改、合并或覆盖用户已有但与当前任务无关的改动。