# E3-D1 命令记录

以下均在 `/data/zjy/EnTeR-Track-main` 执行。

## 已执行：单元/E3 回归

```bash
MPLCONFIGDIR=/tmp/matplotlib-e3d1 PYTHONPATH=. \
  /home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_target_prompt_asymmetric_degradation \
  tests.test_target_prompt_collaboration \
  tests.test_target_prompt_e3_u1
```

CPU环境下真实 GPU case 按设计 skip；其余 28 tests PASS。

## 已执行：真实模型 GPU smoke

```bash
MPLCONFIGDIR=/tmp/matplotlib-e3d1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_target_prompt_asymmetric_degradation.AsymmetricDegradationRealModelSmoke.test_b0_initialized_actor_train_val_checkpoint_smoke
```

最终 1 test PASS。

## 已执行：真实 train/val 单批数据流

```bash
MPLCONFIGDIR=/tmp/matplotlib-e3d1 PYTHONPATH=. \
  /home/user/.conda/envs/zjy/bin/python \
  tracking/audit_target_prompt_e3_d1_dataflow.py --seed 20260829
```

结果 PASS；`official_test_accessed=false`。

## 已执行：actor/PCUM 回归

```bash
MPLCONFIGDIR=/tmp/matplotlib-e3d1 PYTHONPATH=. \
  /home/user/.conda/envs/zjy/bin/python -m unittest tests.test_pcum
```

90 tests PASS。

## 未执行：未来四卡正式训练

只有在用户另行授权长期训练后才执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/user/.conda/envs/zjy/bin/torchrun \
  --standalone --nproc_per_node=4 \
  lib/train/run_training.py \
  --script entertrack \
  --config target_prompt_collaboration_e3_d1 \
  --save_dir output/diagnostics/target_prompt_collaboration_e3_d1/<NEW_RUN_ID> \
  --seed 42 \
  --use_wandb 0
```

不得复用已有 run id。这里不提供 official test命令。
