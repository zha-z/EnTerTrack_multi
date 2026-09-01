# D2-S1 smoke 复现命令

工作目录：`/data/zjy/EnTeR-Track-main`；分支：`feature/pcum-cross-layer-arp`。

## 静态与 CPU 回归

```bash
/home/user/.conda/envs/zjy/bin/python -m py_compile \
  lib/train/target_prompt_d2_s1_source_degradation.py \
  tracking/audit_target_prompt_e3_d2_s1_smoke.py \
  tests/test_target_prompt_d2_s1_source_degradation.py

MPLCONFIGDIR=/tmp/mplconfig-d2s1 \
/home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_target_prompt_collaboration \
  tests.test_target_prompt_asymmetric_degradation.AsymmetricDegradationTests \
  tests.test_target_prompt_d2_p2_calibration \
  tests.test_target_prompt_d2_s1_source_degradation
```

结果：36 tests，failures 0，errors 0。

## 真实 Three-MDOT loader

```bash
MPLCONFIGDIR=/tmp/mplconfig-d2s1 \
/home/user/.conda/envs/zjy/bin/python \
  tracking/audit_target_prompt_e3_d2_s1_smoke.py --mode loader
```

该命令只读取一个 train batch 和一个 validation batch；正式 YAML 仅在进程内复制后改成 worker=0、batch=2、samples=2 的 smoke bound，不写回配置。

## 单 GPU forward/backward

```bash
/home/user/.conda/envs/zjy/bin/python \
  tracking/audit_target_prompt_e3_d2_s1_smoke.py --mode gpu
```

## 两卡 DDP forward/backward

```bash
/home/user/.conda/envs/zjy/bin/torchrun --standalone --nproc_per_node=2 \
  tracking/audit_target_prompt_e3_d2_s1_smoke.py --mode ddp
```

GPU/DDP 命令均只运行一个真实 batch/rank 的 loss + backward；不调用 `optimizer.step()`，不写 checkpoint，不进入 epoch loop。

## 未运行

以下命令/阶段本轮均未执行：25 epoch D2-S1 training、任何 D1 checkpoint resume、`threemdot_test`、official test、tracking rollout、Rescue/Preserve loss、selector/gate、degradation sweep 或超参调整。
