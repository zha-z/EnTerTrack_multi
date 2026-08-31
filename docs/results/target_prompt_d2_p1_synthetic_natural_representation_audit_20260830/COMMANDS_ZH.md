# D2-P1 复现命令

## 环境

从仓库根目录 `/data/zjy/EnTeR-Track-main` 执行。Python 为 `/home/user/.conda/envs/zjy/bin/python`，直接运行 tracking script 时必须设置 `PYTHONPATH=.`。

结果目录采用冻结路径：

```bash
export D2P1_OUT=docs/results/target_prompt_d2_p1_synthetic_natural_representation_audit_20260830
```

注意：inventory 与 representation writer 会拒绝覆盖已有冻结 artifact。完整重放必须在独立 worktree 或空的同名结果目录中进行；不要删除当前已归档结果来重跑。

## 1. 开始检查

```bash
git branch --show-current
git rev-parse HEAD
git status --short
sha256sum output/diagnostics/b0_abc_plain/run_20260818_seed42_4gpu_r001/checkpoints/train/entertrack/b0_abc_plain_4gpu/EnTeRTrack_ep0025.pth.tar
```

本轮 source HEAD：`2abaeea8adaa6a0a2b4eaf93460c1e1780336ccf`；checkpoint SHA256：`363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3`。

## 2. 单元测试

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python -m unittest \
  tests.test_target_prompt_d2_p1_representation \
  tests.test_target_prompt_e3_u1
```

结果：13 tests，PASS。

## 3. 先冻结 inventory

这一步必须在任何 model forward 前完成：

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/run_target_prompt_d2_p1_representation.py \
  --phase inventory \
  --output-dir "$D2P1_OUT"
```

验证：

```bash
sha256sum "$D2P1_OUT/sample_inventory.csv"
```

期望：`ca20bc2b0482dde2088bff5e8a2ff2e2c545cceb4ac6302de3e6ad68fe17b911`。

## 4. 冻结 Local/B0 representation

本轮实际使用单 GPU、batch size 64：

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/run_target_prompt_d2_p1_representation.py \
  --phase representation \
  --output-dir "$D2P1_OUT" \
  --gpu 0 \
  --batch-size 64
```

writer 在输出前验证 inventory SHA、B0 checkpoint SHA、strict local-core identity、K=8，并记录 adapter forward 次数。输出使用原子 staging；已存在目标文件时拒绝覆盖。

## 5. 冻结后描述性分析

```bash
env PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/analyze_target_prompt_d2_p1_representation.py \
  --output-dir "$D2P1_OUT"
```

analyzer 会再次验证 inventory、feature CSV 与 NPZ SHA256，以及 sample order/shape；之后才生成 paired 与 distribution artifacts。

## 6. 完整性检查

```bash
sha256sum \
  "$D2P1_OUT/sample_inventory.csv" \
  "$D2P1_OUT/representation_features.csv" \
  "$D2P1_OUT/prompt_features.npz" \
  "$D2P1_OUT/clean_synthetic_paired_analysis.csv" \
  "$D2P1_OUT/group_distribution_summary.csv" \
  "$D2P1_OUT/distribution_distance.csv" \
  "$D2P1_OUT/per_target_summary.csv" \
  "$D2P1_OUT/per_view_summary.csv"
```

```bash
git diff --check
git status --short
```

本任务没有执行 training、backward、optimizer、checkpoint update、`threemdot_test`、official test、D2 implementation 或任何 sweep。
