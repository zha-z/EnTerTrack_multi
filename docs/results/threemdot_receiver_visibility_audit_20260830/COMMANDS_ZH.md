# D2-P0 实际命令

所有命令均从 `/data/zjy/EnTeR-Track-main` 执行。

## 1. 开始前检查

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git remote -v
```

结果：

```text
feature/pcum-cross-layer-arp
5b0e6a9b446b08e561c067b52db1bc38b1ec3081
(git status --short 无输出，worktree clean)
origin  git@github.com:zha-z/EnTerTrack_multi.git
```

没有执行 reset、checkout、clean 或 discard；本输出目录在 clean 检查后新建，没有覆盖已有 result directory。

## 2. 静态代码与 schema 审计

使用 `sed` / `nl` / `rg` 只读检查：

```bash
sed -n '1,420p' docs/project_context_for_chatgpt_zh.md
sed -n '1,180p' docs/results/target_prompt_collaboration_e3_u1_utility_audit_20260829/REPORT_ZH.md
sed -n '1,140p' docs/results/target_prompt_collaboration_e3_d1_implementation_smoke_20260829/DATA_FLOW_AUDIT_ZH.md
sed -n '1,140p' docs/results/target_prompt_collaboration_e3_d1_ep25_test_20260830/REPORT_ZH.md
sed -n '1,130p' experiments/entertrack/target_prompt_collaboration_e3.yaml
sed -n '1,80p' experiments/entertrack/target_prompt_collaboration_e3_d1.yaml
```

```bash
rg -n "TrackingSamplerThreeMDOT|build_dataloaders_threemdot|REQUIRE_ALL_VIEWS_VISIBLE|INDEPENDENT_VIEW_SAMPLING|target_visible|visible|valid|absence|out_of_view|occlusion|common.visible" lib experiments
```

重点逐行读取：

```bash
nl -ba lib/train/dataset/threemdot.py
nl -ba lib/train/data/sampler_threemdot.py
nl -ba lib/train/base_functions.py
nl -ba lib/train/actors/target_prompt_collaboration.py
nl -ba lib/train/target_prompt_asymmetric_degradation.py
```

## 3. 冻结统计

协议先写入 `PROTOCOL_ZH.md`，再执行：

```bash
PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  docs/results/threemdot_receiver_visibility_audit_20260830/audit_visibility.py \
  --output-dir docs/results/threemdot_receiver_visibility_audit_20260830
```

脚本只硬编码读取 train/val split；没有 test split 参数或 test dataset 调用。输出摘要：

```text
train: sync=16138, N_common=12064, 3*N_common=36192,
       N_receiver_total=42737, extra=6545, increase=18.084107%,
       natural_asymmetric=6545 (15.314599%)
val:   sync=4704, N_common=4107, 3*N_common=12321,
       N_receiver_total=13513, extra=1192, increase=9.674539%,
       natural_asymmetric=1192 (8.821135%)
```

## 4. 复验

```bash
/home/user/.conda/envs/zjy/bin/python -m json.tool \
  docs/results/threemdot_receiver_visibility_audit_20260830/provenance.json
```

```bash
sha256sum \
  docs/results/threemdot_receiver_visibility_audit_20260830/*.csv \
  docs/results/threemdot_receiver_visibility_audit_20260830/audit_visibility.py
```

```bash
git diff --check
git status --short
```

本轮没有 training、backward、validation tracking、`tracking/test.py`、`threemdot_test` 或 official test 命令。
