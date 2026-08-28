# E3 smoke test

## 范围

- 数据集：`threemdot_val`，不是 `threemdot_test`。
- 自动选择最短完整 target：`md3048`。
- 序列：`md3048-1/2/3`，各 530 帧。
- 实际执行：初始化帧 + 1 tracking 帧，共 2 帧。
- checkpoint：B0 ep25；SHA256 见 provenance。
- GPU：物理 GPU 0（RTX 4090）。
- 不调用 result writer，不生成正式 tracking/evaluator 结果。

## PASS 清单

| 检查 | 结果 |
|---|---|
| E3/V1 联合 unittest | PASS，42/42 |
| extractor `[B,8,192]` / 0 parameter | PASS |
| top-k index 与 gathered token | PASS |
| prediction-only、无 GT API | PASS |
| remote `[B,2,8,192]`，K != 256 | PASS |
| disabled/no-remote exact bypass | PASS |
| invalid/non-finite fail-closed | PASS |
| relative residual cap | PASS |
| CENTER `[1,1,16,16]` | PASS，A/B/C |
| strict B0 core load + fresh E3 | PASS，A/B/C |
| 每 view/frame 一次 local backbone | PASS，A/B/C |
| sender prompt source=local | PASS，A/B/C |
| collaboration forward state identity | PASS，A/B/C |
| reported output=E3 | PASS，A/B/C |
| persistent state/next crop=local | PASS，A/B/C |
| valid remote count=2、K=8 | PASS，A/B/C |
| runtime uses_gt=false | PASS |
| V1 strict load与既有测试 | PASS，27/27 |

## 实际命令

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/user/.conda/envs/zjy/bin/python \
  tracking/smoke_target_prompt_collaboration_e3.py \
  --gpu 0 \
  --checkpoint output/diagnostics/b0_abc_plain/run_20260818_seed42_4gpu_r001/checkpoints/train/entertrack/b0_abc_plain_4gpu/EnTeRTrack_ep0025.pth.tar
```

结论只限于实现 smoke：**PASS**。未训练 E3，不能据此评价 tracking AUC。
