# E1.5 smoke test 与完整性审计

## 结果

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 分支与源 HEAD | PASS | `feature/pcum-cross-layer-arp`，源 HEAD `affcd40`，与 origin 同步 |
| 配置默认关闭 | PASS | 新开关默认 `false`；只有 E1.5 YAML 打开 |
| 配置解析 | PASS | E1.5 继承 V1，并要求 collaboration + SAFE_COMMIT |
| Python 编译 | PASS | config、tracker、runner、analysis script 全部 `py_compile` 通过 |
| 单元测试 | PASS | targeted：26 tests；最终 `tests.test_plain_collaboration + tests.test_pcum`：116 tests passed |
| 真实 checkpoint forward | PASS | GPU 0，`md3016` A/B/C，各 3 帧 |
| local backbone 次数 | PASS | 每个 receiver/frame 恰好一次，三路计数增量 `(1,1,1)` |
| 四 branch | PASS | local、sender0-only、sender1-only、both 均从同一 local candidate/pre-state 产生 |
| R=1 隔离 | PASS | 正常模式拒绝单 sender；仅 diagnostic 模式接受 R=1/2 |
| CENTER 输出 | PASS | 四 branch 的 bbox/score/APCE 均 finite |
| 完整 search token | PASS | active frame `search_token_count=256` |
| sender 数量与权重 | PASS | single remote count=1、weight=1.0；both count=2、weight=0.5/0.5 |
| prediction-only | PASS | 56,448 行均 `uses_gt=false`，冻结 schema 无 GT/IoU/label |
| branch 不写状态 | PASS | active rows 的 before/after persistent digest 全相同 |
| Safe Commit | PASS | 最终 tracker state 为 local，默认 report 为 both |
| local identity | PASS | 与既有 D0-local bbox 逐行一致，mismatch=0 |
| both identity | PASS | 与既有 D0-safe bbox 逐行一致，mismatch=0 |
| 标准结果一致 | PASS | both report 与本轮结果 txt 逐行一致，mismatch=0 |
| CSV 保存 | PASS | 15 个 sequence counterfactual CSV，56,448 rows，四分支/group |
| 冻结后 join | PASS | join 前重新验证预测 SHA256 |
| official test | PASS | 未访问、未运行；分析脚本显式拒绝 dataset 名包含 `test` |
| 训练边界 | PASS | 没有 backward、optimizer、checkpoint 保存或训练 |

## 真实 checkpoint 小样本 smoke

- 临时目录：`/tmp/plain-collaboration-e15-smoke-28300`
- sequence：`md3016-1/2/3`
- 每路 3 帧（含初始化）
- 每路 sender-counterfactual CSV：12 行，即 `3 frames x 4 branches`
- 临时产物未写入正式结果目录，也不提交 Git。

## 完整 inner-val rollout 审计

- 结果目录：`output/test/tracking_results/entertrack/plain_collaboration_v1_e15_sender_counterfactual_28315`
- 5 targets、15 sequences、91 个结果/诊断文件。
- receiver/frame groups：14,112。
- prediction rows：56,448。
- target-visible valid frames：13,513。
- active rows：56,388；初始化 rows：60。
- persistent mutations：0。
- `uses_gt=true`：0。

冻结 prediction SHA256：

```text
75ebb3cb292202346619e6f81cc46d050fa394a73e7c0d01ad9376545ba95f43
```

本轮 smoke 不包含 backward/checkpoint-save 测试，因为任务明确禁止训练；没有把未执行项目报告为 PASS。
