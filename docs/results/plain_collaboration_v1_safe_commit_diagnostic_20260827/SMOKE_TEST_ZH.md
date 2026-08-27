# Safe Commit smoke test

## 结果

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 配置解析 | PASS | local/closed/safe 三个 YAML 可加载，开关分别符合预期 |
| Python 编译 | PASS | 修改的 config、tracker、runner、analysis script 全部 `py_compile` 通过 |
| 单元测试 | PASS | `tests.test_plain_collaboration`：23 tests passed |
| 真实 checkpoint forward | PASS | GPU 0，`md3016` A/B/C，各截取 3 帧 |
| CENTER 输出 | PASS | score、APCE、bbox 均 finite，输出文件各 3 行 |
| 完整 search tokens | PASS | active frame `search_token_count=256` |
| 两个 remote sender | PASS | A/B/C 每个 receiver 均有 canonical 两 sender、权重 0.5/0.5 |
| collaborative forward 无持久状态写入 | PASS | forward 前后 persistent digest 相同 |
| Safe state output | PASS | `state_output_bbox == local_bbox` |
| Safe reported output | PASS | `reported_output_bbox == collaborative_bbox` |
| next crop 来源 | PASS | finalize 后 `self.state` 为 local；真实多帧 smoke 正常继续 |
| sender 来源 | PASS | runner 先生成三路 local candidate，再交换；reported output 不入 sender queue |
| legacy false 路径 | PASS | 单元测试确认仍提交 collaborative bbox |
| disabled/local 路径 | PASS | adapter 关闭时 local core 无 missing key，只有 9 个 adapter key 被忽略 |
| prediction-only 日志 | PASS | `uses_gt=false`，无 GT/IoU/label 字段 |
| CSV 保存 | PASS | A/B/C diagnostics 与 counterfactual CSV 均成功保存 |

Smoke 产物位于 `/tmp/plain-collaboration-safe-smoke-28200`，不属于正式结果目录，也未提交 Git。

本轮不涉及 backward、optimizer、checkpoint 保存或训练 smoke，因为任务明确禁止新训练；这些项目没有伪报为已执行。
