# E3 Target Semantic Prompt Collaboration 实现与 smoke

状态：**实现完成，联合回归与真实 checkpoint 2 帧 smoke 均 PASS；未启动长训练，未运行 Three-MDOT official test。**

- 预注册设计：`docs/plans/plain_collaboration_e3_target_semantic_prompt.md`
- 实现说明：`IMPLEMENTATION_ZH.md`
- smoke 明细：`SMOKE_TEST_ZH.md`
- trainable 审计：`TRAINABLE_PARAMS.txt`
- 通信核算：`COMMUNICATION_ACCOUNTING.md`
- 未来命令：`COMMANDS_ZH.md`
- provenance：`provenance.json`

核心结论：E3 用 sender 本地 CENTER raw score map 选择 K=8 search token；receiver 的 256 个 local search token 对两个 sender prompt 分别做共享 cross-attention 后等权均值。默认关闭；启用时强制 Safe Commit，reported output 为 E3、persistent state/next crop 为 local。
