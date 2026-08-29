# E3-D1 实现与 smoke 结论

状态：**PASS，可以进入后续单独授权的 25-epoch 训练；本任务未启动训练。**

E3-D1 已按预注册实现为唯一训练期变量：每个 rank 的 `B=2` synchronized ABC batch 中，随机选择恰好 `1/2` triplet，并在每个 selected triplet 中均匀随机选择 exactly one view，仅将该 view 的 normalized search target bbox 填为 `0.0`。template、annotation、另外两个 view、sampler、模型、loss 和推理均不变。

冻结确认：

- B0 ep25 strict core initialization，SHA256 `363fb06b05e4f0591ca1efca5b31ad38c7f5e0865048bb546dcd28dd0463edd3`；
- E3 `K=8`、148,993 adapter-only parameters、LR `8e-5`；
- Plain ViT-Tiny、完整 256 search tokens、CENTER `16x16`；
- common-visible、canonical ABC、view-major flatten；
- loss `GIoU/L1/Focal = 2/5/1`、batch/epoch/samples、residual 和 Safe Commit均未改变；
- validation/inference exact bypass；
- official test 未访问，长期训练未启动。

详细证据：

- [DATA_FLOW_AUDIT_ZH.md](DATA_FLOW_AUDIT_ZH.md)
- [IMPLEMENTATION_ZH.md](IMPLEMENTATION_ZH.md)
- [SMOKE_TEST_ZH.md](SMOKE_TEST_ZH.md)
- [COMMANDS_ZH.md](COMMANDS_ZH.md)
- [provenance.json](provenance.json)
