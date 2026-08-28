# 理论通信 payload 核算

仅计算每 sender、每 event 的 tensor payload；不包含协议头、序列化、传输栈、重传或压缩开销，因此不等同于实测带宽。

| 方法 | shape | token/element | FP32 bytes | FP16 bytes |
|---|---|---:|---:|---:|
| V1 full search | `[256,192]` | 49,152 | 196,608 | 98,304 |
| E3 target prompt | `[8,192]` | 1,536 | 6,144 | 3,072 |

计算：

```text
196608 / 6144 = 32
98304 / 3072 = 32
256 / 8 = 32
```

E3 相对 V1 为严格 `32x` token/payload reduction。压缩本身不是成功条件；未来仍须满足 AUC、per-target safety 与 V1 对照门槛。
