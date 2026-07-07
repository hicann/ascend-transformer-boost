# multi_latent_attention — 路由文件

> **分类**: infer | **复杂度**: M | **文件数**: 9
> **Runner 类型**: OpsRunner,Operation | **ACLNN**: no
> **预估阅读时间**: 5-10 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `atb_acl_mla.cpp` | 源码 |
| 2 | `multi_latent_attention_operation.cpp` | Operation 定义 |
| 3 | `multi_latent_attention_operation.h` | Operation 定义 |
| 4 | `multi_latent_attention_ops_runner.cpp` | Ops Runner |
| 5 | `multi_latent_attention_ops_runner.h` | Ops Runner |
| 6 | `multi_latent_attention_ops_runner_prefill.cpp` | Ops Runner |
| 7 | `multi_latent_attention_ops_runner_prefill.h` | Ops Runner |
| 8 | `param.cpp` | 源码 |
| 9 | `param.h` | 头文件 |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `multi_latent_attention_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `multi_latent_attention_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `multi_latent_attention_ops_runner.h` | 原生 Ops 执行接口 |
| 4 | `multi_latent_attention_ops_runner_prefill.h` | 原生 Ops 执行接口 |
| 5 | `multi_latent_attention_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 6 | `multi_latent_attention_ops_runner_prefill.cpp` | 原生 Ops 调用链 + 平台适配 |
| 7 | `atb_acl_mla.cpp` | 辅助文件 |
| 8 | `param.cpp` | 辅助文件 |
| 9 | `param.h` | 辅助文件 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/multi_latent_attention/`
- **Kernel 目录**: `src/kernels/mixkernels/multi_latent_attention`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/multi_latent_attention/index.md`](../ops/multi_latent_attention/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
