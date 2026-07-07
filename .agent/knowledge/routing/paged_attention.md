# paged_attention — 路由文件

> **分类**: infer | **复杂度**: L | **文件数**: 12
> **Runner 类型**: OpsRunner,ACLNNRunner,Operation | **ACLNN**: yes
> **预估阅读时间**: 5-10 分钟

> **变体**: 910A

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `paged_attention_aclnn_runner.cpp` | ACLNN Runner |
| 2 | `paged_attention_aclnn_runner.h` | ACLNN Runner |
| 3 | `paged_attention_operation.cpp` | Operation 定义 |
| 4 | `paged_attention_operation.h` | Operation 定义 |
| 5 | `paged_attention_ops_runner.cpp` | Ops Runner |
| 6 | `paged_attention_ops_runner.h` | Ops Runner |
| 7 | `paged_attention_ops_runner_910a.cpp` | Ops Runner |
| 8 | `paged_attention_ops_runner_910a.h` | Ops Runner |
| 9 | `paged_attention_runner_utils.cpp` | 源码 |
| 10 | `paged_attention_runner_utils.h` | 头文件 |
| 11 | `param.cpp` | 源码 |
| 12 | `param.h` | 头文件 |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `paged_attention_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `paged_attention_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `paged_attention_aclnn_runner.h` | ACLNN API 封装接口 |
| 4 | `paged_attention_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 5 | `paged_attention_ops_runner.h` | 原生 Ops 执行接口 |
| 6 | `paged_attention_ops_runner_910a.h` | 原生 Ops 执行接口 |
| 7 | `paged_attention_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 8 | `paged_attention_ops_runner_910a.cpp` | 原生 Ops 调用链 + 平台适配 |
| 9 | `paged_attention_runner_utils.cpp` | 辅助文件 |
| 10 | `paged_attention_runner_utils.h` | 辅助文件 |
| 11 | `param.cpp` | 辅助文件 |
| 12 | `param.h` | 辅助文件 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/paged_attention/`
- **Kernel 目录**: `src/kernels/mixkernels/laser_attention`
- **参数头文件**: `include/atb/infer_op_params.h`

## 4. 变体说明

| 变体 | 说明 | 关联文件 |
|------|------|---------|
| 910A | 昇腾 910A 平台适配 | `paged_attention_ops_runner_910a.cpp`, `paged_attention_ops_runner_910a.h` |

## 5. 知识条目

> 详细知识条目: [`ops/paged_attention/index.md`](../ops/paged_attention/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
