# reshape_and_cache — 路由文件

> **分类**: infer | **复杂度**: L | **文件数**: 14
> **Runner 类型**: OpsRunner,ACLNNRunner,Operation | **ACLNN**: yes
> **预估阅读时间**: 10-20 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `reshape_and_cache_aclnn_runner.cpp` | ACLNN Runner |
| 2 | `reshape_and_cache_aclnn_runner.h` | ACLNN Runner |
| 3 | `reshape_and_cache_operation.cpp` | Operation 定义 |
| 4 | `reshape_and_cache_operation.h` | Operation 定义 |
| 5 | `reshape_and_cache_ops_runner.cpp` | Ops Runner |
| 6 | `reshape_and_cache_ops_runner.h` | Ops Runner |
| 7 | `reshape_and_cache_ops_runner_310p.cpp` | Ops Runner |
| 8 | `reshape_and_cache_ops_runner_310p.h` | Ops Runner |
| 9 | `reshape_and_cache_ops_runner_A2_NZ.cpp` | Ops Runner |
| 10 | `reshape_and_cache_ops_runner_A2_NZ.h` | Ops Runner |
| 11 | `reshape_and_cache_ops_runner_SISO.cpp` | Ops Runner |
| 12 | `reshape_and_cache_ops_runner_SISO.h` | Ops Runner |
| 13 | `reshape_and_cache_siso_aclnn_runner.cpp` | ACLNN Runner |
| 14 | `reshape_and_cache_siso_aclnn_runner.h` | ACLNN Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `reshape_and_cache_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `reshape_and_cache_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `reshape_and_cache_aclnn_runner.h` | ACLNN API 封装接口 |
| 4 | `reshape_and_cache_siso_aclnn_runner.h` | ACLNN API 封装接口 |
| 5 | `reshape_and_cache_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 6 | `reshape_and_cache_siso_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 7 | `reshape_and_cache_ops_runner.h` | 原生 Ops 执行接口 |
| 8 | `reshape_and_cache_ops_runner_310p.h` | 原生 Ops 执行接口 |
| 9 | `reshape_and_cache_ops_runner_A2_NZ.h` | 原生 Ops 执行接口 |
| 10 | `reshape_and_cache_ops_runner_SISO.h` | 原生 Ops 执行接口 |
| 11 | `reshape_and_cache_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 12 | `reshape_and_cache_ops_runner_310p.cpp` | 原生 Ops 调用链 + 平台适配 |
| 13 | `reshape_and_cache_ops_runner_A2_NZ.cpp` | 原生 Ops 调用链 + 平台适配 |
| 14 | `reshape_and_cache_ops_runner_SISO.cpp` | 原生 Ops 调用链 + 平台适配 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/reshape_and_cache/`
- **Kernel 目录**: `src/kernels/mixkernels/reshape_and_cache`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/reshape_and_cache/index.md`](../ops/reshape_and_cache/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
