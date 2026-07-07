# linear — 路由文件

> **分类**: infer | **复杂度**: L | **文件数**: 10
> **Runner 类型**: OpsRunner,ACLNNRunner,Operation | **ACLNN**: yes
> **预估阅读时间**: 5-10 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `linear_aclnn_runner.cpp` | ACLNN Runner |
| 2 | `linear_aclnn_runner.h` | ACLNN Runner |
| 3 | `linear_dequant_aclnn_runner.cpp` | ACLNN Runner |
| 4 | `linear_dequant_aclnn_runner.h` | ACLNN Runner |
| 5 | `linear_einsum_aclnn_runner.cpp` | ACLNN Runner |
| 6 | `linear_einsum_aclnn_runner.h` | ACLNN Runner |
| 7 | `linear_operation.cpp` | Operation 定义 |
| 8 | `linear_operation.h` | Operation 定义 |
| 9 | `linear_ops_runner.cpp` | Ops Runner |
| 10 | `linear_ops_runner.h` | Ops Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `linear_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `linear_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `linear_aclnn_runner.h` | ACLNN API 封装接口 |
| 4 | `linear_dequant_aclnn_runner.h` | ACLNN API 封装接口 |
| 5 | `linear_einsum_aclnn_runner.h` | ACLNN API 封装接口 |
| 6 | `linear_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 7 | `linear_dequant_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 8 | `linear_einsum_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 9 | `linear_ops_runner.h` | 原生 Ops 执行接口 |
| 10 | `linear_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/linear/`
- **Kernel 目录**: `src/kernels/mixkernels/laser_attention`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/linear/index.md`](../ops/linear/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
