# fill — 路由文件

> **分类**: infer | **复杂度**: M | **文件数**: 8
> **Runner 类型**: OpsRunner,ACLNNRunner | **ACLNN**: yes
> **预估阅读时间**: 5-10 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `fill_aclnn_runner.cpp` | ACLNN Runner |
| 2 | `fill_aclnn_runner.h` | ACLNN Runner |
| 3 | `fill_operation.cpp` | Operation 定义 |
| 4 | `fill_operation.h` | Operation 定义 |
| 5 | `fill_ops_runner.cpp` | Ops Runner |
| 6 | `fill_ops_runner.h` | Ops Runner |
| 7 | `masked_fill_aclnn_runner.cpp` | ACLNN Runner |
| 8 | `masked_fill_aclnn_runner.h` | ACLNN Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `fill_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `fill_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `fill_aclnn_runner.h` | ACLNN API 封装接口 |
| 4 | `masked_fill_aclnn_runner.h` | ACLNN API 封装接口 |
| 5 | `fill_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 6 | `masked_fill_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 7 | `fill_ops_runner.h` | 原生 Ops 执行接口 |
| 8 | `fill_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/fill/`
- **Kernel 目录**: `src/kernels/kernels/fill`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/fill/index.md`](../ops/fill/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
