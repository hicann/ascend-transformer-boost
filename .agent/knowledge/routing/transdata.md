# transdata — 路由文件

> **分类**: infer | **复杂度**: M | **文件数**: 6
> **Runner 类型**: OpsRunner,ACLNNRunner,Operation | **ACLNN**: yes
> **预估阅读时间**: 3-5 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `transdata_aclnn_runner.cpp` | ACLNN Runner |
| 2 | `transdata_aclnn_runner.h` | ACLNN Runner |
| 3 | `transdata_operation.cpp` | Operation 定义 |
| 4 | `transdata_operation.h` | Operation 定义 |
| 5 | `transdata_ops_runner.cpp` | Ops Runner |
| 6 | `transdata_ops_runner.h` | Ops Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `transdata_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `transdata_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `transdata_aclnn_runner.h` | ACLNN API 封装接口 |
| 4 | `transdata_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 5 | `transdata_ops_runner.h` | 原生 Ops 执行接口 |
| 6 | `transdata_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/transdata/`
- **Kernel 目录**: `src/kernels/kernels/transdata`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/transdata/index.md`](../ops/transdata/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
