# index_add — 路由文件

> **分类**: infer | **复杂度**: S | **文件数**: 4
> **Runner 类型**: OpsRunner,Operation | **ACLNN**: no
> **预估阅读时间**: 3-5 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `index_add_operation.cpp` | Operation 定义 |
| 2 | `index_add_operation.h` | Operation 定义 |
| 3 | `index_add_ops_runner.cpp` | Ops Runner |
| 4 | `index_add_ops_runner.h` | Ops Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `index_add_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `index_add_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `index_add_ops_runner.h` | 原生 Ops 执行接口 |
| 4 | `index_add_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/index_add/`
- **Kernel 目录**: `src/kernels/mixkernels/laser_attention`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/index_add/index.md`](../ops/index_add/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
