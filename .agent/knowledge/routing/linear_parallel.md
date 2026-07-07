# linear_parallel — 路由文件

> **分类**: infer | **复杂度**: L | **文件数**: 8
> **Runner 类型**: OpsRunner,ACLNNRunner,Operation,Kernel | **ACLNN**: yes
> **预估阅读时间**: 5-10 分钟

> **变体**: Graph, LCOC

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `linear_parallel_aclnn_runner.cpp` | ACLNN Runner |
| 2 | `linear_parallel_aclnn_runner.h` | ACLNN Runner |
| 3 | `linear_parallel_graph_runner.cpp` | 源码 |
| 4 | `linear_parallel_graph_runner.h` | 头文件 |
| 5 | `linear_parallel_lcoc_runner.cpp` | 源码 |
| 6 | `linear_parallel_lcoc_runner.h` | 头文件 |
| 7 | `linear_parallel_operation.cpp` | Operation 定义 |
| 8 | `linear_parallel_operation.h` | Operation 定义 |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `linear_parallel_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `linear_parallel_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `linear_parallel_aclnn_runner.h` | ACLNN API 封装接口 |
| 4 | `linear_parallel_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 5 | `linear_parallel_graph_runner.cpp` | 辅助文件 |
| 6 | `linear_parallel_graph_runner.h` | 辅助文件 |
| 7 | `linear_parallel_lcoc_runner.cpp` | 辅助文件 |
| 8 | `linear_parallel_lcoc_runner.h` | 辅助文件 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/linear_parallel/`
- **Kernel 目录**: `src/kernels/mixkernels/laser_attention`
- **参数头文件**: `include/atb/infer_op_params.h`

## 4. 变体说明

| 变体 | 说明 | 关联文件 |
|------|------|---------|
| Graph | 图模式执行路径 | `linear_parallel_graph_runner.cpp`, `linear_parallel_graph_runner.h` |
| LCOC | LCOC 执行路径 | `linear_parallel_lcoc_runner.cpp`, `linear_parallel_lcoc_runner.h` |

## 5. 知识条目

> 详细知识条目: [`ops/linear_parallel/index.md`](../ops/linear_parallel/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
