# event_operation — 路由文件

> **分类**: common | **复杂度**: S | **文件数**: 4
> **Runner 类型**:  | **ACLNN**: no
> **预估阅读时间**: 3-5 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `event_operation.cpp` | Operation 定义 |
| 2 | `event_operation.h` | Operation 定义 |
| 3 | `event_runner.cpp` | 源码 |
| 4 | `event_runner.h` | 头文件 |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `event_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `event_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `event_runner.cpp` | 辅助文件 |
| 4 | `event_runner.h` | 辅助文件 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_common/event_operation/`
- **Kernel 目录**: `src/kernels/mixkernels/laser_attention`
- **参数头文件**: `include/atb/train_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/event_operation/index.md`](../ops/event_operation/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: common](../README.md#common)
