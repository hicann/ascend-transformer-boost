# ring_mla — 路由文件

> **分类**: infer | **复杂度**: L | **文件数**: 7
> **Runner 类型**: OpsRunner,Operation | **ACLNN**: no
> **预估阅读时间**: 5-10 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `atb_acl_ring_mla.cpp` | 源码 |
| 2 | `param.cpp` | 源码 |
| 3 | `param.h` | 头文件 |
| 4 | `ring_mla_operation.cpp` | Operation 定义 |
| 5 | `ring_mla_operation.h` | Operation 定义 |
| 6 | `ring_mla_ops_runner.cpp` | Ops Runner |
| 7 | `ring_mla_ops_runner.h` | Ops Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `ring_mla_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `ring_mla_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `ring_mla_ops_runner.h` | 原生 Ops 执行接口 |
| 4 | `ring_mla_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 5 | `atb_acl_ring_mla.cpp` | 辅助文件 |
| 6 | `param.cpp` | 辅助文件 |
| 7 | `param.h` | 辅助文件 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/ring_mla/`
- **Kernel 目录**: `src/kernels/mixkernels/ring_mla`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/ring_mla/index.md`](../ops/ring_mla/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
