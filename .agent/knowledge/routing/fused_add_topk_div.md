# fused_add_topk_div — 路由文件

> **分类**: infer | **复杂度**: M | **文件数**: 5
> **Runner 类型**: OpsRunner,Operation | **ACLNN**: no
> **预估阅读时间**: 3-5 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `atb_acl_fused_add_topk_div.cpp` | 源码 |
| 2 | `fused_add_topk_div_operation.cpp` | Operation 定义 |
| 3 | `fused_add_topk_div_operation.h` | Operation 定义 |
| 4 | `fused_add_topk_div_ops_runner.cpp` | Ops Runner |
| 5 | `fused_add_topk_div_ops_runner.h` | Ops Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `fused_add_topk_div_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `fused_add_topk_div_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `fused_add_topk_div_ops_runner.h` | 原生 Ops 执行接口 |
| 4 | `fused_add_topk_div_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 5 | `atb_acl_fused_add_topk_div.cpp` | 辅助文件 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/fused_add_topk_div/`
- **Kernel 目录**: `src/kernels/mixkernels/fused_add_topk_div`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/fused_add_topk_div/index.md`](../ops/fused_add_topk_div/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
