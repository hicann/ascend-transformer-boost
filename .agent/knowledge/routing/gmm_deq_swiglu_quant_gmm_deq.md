# gmm_deq_swiglu_quant_gmm_deq — 路由文件

> **分类**: infer | **复杂度**: M | **文件数**: 4
> **Runner 类型**: OpsRunner,Operation | **ACLNN**: no
> **预估阅读时间**: 3-5 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `gmm_deq_swiglu_quant_gmm_deq_operation.cpp` | Operation 定义 |
| 2 | `gmm_deq_swiglu_quant_gmm_deq_operation.h` | Operation 定义 |
| 3 | `gmm_deq_swiglu_quant_gmm_deq_ops_runner.cpp` | Ops Runner |
| 4 | `gmm_deq_swiglu_quant_gmm_deq_ops_runner.h` | Ops Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `gmm_deq_swiglu_quant_gmm_deq_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `gmm_deq_swiglu_quant_gmm_deq_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `gmm_deq_swiglu_quant_gmm_deq_ops_runner.h` | 原生 Ops 执行接口 |
| 4 | `gmm_deq_swiglu_quant_gmm_deq_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/gmm_deq_swiglu_quant_gmm_deq/`
- **Kernel 目录**: `src/kernels/mixkernels/gmm_deq_swiglu_quant_gmm_deq`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/gmm_deq_swiglu_quant_gmm_deq/index.md`](../ops/gmm_deq_swiglu_quant_gmm_deq/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
