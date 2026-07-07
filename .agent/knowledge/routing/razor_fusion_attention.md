# razor_fusion_attention — 路由文件

> **分类**: infer | **复杂度**: S | **文件数**: 4
> **Runner 类型**: OpsRunner,Operation | **ACLNN**: no
> **预估阅读时间**: 3-5 分钟

> **变体**: Fusion

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `razor_fusion_attention_operation.cpp` | Operation 定义 |
| 2 | `razor_fusion_attention_operation.h` | Operation 定义 |
| 3 | `razor_fusion_attention_ops_runner.cpp` | Ops Runner |
| 4 | `razor_fusion_attention_ops_runner.h` | Ops Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `razor_fusion_attention_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `razor_fusion_attention_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `razor_fusion_attention_ops_runner.h` | 原生 Ops 执行接口 |
| 4 | `razor_fusion_attention_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/razor_fusion_attention/`
- **Kernel 目录**: `src/kernels/mixkernels/laser_attention`
- **参数头文件**: `include/atb/infer_op_params.h`

## 4. 变体说明

| 变体 | 说明 | 关联文件 |
|------|------|---------|
| Fusion | 融合算子路径 | `razor_fusion_attention_operation.cpp`, `razor_fusion_attention_operation.h`, `razor_fusion_attention_ops_runner.cpp` |

## 5. 知识条目

> 详细知识条目: [`ops/razor_fusion_attention/index.md`](../ops/razor_fusion_attention/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
