# mla_preprocess — 路由文件

> **分类**: infer | **复杂度**: L | **文件数**: 9
> **Runner 类型**: OpsRunner,ACLNNRunner,Operation | **ACLNN**: yes
> **预估阅读时间**: 5-10 分钟

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `atb_acl_mla_preprocess.cpp` | 源码 |
| 2 | `mla_preprocess_aclnn_runner.cpp` | ACLNN Runner |
| 3 | `mla_preprocess_aclnn_runner.h` | ACLNN Runner |
| 4 | `mla_preprocess_operation.cpp` | Operation 定义 |
| 5 | `mla_preprocess_operation.h` | Operation 定义 |
| 6 | `mla_preprocess_ops_runner.cpp` | Ops Runner |
| 7 | `mla_preprocess_ops_runner.h` | Ops Runner |
| 8 | `mla_preprocess_ops_runner_split.cpp` | Ops Runner |
| 9 | `mla_preprocess_ops_runner_split.h` | Ops Runner |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `mla_preprocess_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `mla_preprocess_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `mla_preprocess_aclnn_runner.h` | ACLNN API 封装接口 |
| 4 | `mla_preprocess_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 5 | `mla_preprocess_ops_runner.h` | 原生 Ops 执行接口 |
| 6 | `mla_preprocess_ops_runner_split.h` | 原生 Ops 执行接口 |
| 7 | `mla_preprocess_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 8 | `mla_preprocess_ops_runner_split.cpp` | 原生 Ops 调用链 + 平台适配 |
| 9 | `atb_acl_mla_preprocess.cpp` | 辅助文件 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/mla_preprocess/`
- **Kernel 目录**: `src/kernels/mixkernels/mla_preprocess`
- **参数头文件**: `include/atb/infer_op_params.h`

## 5. 知识条目

> 详细知识条目: [`ops/mla_preprocess/index.md`](../ops/mla_preprocess/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
