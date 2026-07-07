# self_attention — 路由文件

> **分类**: infer | **复杂度**: XL | **文件数**: 27
> **Runner 类型**: OpsRunner,ACLNNRunner,Operation | **ACLNN**: yes
> **预估阅读时间**: 10-20 分钟

> **变体**: 910A, BNSD, Bypass, Encoder, Fusion, PrefixEncoder

---

## 1. 文件清单

| # | 文件 | 角色 |
|---|------|------|
| 1 | `atb_acl_self_attention_prefix_encoder.cpp` | 源码 |
| 2 | `param.cpp` | 源码 |
| 3 | `param.h` | 头文件 |
| 4 | `self_attention_aclnn_runner.cpp` | ACLNN Runner |
| 5 | `self_attention_aclnn_runner.h` | ACLNN Runner |
| 6 | `self_attention_encoder_fuison_ops_runner_910a.cpp` | Ops Runner |
| 7 | `self_attention_encoder_fusion_ops_runner.cpp` | Ops Runner |
| 8 | `self_attention_encoder_fusion_ops_runner.h` | Ops Runner |
| 9 | `self_attention_encoder_fusion_ops_runner_910a.h` | Ops Runner |
| 10 | `self_attention_fusion_bypass_ops_runner.cpp` | Ops Runner |
| 11 | `self_attention_fusion_bypass_ops_runner.h` | Ops Runner |
| 12 | `self_attention_fusion_bypass_ops_runner_910a.cpp` | Ops Runner |
| 13 | `self_attention_fusion_bypass_ops_runner_910a.h` | Ops Runner |
| 14 | `self_attention_fusion_bypass_ops_runner_BNSD.cpp` | Ops Runner |
| 15 | `self_attention_fusion_bypass_ops_runner_BNSD.h` | Ops Runner |
| 16 | `self_attention_fusion_bypass_ops_runner_BNSD_910a.cpp` | Ops Runner |
| 17 | `self_attention_fusion_bypass_ops_runner_BNSD_910a.h` | Ops Runner |
| 18 | `self_attention_fusion_ops_runner.cpp` | Ops Runner |
| 19 | `self_attention_fusion_ops_runner.h` | Ops Runner |
| 20 | `self_attention_fusion_ops_runner_910a.cpp` | Ops Runner |
| 21 | `self_attention_fusion_ops_runner_910a.h` | Ops Runner |
| 22 | `self_attention_operation.cpp` | Operation 定义 |
| 23 | `self_attention_operation.h` | Operation 定义 |
| 24 | `self_attention_prefix_encoder_ops_runner.cpp` | Ops Runner |
| 25 | `self_attention_prefix_encoder_ops_runner.h` | Ops Runner |
| 26 | `self_attention_runner_utils.cpp` | 源码 |
| 27 | `self_attention_runner_utils.h` | 头文件 |

## 2. 推荐阅读顺序

| 顺序 | 文件 | 重点关注 |
|------|------|---------|
| 1 | `self_attention_operation.h` | 了解输入输出数量、InferShape 签名 |
| 2 | `self_attention_operation.cpp` | CreateRunner() 决策逻辑 |
| 3 | `self_attention_aclnn_runner.h` | ACLNN API 封装接口 |
| 4 | `self_attention_aclnn_runner.cpp` | Workspace 计算 + ACLNN API 调用 |
| 5 | `self_attention_encoder_fusion_ops_runner.h` | 原生 Ops 执行接口 |
| 6 | `self_attention_encoder_fusion_ops_runner_910a.h` | 原生 Ops 执行接口 |
| 7 | `self_attention_fusion_bypass_ops_runner.h` | 原生 Ops 执行接口 |
| 8 | `self_attention_fusion_bypass_ops_runner_910a.h` | 原生 Ops 执行接口 |
| 9 | `self_attention_fusion_bypass_ops_runner_BNSD.h` | 原生 Ops 执行接口 |
| 10 | `self_attention_fusion_bypass_ops_runner_BNSD_910a.h` | 原生 Ops 执行接口 |
| 11 | `self_attention_fusion_ops_runner.h` | 原生 Ops 执行接口 |
| 12 | `self_attention_fusion_ops_runner_910a.h` | 原生 Ops 执行接口 |
| 13 | `self_attention_prefix_encoder_ops_runner.h` | 原生 Ops 执行接口 |
| 14 | `self_attention_encoder_fuison_ops_runner_910a.cpp` | 原生 Ops 调用链 + 平台适配 |
| 15 | `self_attention_encoder_fusion_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 16 | `self_attention_fusion_bypass_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 17 | `self_attention_fusion_bypass_ops_runner_910a.cpp` | 原生 Ops 调用链 + 平台适配 |
| 18 | `self_attention_fusion_bypass_ops_runner_BNSD.cpp` | 原生 Ops 调用链 + 平台适配 |
| 19 | `self_attention_fusion_bypass_ops_runner_BNSD_910a.cpp` | 原生 Ops 调用链 + 平台适配 |
| 20 | `self_attention_fusion_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 21 | `self_attention_fusion_ops_runner_910a.cpp` | 原生 Ops 调用链 + 平台适配 |
| 22 | `self_attention_prefix_encoder_ops_runner.cpp` | 原生 Ops 调用链 + 平台适配 |
| 23 | `atb_acl_self_attention_prefix_encoder.cpp` | 辅助文件 |
| 24 | `param.cpp` | 辅助文件 |
| 25 | `param.h` | 辅助文件 |
| 26 | `self_attention_runner_utils.cpp` | 辅助文件 |
| 27 | `self_attention_runner_utils.h` | 辅助文件 |

## 3. 源码路径

- **Op 目录**: `src/ops/ops_infer/self_attention/`
- **Kernel 目录**: `src/kernels/mixkernels/laser_attention`
- **参数头文件**: `include/atb/infer_op_params.h`

## 4. 变体说明

| 变体 | 说明 | 关联文件 |
|------|------|---------|
| 910A | 昇腾 910A 平台适配 | `self_attention_encoder_fuison_ops_runner_910a.cpp`, `self_attention_encoder_fusion_ops_runner_910a.h`, `self_attention_fusion_bypass_ops_runner_910a.cpp` |
| BNSD | BNSD 平台适配 | `self_attention_fusion_bypass_ops_runner_BNSD.cpp`, `self_attention_fusion_bypass_ops_runner_BNSD.h`, `self_attention_fusion_bypass_ops_runner_BNSD_910a.cpp` |
| Bypass | KVCache Bypass 路径 | `self_attention_fusion_bypass_ops_runner.cpp`, `self_attention_fusion_bypass_ops_runner.h`, `self_attention_fusion_bypass_ops_runner_910a.cpp` |
| Encoder | Encoder 计算路径 | `atb_acl_self_attention_prefix_encoder.cpp`, `self_attention_encoder_fuison_ops_runner_910a.cpp`, `self_attention_encoder_fusion_ops_runner.cpp` |
| Fusion | 融合算子路径 | `self_attention_encoder_fusion_ops_runner.cpp`, `self_attention_encoder_fusion_ops_runner.h`, `self_attention_encoder_fusion_ops_runner_910a.h` |
| PrefixEncoder | Prefix Encoder 融合路径 | - |

## 5. 知识条目

> 详细知识条目: [`ops/self_attention/index.md`](../ops/self_attention/index.md)

## 6. 快速导航

- [返回主索引](../README.md)
- [分类: infer](../README.md#infer)
