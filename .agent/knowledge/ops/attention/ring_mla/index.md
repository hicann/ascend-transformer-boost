# ring_mla — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "ring_mla"
  category: "attention"
  tier: "L"
  type: "single"

source:
  repo_path: "src/ops/ops_infer/ring_mla/"
  kernel_path: "src/kernels/mixkernels/ring_mla"
  param_header: "src/ops/ops_infer/ring_mla/param.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `ring_mla_operation.h` | Operation 定义 |
| 2 | `ring_mla_operation.cpp` | CreateRunner/InferShape |
| 3 | `ring_mla_ops_runner.h` | OpsRunner |
| 4 | `ring_mla_ops_runner.cpp` | Graph + Kernel（Ring 通信） |
| 5 | `param.h` | 参数定义 |
| 6 | `param.cpp` | 参数辅助逻辑 |
| 7 | `atb_acl_ring_mla.cpp` | ACL 辅助文件 |

## 2. Reading Order

```
[1] param.h → RingMLAVariantPackParam
[2] ring_mla_operation.h → 接口
[3] ring_mla_operation.cpp → CreateRunner + InferShape
[4] ring_mla_ops_runner.cpp → Ring 通信 + MLA Kernel
```

## 3. Parameter Constraints

```cpp
struct RingMLAVariantPackParam {
    std::vector<int32_t> qSeqLen;    // Q 序列长度
    std::vector<int32_t> kvSeqLen;   // KV 序列长度
    bool BuildFromTensor(const SVector<Mki::Tensor> &inTensors, size_t seqLenTensorId);
};
```

| 参数 | 类型 | 约束 |
|------|------|------|
| `qSeqLen` | vector<int32> | — |
| `kvSeqLen` | vector<int32> | — |

> 注意: Ring MLA 参数在本地 `param.h` 中定义，非全局 `infer_op_params.h`。主体参数通过 `VariantPack` 运行时传入。

## 4. Computation Pipeline

```yaml
pipeline_type: multi_stage
note: |
  Ring MLA（Multi-head Latent Attention with Ring topology）:
  多卡间通过 Ring 拓扑交换 KV Cache 分片，各卡本地计算 MLA Attention。
  涉及：Ring AllReduce（KV 通信）→ MLA Kernel（QKV Attention）。
```

## 5. Execution Paths

```
RingMlaOperation::CreateRunner()
  └── → RingMlaOpsRunner（单一路径，无 ACLNN）
         Kernel: ring_mla（含 Ring 通信 + MLA 计算）
```

| 平台 | Runner | 限制 |
|------|--------|------|
| 全部 | OpsRunner | 多卡 Ring 拓扑依赖 |

## 6. Kernel Dependencies

| Kernel | 文件 | 用途 |
|--------|------|------|
| `ring_mla` | `src/kernels/mixkernels/ring_mla/` | Ring 通信 + MLA 融合 Kernel |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 参数非标准（本地 param.h，非全局） | 注意 |
| 2 | 多卡 Ring 拓扑依赖 | lim |
| 3 | 无 ACLNN 加速路径 | lim |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `multi_latent_attention` | 单卡 MLA 版本 |
| `mla_preprocess` | MLA 前处理（QKV 投影 + RoPE + Cache） |
| `ring_mla` 是 MLA + Ring 通信的多卡分布式版本 |
