# gen_attention_mask — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "gen_attention_mask"
  category: "attention"
  tier: "S"
  type: "single"

source:
  repo_path: "src/ops/ops_train/gen_attention_mask/"
  kernel_path: "src/kernels/mixkernels/laser_attention"
  param_header: "include/atb/train_op_params.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `genattentionmask_operation.h` | Operation 定义 |
| 2 | `genattentionmask_operation.cpp` | CreateRunner/InferShape |
| 3 | `genattentionmask_ops_runner.h` | OpsRunner |
| 4 | `genattentionmask_ops_runner.cpp` | Graph + Kernel |

## 2. Reading Order

```
[1] operation.h → 接口签名
[2] operation.cpp → CreateRunner + InferShape
[3] ops_runner.cpp → Graph 构建 + Kernel 调用
```

## 3. Parameter Constraints

```cpp
struct GenAttentionMaskParam {
    int32_t headNum = 1;           // 多头注意力 head 数
    atb::SVector<int32_t> seqLen;  // 每个 batch 实际 seqlen（max 32）
};
```

| 参数 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| `headNum` | int32 | 1 | > 0 |
| `seqLen` | SVector<int32> | — | 元素数 ≤ batchSize ≤ 32 |

## 4. Computation Pipeline

```yaml
pipeline_type: single_stage
note: "生成注意力掩码矩阵（causal mask + padding mask），用于 SelfAttention 前处理"
```

## 5. Execution Paths

```
GenAttentionMaskOperation::CreateRunner()
  └── → GenAttentionMaskOpsRunner（单一路径）
         Kernel: laser_attention
```

## 6. Kernel Dependencies

| Kernel | 文件 |
|--------|------|
| `laser_attention` | `src/kernels/mixkernels/laser_attention/` |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 无 ACLNN 路径 | lim |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `laser_attention` | 下游：使用生成的 mask |
| `self_attention` | 对应推理侧 attention mask 生成 |
