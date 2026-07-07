# laser_attention_grad — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "laser_attention_grad"
  category: "attention"
  tier: "S"
  type: "single"

source:
  repo_path: "src/ops/ops_train/laser_attention_grad/"
  kernel_path: "src/kernels/mixkernels/laser_attention_grad"
  param_header: "include/atb/train_op_params.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `laser_attention_grad_operation.h` | Operation 定义 |
| 2 | `laser_attention_grad_operation.cpp` | CreateRunner/InferShape |
| 3 | `laser_attention_grad_ops_runner.h` | OpsRunner |
| 4 | `laser_attention_grad_ops_runner.cpp` | Graph + Kernel |

## 2. Reading Order

```
[1] operation.h → 梯度接口
[2] operation.cpp → CreateRunner + InferShape
[3] ops_runner.cpp → Gradient Graph + Kernel
```

## 3. Parameter Constraints

```cpp
struct LaserAttentionGradParam {
    int32_t headNum = 0;              // head 个数，> 0
    std::vector<int32_t> qSeqLen;     // 每个 batch 实际 seqlen（≤ 32）
};
```

| 参数 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| `headNum` | int32 | 0 | > 0 |
| `qSeqLen` | vector<int32> | — | 元素数 ≤ batchSize ≤ 32 |

**平台限制**: 仅 Atlas 800I A2 推理产品支持。

## 4. Computation Pipeline

```yaml
pipeline_type: single_stage
note: "LaserAttention 反向传播。梯度从 QKV 回传到输入。"
```

## 5. Execution Paths

```
LaserAttentionGradOperation::CreateRunner()
  └── → LaserAttentionGradOpsRunner
         Kernel: laser_attention_grad
```

## 6. Kernel Dependencies

| Kernel | 文件 |
|--------|------|
| `laser_attention_grad` | `src/kernels/mixkernels/laser_attention_grad/` |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 仅 A2 支持 | lim |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `laser_attention` | 前向 |
| `self_attention` | 推理侧对应（但 SA 无单独 backward Op） |
| `fast_soft_max_grad` | 同训练侧 Softmax 反向 |
