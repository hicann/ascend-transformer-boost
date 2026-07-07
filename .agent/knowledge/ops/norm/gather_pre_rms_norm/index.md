# gather_pre_rms_norm — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "gather_pre_rms_norm"
  category: "norm"
  tier: "M"
  type: "fusion"

source:
  repo_path: "src/ops/ops_infer/gather_pre_rms_norm/"
  kernel_path: "src/kernels/kernels/norm/gatherprermsnorm"
  param_header: "include/atb/infer_op_params.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `gather_pre_rms_norm_operation.h` | Operation 定义 |
| 2 | `gather_pre_rms_norm_operation.cpp` | 参数校验、InferShape、CreateRunner |
| 3 | `gather_pre_rms_norm_ops_runner.h` | OpsRunner |
| 4 | `gather_pre_rms_norm_ops_runner.cpp` | Graph 构建 + Kernel |

## 2. Reading Order

```
[1] operation.h → 融合算子接口（Gather + Add + RMSNorm）
[2] operation.cpp → 两阶段参数校验 + CreateRunner
[3] ops_runner.cpp → Graph 构建（Gather→Add→RMSNorm）
```

## 3. Parameter Constraints

```cpp
struct GatherPreRmsNormParam {
    float epsilon = 1e-5;   // Norm epsilon
};
```

| 参数 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| `epsilon` | float | 1e-5 | > 0 |

**平台限制**: 仅 Atlas 800I A2 推理产品支持。

**输入**:
- `ResIn`: Gather 索引源
- `x`: 主输入 Tensor
- `gamma`: RMS Norm 权重
- `gatherIndices`: 索引张量

**流程**: `Gather(ResIn, indices) → Add(x) → RMS Norm`

## 4. Computation Pipeline

```yaml
pipeline_type: multi_stage_fusion
stages:
  - stage: Gather
    note: "ResIn[gatherIndices] → gatherOut"
  - stage: Add
    note: "x + gatherOut → sum"
  - stage: RMSNorm
    note: "rmsnorm(sum, gamma) → output"
    dtype: "fp16/bf16 in → fp32 acc → fp16/bf16 out"
```

## 5. Execution Paths

```
GatherPreRmsNormOperation::CreateRunner()
  └── → GatherPreRmsNormOpsRunner（单一 OpsRunner，仅 A2）
         Kernel: gatherprermsnorm
```

| 平台 | Runner | 限制 |
|------|--------|------|
| Atlas A2 | OpsRunner | **仅 A2** |

## 6. Kernel Dependencies

| Kernel | 文件 |
|--------|------|
| `gatherprermsnorm` | `src/kernels/kernels/norm/gatherprermsnorm/` |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 仅 Atlas A2 推理产品支持 | lim |
| 2 | 参数极少（仅 epsilon），三阶段融合 | 注意 |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `rms_norm` | 融合的 RMS Norm 阶段 |
| `gather` | 融合的第一个阶段 |
| `gather_pre_rms_norm` 是 Gather + Add + RMSNorm 的三阶段融合版 |
