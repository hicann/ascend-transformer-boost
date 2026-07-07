# rms_norm — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "rms_norm"
  category: "norm"
  tier: "L"
  type: "single"

source:
  repo_path: "src/ops/ops_infer/rms_norm/"
  kernel_path: "src/kernels/kernels/norm/rmsnorm"
  param_header: "include/atb/infer_op_params.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"

test:
  atk_test_path: "atk_test/atk_cida_atb/atb/infer/RmsNormOperation/"
  has_atk_tests: true
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `rms_norm_operation.h` | Operation 定义 |
| 2 | `rms_norm_operation.cpp` | 参数校验、InferShape、CreateRunner（3 路 ACLNN 决策） |
| 3 | `rms_norm_aclnn_runner.h/cpp` | ACLNN Runner（950 NORM 非量化） |
| 4 | `rms_norm_quant_aclnn_runner.h/cpp` | ACLNN Runner（950 NORM 量化） |
| 5 | `add_rms_norm_aclnn_runner.h/cpp` | ACLNN Runner（950 PRENORM） |
| 6 | `rms_norm_ops_runner.h/cpp` | OpsRunner（fallback + legacy） |
| 7 | `rmsnorm_kernel.cpp` | Kernel |

## 2. Reading Order

```
[1] rms_norm_operation.h → 接口 + 变量声明
[2] rms_norm_operation.cpp → CreateRunner 决策树（3 路 ACLNN + 1 Ops）
[3] rms_norm_aclnn_runner.cpp → 标准 ACLNN 路径
[4] rms_norm_quant_aclnn_runner.cpp → 量化 ACLNN 路径
[5] add_rms_norm_aclnn_runner.cpp → PRENORM ACLNN 路径
[6] rms_norm_ops_runner.cpp → Ops Graph 构建
```

## 3. Parameter Constraints

```cpp
struct RmsNormParam {
    enum RmsNormType { RMS_NORM_UNDEFINED=0, RMS_NORM_NORM,
                       RMS_NORM_PRENORM, RMS_NORM_POSTNORM };
    enum PrecisionMode { HIGH_PRECISION_MODE=0, HIGH_PERFORMANCE_MODE };
    enum ModelType { LLAMA_MODEL=0, GEMMA_MODEL };
    struct NormParam {
        QuantType quantType = QUANT_UNQUANT;
        float epsilon = 1e-5;
        double layerNormEps = 1e-5;
        bool rstd = false;
        PrecisionMode precisionMode = HIGH_PRECISION_MODE;
        ModelType modelType = LLAMA_MODEL;
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
    };
    // + PreNormParam / PostNormParam (类似 layer_norm)
};
```

| 参数 | 约束 |
|------|------|
| `rmsNormType` | NORM/PRENORM/POSTNORM |
| `precisionMode` | HIGH_PRECISION(fp32) / HIGH_PERFORMANCE(fp16) |
| `modelType` | LLAMA / GEMMA 公式 |
| `rstd` | 训练版 rmsnormforward，仅 A2，与 precisionMode/modelType 互斥 |
| `dynamicQuantType` | UNDEFINED / SYMMETRIC（不支持 ASYMMETRIC） |
| `epsilon` | ≥ 2e-38 |
| bf16 | Atlas 推理系列不支持 |
| lastDim 对齐 | 32B（量化/PRENORM/POSTNORM） |

## 4. Computation Pipeline

```yaml
pipeline_type: single_stage
note: "RMS Norm = 无中心化 LayerNorm（无 mean 减除，仅 scale）。HIGH_PERFORMANCE_MODE 用 fp16 中间计算。"
```

## 5. Execution Paths

```
RmsNormOperation::CreateRunner()
  │
  ├── [950 + NORM + QUANT_UNQUANT] → RmsNormAclnnRunner
  ├── [950 + NORM + QUANT_INT8]    → RmsNormQuantAclnnRunner
  ├── [950 + PRENORM]              → AddRmsNormAclnnRunner
  └── [fallback]                   → RmsNormOpsRunner
```

| 平台 | Runner | 支持 |
|------|--------|------|
| Atlas 910B | OpsRunner | NORM/PRENORM/POSTNORM + 量化 + 动态量化 |
| Atlas 950 | ACLNN × 3 | NORM(非量化+量化) + PRENORM；POSTNORM → OpsRunner |

## 6. Kernel Dependencies

| Kernel | 文件 | 用途 |
|--------|------|------|
| `rmsnorm` | `src/kernels/kernels/norm/rmsnorm/` | RMS Norm 计算 |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | rstd 与 precisionMode/modelType 互斥 | lim |
| 2 | 动态量化不支持 ASYMMETRIC | lim |
| 3 | HIGH_PERFORMANCE_MODE 与 rstd/modelType 互斥 | lim |
| 4 | quantType + precisionMode/modelType 不可同时设置 | lim |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `layer_norm` | 全功能 LayerNorm（含 mean 减除） |
| `rms_norm_with_stride` | stride 变体 |
| `rms_norm_backward` | 反向传播 |
| `gather_pre_rms_norm` | Gather + Add + RMSNorm 融合 |
| `norm_rope_reshape` | RMSNorm + RoPE + Reshape 融合 |
