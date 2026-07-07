# rms_norm_with_stride — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "rms_norm_with_stride"
  category: "norm"
  tier: "S"
  type: "single"

source:
  repo_path: "src/ops/ops_infer/rms_norm_with_stride/"
  kernel_path: "src/kernels/mixkernels/laser_attention"
  param_header: "include/atb/infer_op_params.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `rms_norm_with_stride_operation.h` | Operation 定义 |
| 2 | `rms_norm_with_stride_operation.cpp` | CreateRunner/InferShape |
| 3 | `rms_norm_with_stride_ops_runner.h` | OpsRunner |
| 4 | `rms_norm_with_stride_ops_runner.cpp` | Graph + Kernel |

## 2. Reading Order

```
[1] operation.h → 接口 + 参数校验签名
[2] operation.cpp → CreateRunner + InferShape
[3] ops_runner.cpp → Graph 构建
```

## 3. Parameter Constraints

```cpp
struct RmsNormWithStrideParam {
    enum RmsNormType { RMS_NORM_UNDEFINED=0, RMS_NORM_NORM,
                       RMS_NORM_PRENORM, RMS_NORM_POSTNORM };
    enum PrecisionMode { HIGH_PRECISION_MODE=0, HIGH_PERFORMANCE_MODE };
    enum ModelType { LLAMA_MODEL=0, GEMMA_MODEL };
    struct NormParam {
        QuantType quantType = QUANT_UNQUANT;
        float epsilon = 1e-5;
        double layerNormEps = 1e-5;
        bool rstd = false;  // 训练 rmsnormforward 算子
    };
};
```

| 参数 | 约束 |
|------|------|
| `rmsNormType` | NORM/PRENORM/POSTNORM |
| `precisionMode` | HIGH_PRECISION(fp32) / HIGH_PERFORMANCE(fp16) |
| `modelType` | LLAMA / GEMMA（不同 rmsnorm 公式） |
| `rstd` | true 时使用训练版 rmsnormforward，仅 A2 推理产品 |
| bf16 | Atlas 推理系列产品不支持 |

## 4. Computation Pipeline

```yaml
pipeline_type: single_stage
note: "RMS Norm（无中心化）+ stride。GEMMA 模型使用不同公式。HIGH_PERFORMANCE_MODE 用 fp16 中间计算。"
```

## 5. Execution Paths

```
RmsNormWithStrideOperation::CreateRunner()
  └── → RmsNormWithStrideOpsRunner（单一路径）
         Kernel: laser_attention
```

## 6. Kernel Dependencies

| Kernel | 文件 |
|--------|------|
| `laser_attention` | `src/kernels/mixkernels/laser_attention/` |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | rstd 与 precisionMode/modelType 互斥 | lim |
| 2 | 量化场景不支持 rstd | lim |
| 3 | 无 ACLNN 加速路径 | lim |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `rms_norm` | 父模式：增加 stride + ModelType(LLAMA/GEMMA) |
| `layer_norm_with_stride` | 同类 stride 变体（LayerNorm vs RMS Norm） |
