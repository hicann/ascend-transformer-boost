# layer_norm_with_stride — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "layer_norm_with_stride"
  category: "norm"
  tier: "S"
  type: "single"

source:
  repo_path: "src/ops/ops_infer/layer_norm_with_stride/"
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
| 1 | `layer_norm_with_stride_operation.h` | Operation 定义 |
| 2 | `layer_norm_with_stride_operation.cpp` | CreateRunner/InferShape |
| 3 | `layer_norm_with_stride_ops_runner.h` | OpsRunner |
| 4 | `layer_norm_with_stride_ops_runner.cpp` | Graph + Kernel |

## 2. Reading Order

```
[1] operation.h → 接口签名
[2] operation.cpp → CreateRunner + InferShape
[3] ops_runner.cpp → Graph 构建
```

## 3. Parameter Constraints

```cpp
struct LayerNormWithStrideParam {
    enum LayerNormType { LAYER_NORM_UNDEFINED=0, LAYER_NORM_NORM,
                         LAYER_NORM_PRENORM, LAYER_NORM_POSTNORM };
    struct NormParam {
        QuantType quantType = QUANT_UNQUANT;       // QUANT_UNQUANT | QUANT_INT8
        float epsilon = 1e-5;
        int32_t beginNormAxis = 0;
        int32_t beginParamsAxis = 0;
    };
};
```

| 参数 | 约束 |
|------|------|
| `layerType` | NORM/PRENORM/POSTNORM |
| `quantType` | QUANT_UNQUANT / QUANT_INT8 |
| `epsilon` | ≥ 2e-38 |
| Shape | 最后维对齐 32B（量化模式） |
| bf16 | Atlas 推理系列产品不支持 |

## 4. Computation Pipeline

```yaml
pipeline_type: single_stage
note: "与 layer_norm 相同，增加 stride 参数支持非连续 Tensor"
```

## 5. Execution Paths

```
LayerNormWithStrideOperation::CreateRunner()
  └── → LayerNormWithStrideOpsRunner（单一路径，无 ACLNN）
         Kernel: laser_attention
```

| 平台 | Runner | 限制 |
|------|--------|------|
| 全部 | OpsRunner | 无 ACLNN 路径 |

## 6. Kernel Dependencies

| Kernel | 文件 |
|--------|------|
| `laser_attention` | `src/kernels/mixkernels/laser_attention/` |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 无 ACLNN 加速路径 | lim |
| 2 | Atlas 推理产品不支持 bf16 | lim |

## 8. Test Coverage

ATK 测试: `atk_test/atk_cida_atb/atb/infer/LayerNormWithStrideOperation/`

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `layer_norm` | 父模式：增加 stride 支持 |
