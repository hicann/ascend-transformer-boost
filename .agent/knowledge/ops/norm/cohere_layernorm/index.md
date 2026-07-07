# cohere_layernorm — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "cohere_layernorm"
  category: "norm"
  tier: "S"
  type: "single"

source:
  repo_path: "src/ops/ops_infer/cohere_layernorm/"
  kernel_path: "src/kernels/kernels/norm/coherelayernorm"
  param_header: "include/atb/infer_op_params.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"

test:
  atk_test_path: "atk_test/atk_cida_atb/atb/infer/CohereLayerNormOperation/"
  has_atk_tests: true
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `cohere_layernorm_operation.h` | Operation 定义 |
| 2 | `cohere_layernorm_operation.cpp` | 参数校验、InferShape、CreateRunner |
| 3 | `cohere_layernorm_runner.cpp` | Ops Runner 实现 |
| 4 | `cohere_layernorm_runner.h` | Ops Runner 声明 |

## 2. Reading Order

```
[1] cohere_layernorm_operation.h  → 接口签名
[2] cohere_layernorm_operation.cpp → CreateRunner + InferShape
[3] cohere_layernorm_runner.cpp    → Graph 构建 + Kernel 调用
```

## 3. Parameter Constraints

```cpp
struct CohereLayerNormParam {
    float epsilon = 1e-5;   // 归一化 epsilon
};
```

| 参数 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| `epsilon` | float | 1e-5 | > 0 |

**Shape 约束**: 针对 Command R Plus 模型，多 batch 数据按最后一维归一化。

## 4. Computation Pipeline

```yaml
pipeline_type: single_stage
note: "Cohere 自定义 LayerNorm，无 dtype 转换"
```

## 5. Execution Paths

```
CohereLayerNormOperation::CreateRunner()
  └── → CohereLayerNormRunner（单一 OpsRunner 路径）
         Kernel: coherelayernorm
```

| 平台 | Runner | 限制 |
|------|--------|------|
| Atlas 910B/950 | CohereLayerNormRunner | — |

## 6. Kernel Dependencies

| Kernel | 文件 | 用途 |
|--------|------|------|
| `coherelayernorm` | `src/kernels/kernels/norm/coherelayernorm/` | Norm 计算 |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 参数极少（仅 epsilon），无量化支持 | lim |

## 8. Test Coverage

- ATK 测试: `atk_test/atk_cida_atb/atb/infer/CohereLayerNormOperation/`

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `layer_norm` | 父模式：标准 LayerNorm（3 种 NormType + 量化） |
| `cohere_layernorm` 是 layer_norm 的简化版（仅 NORM，仅 epsilon 参数） |
