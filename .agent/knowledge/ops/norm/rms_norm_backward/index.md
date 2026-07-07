# rms_norm_backward — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "rms_norm_backward"
  category: "norm"
  tier: "S"
  type: "single"

source:
  repo_path: "src/ops/ops_train/rms_norm_backward/"
  kernel_path: "src/kernels/kernels/norm/rmsnormbackward"
  param_header: "include/atb/train_op_params.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"

test:
  atk_test_path: "atk_test/atk_cida_atb/atb/train/RmsNormBackwardOperation/"
  has_atk_tests: true
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `rms_norm_backward_operation.h` | Operation 定义 |
| 2 | `rms_norm_backward_operation.cpp` | CreateRunner/InferShape |
| 3 | `rms_norm_backward_ops_runner.h` | OpsRunner |
| 4 | `rms_norm_backward_ops_runner.cpp` | Graph + Kernel |

## 2. Reading Order

```
[1] operation.h → 梯度输入输出接口
[2] operation.cpp → CreateRunner + 梯度维度校验
[3] ops_runner.cpp → Backward Graph 构建
```

## 3. Parameter Constraints

```cpp
struct RmsNormBackwardParam {
    // 无参数，仅预留字段
    uint8_t rsv[8] = {0};
};
```

| 参数 | 说明 |
|------|------|
| （无） | 反向算子，参数由前向 `rms_norm` 的上下文决定 |

**平台限制**: 仅 Atlas 800I A2 推理产品支持。

**输入约束**:
- `dy`: 输出梯度，shape 同 `y`（前向输出）
- `x`: 前向输入
- `gamma`: RMS Norm 权重，shape `[lastDim]`

**输出**:
- `dx`: 输入梯度，shape 同 `x`

## 4. Computation Pipeline

```yaml
pipeline_type: single_stage
note: "RMS Norm 反向传播：计算 dx = ∂L/∂x = RMSNormBackward(dy, x, gamma)"
```

## 5. Execution Paths

```
RmsNormBackwardOperation::CreateRunner()
  └── → RmsNormBackwardOpsRunner（单一路径）
         Kernel: rmsnormbackward
```

| 平台 | Runner | 限制 |
|------|--------|------|
| Atlas A2 | OpsRunner | **仅 A2 推理产品** |

## 6. Kernel Dependencies

| Kernel | 文件 | 用途 |
|--------|------|------|
| `rmsnormbackward` | `src/kernels/kernels/norm/rmsnormbackward/` | 梯度计算 |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 仅 Atlas A2 推理产品支持 | lim |
| 2 | 参数为空（仅 rsv），复用前向 rms_norm 参数 | 注意 |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `rms_norm` | 前向算子 |
| `rms_norm_with_stride` | stride 变体的前向（backward 无 stride 变体） |
| `layer_norm` | LayerNorm 的反向（未单独实现） |
