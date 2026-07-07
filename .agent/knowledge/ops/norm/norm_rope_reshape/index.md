# norm_rope_reshape — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "norm_rope_reshape"
  category: "norm"
  tier: "S"
  type: "fusion"

source:
  repo_path: "src/ops/ops_infer/norm_rope_reshape/"
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
| 1 | `norm_rope_reshape_operation.h` | Operation 定义 |
| 2 | `norm_rope_reshape_operation.cpp` | CreateRunner/InferShape |
| 3 | `norm_rope_reshape_ops_runner.h` | OpsRunner |
| 4 | `norm_rope_reshape_ops_runner.cpp` | Graph + Kernel |

## 2. Reading Order

```
[1] operation.h → 融合算子接口
[2] operation.cpp → 三阶段参数校验
[3] ops_runner.cpp → Graph 构建（RMSNorm→RoPE→Reshape）
```

## 3. Parameter Constraints

```cpp
struct NormRopeReshapeParam {
    uint32_t precisionMode = 0;   // 精度模式
    uint32_t rotaryCoeff = 2;     // RoPE 旋转系数
    float epsilon = 1e-5;         // Norm epsilon
};
```

| 参数 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| `precisionMode` | uint32 | 0 | 精度模式 |
| `rotaryCoeff` | uint32 | 2 | RoPE 旋转系数 |
| `epsilon` | float | 1e-5 | > 0 |

**平台限制**: 仅 Atlas 800I A2 推理产品支持。

## 4. Computation Pipeline

```yaml
pipeline_type: multi_stage_fusion
stages:
  - stage: RMSNorm（归一化）
    note: "fp16 in → fp32 acc → fp16 out"
  - stage: RoPE（旋转位置编码）
    note: "rotaryCoeff 控制旋转系数"
  - stage: ReshapeAndCache
    note: "KV Cache 写入"
note: "三阶段融合算子，单 kernel 内完成。Golden 生成需匹配中间 dtype。"
```

## 5. Execution Paths

```
NormRopeReshapeOperation::CreateRunner()
  └── → NormRopeReshapeOpsRunner（单一路径，仅 A2）
         Kernel: laser_attention（融合 kernel）
```

| 平台 | Runner | 限制 |
|------|--------|------|
| Atlas A2 | OpsRunner | **仅 A2**，其他平台不支持 |

## 6. Kernel Dependencies

| Kernel | 文件 | 用途 |
|--------|------|------|
| `laser_attention` | `src/kernels/mixkernels/laser_attention/` | 融合 kernel（RMSNorm+RoPE+Reshape） |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 仅 Atlas A2 支持 | lim |
| 2 | 三阶段融合，Golden 生成需匹配中间 dtype | 注意 |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `rms_norm` | 融合的第一个阶段 |
| `rope` | 融合的第二个阶段（RoPE） |
| `kv_cache` | ReshapeAndCache 阶段关联 |
