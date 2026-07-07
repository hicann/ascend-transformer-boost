# laser_attention — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "laser_attention"
  category: "attention"
  tier: "S"
  type: "single"

source:
  repo_path: "src/ops/ops_train/laser_attention/"
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
| 1 | `laser_attention_operation.h` | Operation 定义 |
| 2 | `laser_attention_operation.cpp` | CreateRunner/InferShape |
| 3 | `laser_attention_ops_runner.h` | OpsRunner |
| 4 | `laser_attention_ops_runner.cpp` | Graph + Kernel |

## 2. Reading Order

```
[1] operation.h → 接口签名
[2] operation.cpp → CreateRunner + 平台检测
[3] ops_runner.cpp → LaserAttention Kernel 调用
```

## 3. Parameter Constraints

```cpp
struct LaserAttentionParam {
    int headNum = 0;                       // head 个数，> 0
    std::string inputLayout = "BNSD";      // BNSD 或 SBH
    float scaleValue = 0.08838834764831843; // 缩放系数，(0, 1]
};
```

| 参数 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| `headNum` | int | 0 | > 0 |
| `inputLayout` | string | "BNSD" | "BNSD" / "SBH" |
| `scaleValue` | float | 0.0884 | (0, 1] |

**平台限制**: 仅 Atlas 800I A2 推理产品支持。

## 4. Computation Pipeline

```yaml
pipeline_type: single_stage
note: "训练侧 Laser 自注意力，替代标准 Flash Attention。scaleValue 控制 QK^T 缩放。"
```

## 5. Execution Paths

```
LaserAttentionOperation::CreateRunner()
  └── → LaserAttentionOpsRunner（单一路径）
         Kernel: laser_attention
```

## 6. Kernel Dependencies

| Kernel | 文件 | 用途 |
|--------|------|------|
| `laser_attention` | `src/kernels/mixkernels/laser_attention/` | 自注意力计算 |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 仅 A2 推理产品支持 | lim |
| 2 | LaserAttention 是训练侧算子 | 注意 |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `self_attention` | 推理侧对应（更完整，含 ACLNN） |
| `laser_attention_grad` | 反向传播 |
| `gen_attention_mask` | 前处理（mask 生成） |
