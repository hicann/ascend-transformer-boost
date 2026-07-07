# AGENTS.md — 训练算子模块

> **位置**: `src/ops/ops_train/` | **算子数**: 10 | **生成**: 2026-07-03

---

## 1. 模块概述


| 类别                 | 数量 | 算子                                                                                                                   |
| -------------------- | ---- | ---------------------------------------------------------------------------------------------------------------------- |
| 前向 (Forward)       | 4    | fast_soft_max, gen_attention_mask, laser_attention, strided_batch_matmul                                               |
| 反向 (Backward/Grad) | 6    | fast_soft_max_grad, laser_attention_grad, rms_norm_backward, rope_grad, pad_with_hidden_state, unpad_with_hidden_state |

---

## 2. 标准文件组织

```
src/ops/ops_train/{op_name}/
  ├── {op_name}_operation.h        ← Operation 类声明
  ├── {op_name}_operation.cpp      ← 参数校验 + Runner 分发
  ├── {op_name}_ops_runner.h       ← 原生 Ops 封装
  └── {op_name}_ops_runner.cpp     ← 原生 Ops Tiling + Kernel 启动
```

**与推理 Op 的区别**: 无 ACLNN Runner，统一 4 文件结构，参数头文件为 `include/atb/train_op_params.h`。

---

## 3. Op 完整列表


| Op                                                  | 类别 | 说明                     |
| --------------------------------------------------- | ---- | ------------------------ |
| [fast_soft_max](fast_soft_max/)                     | 前向 | 快速 Softmax 训练前向    |
| [gen_attention_mask](gen_attention_mask/)           | 前向 | 生成 Attention Mask      |
| [laser_attention](laser_attention/)                 | 前向 | Laser Attention 训练前向 |
| [strided_batch_matmul](strided_batch_matmul/)       | 前向 | Strided Batch MatMul     |
| [fast_soft_max_grad](fast_soft_max_grad/)           | 反向 | Fast Softmax 反向        |
| [laser_attention_grad](laser_attention_grad/)       | 反向 | Laser Attention 反向     |
| [rms_norm_backward](rms_norm_backward/)             | 反向 | RMS Norm 反向            |
| [rope_grad](rope_grad/)                             | 反向 | RoPE 反向                |
| [pad_with_hidden_state](pad_with_hidden_state/)     | 混合 | Padding + Hidden State   |
| [unpad_with_hidden_state](unpad_with_hidden_state/) | 混合 | Unpadding + Hidden State |

---

## 4. Agent 工作流

```
1. 读 include/atb/train_op_params.h  — 查参数结构体
2. 读 {op}_operation.cpp             — 理解参数校验
3. 读 {op}_ops_runner.cpp            — 理解 Kernel 调用
4. 读 .agent/knowledge/routing/{op}.md — 路由指引（如已生成）
```

---

## 5. 关联模块

- [推理算子](../ops_infer/AGENTS.md)
- [Kernel 模块](../../kernels/kernels/AGENTS.md)
- [知识库主索引](../../.agent/knowledge/README.md)
