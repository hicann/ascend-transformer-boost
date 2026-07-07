# AGENTS.md — Kernel 模块

> **位置**: `src/kernels/kernels/` + `src/kernels/mixkernels/` | **生成**: 2026-07-03

---

## 1. 模块概述

| 目录 | 用途 | 子目录数 |
|------|------|---------|
| `src/kernels/kernels/` | 标准 Kernel — 按功能分类 | ~35 |
| `src/kernels/mixkernels/` | 融合 Kernel — 多算子大 Kernel | ~5 |

---

## 2. 标准 Kernel 目录

| 子目录 | 覆盖 Op | 说明 |
|--------|--------|------|
| `norm/` | layer_norm, rms_norm, cohere_layernorm | 归一化 |
| `attention/` | attention 系列 | 注意力（不含 FA） |
| `activation/` | activation 系列 | 激活函数 |
| `elewise/` | elewise | 逐元素运算 |
| `softmax/` | softmax | Softmax |
| `matmul/` | linear, strided_batch_matmul | 矩阵乘 |
| `concat/` | concat | Tensor 拼接 |
| `split/` | split | Tensor 切分 |
| `slice/` | slice | Tensor 切片 |
| `gather/` | gather | Tensor 收集 |
| `transpose/` | transpose | Tensor 转置 |
| `transdata/` | transdata | 数据格式转换 |
| `reduce/` | reduce | Tensor 规约 |
| `sort/` | sort | 排序 |
| `fill/` | fill | 填充 |
| `copy/` | block_copy | 拷贝 |
| `cumsum/` | cumsum | 累积求和 |
| `expand/` | broadcast, repeat | 扩展/广播 |
| `asstrided/` | as_strided | 步长视图 |
| `nonzero/` | nonzero | 非零索引 |
| `onehot/` | onehot | One-Hot 编码 |
| `multinomial/` | multinomial | 多项式采样 |
| `index/` | index_add | 索引操作 |
| `logprobs_sample/` | topk_topp_sampling | 采样 |
| `dynamic_ntk/` | dynamic_ntk | 动态 NTK |
| `faUpdate/` | faupdate | FA 更新 |
| `group_topk/` | group_topk | 分组 TopK |
| `moe_gate_corr/` | gating | MoE Gate |
| `scatter_elements_v2/` | scatter_elements_v2 | 散射更新 |
| `reverse/`, `zeroslike/` | — | 辅助操作 |

### 标准 Kernel 文件组织

```
src/kernels/kernels/{subpath}/
  ├── {name}_kernel.h / .cpp      ← Kernel 主实现
  ├── {name}_tiling.h / .cpp      ← Tiling 计算
  └── CMakeLists.txt              ← 编译配置
```

---

## 3. 融合 Kernel (`mixkernels/`)

| 子目录 | 覆盖 Op |
|--------|--------|
| `laser_attention/` | self_attention, paged_attention, multi_latent_attention, relay_attention, swiglu_quant, linear, rms_norm |
| `razor_fusion_attention/` | razor_fusion_attention |

融合 Kernel 特点: 单 Kernel 完成多阶段计算（如 BMM1→Softmax→BMM2），含 SoC 特定代码。

---

## 4. Agent 工作流

### 分析一个 Kernel

```
1. 在上面表格中搜索 Op 名 → 找到 Kernel 子目录
2. 读 {name}_tiling.h    → Tiling 策略
3. 读 {name}_kernel.cpp  → Kernel 启动参数 + workspaceSize
4. 读 {name}_compute.h   → 计算逻辑（dtype 转换在这里）
```

### 关键概念

| 术语 | 含义 | 查找位置 |
|------|------|---------|
| **Tiling** | 大 Tensor 切分为 UB 适配小块 | `*_tiling.h` |
| **Workspace** | 额外 HBM 内存需求 | `*_kernel.cpp` → workspaceSize |
| **UB** | 片上缓存 192KB | Tiling 约束 |
| **PrecType** | Kernel 模板精度参数 | `*_common.h` |

---

## 5. 关联模块

- [推理算子](../../ops/ops_infer/AGENTS.md)
- [训练算子](../../ops/ops_train/AGENTS.md)
- [知识库主索引](../../.agent/knowledge/README.md)
