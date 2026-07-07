# AGENTS.md — 推理算子模块

> **位置**: `src/ops/ops_infer/` | **算子数**: 71 | **生成**: 2026-07-03
>
> AI Agent 首次接触此模块时，阅读本文即可了解全貌。详细知识条目见 `.agent/knowledge/`。

---

## 1. 模块概述

`ops_infer/` 包含所有**推理**（Inference）ATB 算子，按功能分为 6 类：

| 类别 | 数量 | 典型算子 |
|------|------|---------|
| Norm (归一化) | 5 | layer_norm, rms_norm, cohere_layernorm |
| Attention (注意力) | 10 | self_attention, paged_attention, multi_latent_attention |
| Activation (激活/量化) | 7 | activation, elewise, swiglu_quant, rope, softmax |
| Communication (通信) | 11 | all_gather, all_reduce, all_to_all, broadcast |
| Linear (线性层) | 3 | linear, linear_parallel, linear_sparse |
| Other (其他) | 35 | slice, concat, gather, transpose, sort, fill |

---

## 2. 标准文件组织

```
src/ops/ops_infer/{op_name}/
  ├── {op_name}_operation.h        ← Operation 类声明
  ├── {op_name}_operation.cpp      ← 参数校验 + Runner 分发
  ├── {op_name}_aclnn_runner.h     ← ACLNN 封装（如有）
  ├── {op_name}_aclnn_runner.cpp   ← ACLNN 实现
  ├── {op_name}_ops_runner.h       ← 原生 Ops 封装
  └── {op_name}_ops_runner.cpp     ← 原生 Ops 实现
```

参数定义: `include/atb/infer_op_params.h`

---

## 3. Agent 工作流

### 理解一个 Op（推荐路径）

```
1. 读 .agent/knowledge/routing/{op_name}.md     — 文件清单 + 阅读顺序（30s）
2. 按需读 ops_infer/{op_name}/ 源码               — 按 routing 顺序阅读
3. 读 .agent/knowledge/ops/{cat}/{op}/index.md   — 完整知识条目（如已生成）
```

### 常见任务

| 任务 | 路径 | 耗时 |
|------|------|------|
| 查参数定义 | `include/atb/infer_op_params.h` | 30s |
| 查 Kernel | `../kernels/kernels/{subpath}/` | 1min |
| 查 Op 路由 | `.agent/knowledge/routing/{op_name}.md` | 30s |
| 查知识条目 | `.agent/knowledge/ops/{cat}/{op}/index.md` | 1min |
| 查所有 Op 列表 | `.agent/knowledge/README.md` | 30s |

---

## 4. 全部 Op 索引

### XL — 复合/超大型 (1)
[self_attention](self_attention/) — 27 文件，BMM1→Softmax→BMM2 Pipeline

### L — 大型 (8)
[elewise](elewise/), [linear](linear/), [linear_parallel](linear_parallel/), [mla_preprocess](mla_preprocess/), [paged_attention](paged_attention/), [reshape_and_cache](reshape_and_cache/), [ring_mla](ring_mla/), [rms_norm](rms_norm/)

### M — 中型 (24)
[activation](activation/), [as_strided](as_strided/), [concat](concat/), [cumsum](cumsum/), [fill](fill/), [fused_add_topk_div](fused_add_topk_div/), [gather](gather/), [gather_pre_rms_norm](gather_pre_rms_norm/), [gmm_deq_swiglu_quant_gmm_deq](gmm_deq_swiglu_quant_gmm_deq/), [grouped_matmul_with_routing](grouped_matmul_with_routing/), [layer_norm](layer_norm/), [mm_deq_swiglu_quant_mm_deq](mm_deq_swiglu_quant_mm_deq/), [multi_latent_attention](multi_latent_attention/), [paged_cache_load](paged_cache_load/), [reduce](reduce/), [repeat](repeat/), [rope](rope/), [slice](slice/), [softmax](softmax/), [sort](sort/), [split](split/), [swiglu_quant](swiglu_quant/), [transdata](transdata/), [transpose](transpose/)

### S — 简单 (38)
[all_gather](all_gather/), [all_gatherv](all_gatherv/), [all_reduce](all_reduce/), [all_to_all](all_to_all/), [all_to_allv](all_to_allv/), [all_to_allvv2](all_to_allvv2/), [block_copy](block_copy/), [broadcast](broadcast/), [cohere_layernorm](cohere_layernorm/), [dynamic_ntk](dynamic_ntk/), [faupdate](faupdate/), [gating](gating/), [group_topk](group_topk/), [grouped_matmul_inplace_add](grouped_matmul_inplace_add/), [index_add](index_add/), [kv_cache](kv_cache/), [layer_norm_with_stride](layer_norm_with_stride/), [linear_sparse](linear_sparse/), [multinomial](multinomial/), [nonzero](nonzero/), [norm_rope_reshape](norm_rope_reshape/), [onehot](onehot/), [pad](pad/), [razor_fusion_attention](razor_fusion_attention/), [recv](recv/), [reduce_scatter](reduce_scatter/), [reduce_scatterv](reduce_scatterv/), [relay_attention](relay_attention/), [reshape_and_cache_omni](reshape_and_cache_omni/), [reshape_and_cache_with_stride](reshape_and_cache_with_stride/), [rms_norm_with_stride](rms_norm_with_stride/), [rope_q_concat](rope_q_concat/), [scatter_elements_v2](scatter_elements_v2/), [send](send/), [set_value](set_value/), [topk_topp_sampling](topk_topp_sampling/), [unpad](unpad/), [where](where/)

---

## 5. 关联模块

- [训练算子](../ops_train/AGENTS.md)
- [Kernel 模块](../../kernels/kernels/AGENTS.md)
- [知识库主索引](../../.agent/knowledge/README.md)
