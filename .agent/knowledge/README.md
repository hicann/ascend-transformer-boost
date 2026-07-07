# ATB Agent 知识库 — 主索引

> **自动生成** | 基于 `inventory.csv` | 覆盖 82 个 ATB 算子
>
> **使用方式**: Agent 先查此索引定位目标 Op，再读 `routing/{op_name}.md` 了解文件清单，
> 最后读 `ops/{category}/{op_name}/index.md` 获取完整知识条目。

---

## 1. 总览

- **算子总数**: 82
  - 推理 (infer): 71
  - 训练 (train): 10
  - 公共 (common): 1
- **复杂度分布**: XL=1, L=8, M=24, S=49
- **ACLNN 支持**: 25/82
- **Dtype 断点**: 38/82

## 2. 分类索引

### 2.1 按功能分类

#### norm（归一化算子）— 8 个

| 算子 | 分类 | Tier | 文件数 | ACLNN | Runner 类型 |
|------|------|------|--------|-------|-----------|
| [cohere_layernorm](routing/cohere_layernorm.md) | infer | S | 4 | no | OpsRunner,Operation |
| [gather_pre_rms_norm](routing/gather_pre_rms_norm.md) | infer | M | 4 | no | OpsRunner,Operation |
| [layer_norm](routing/layer_norm.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [layer_norm_with_stride](routing/layer_norm_with_stride.md) | infer | S | 4 | no | OpsRunner,Operation |
| [norm_rope_reshape](routing/norm_rope_reshape.md) | infer | S | 4 | no | OpsRunner,Operation |
| [rms_norm](routing/rms_norm.md) | infer | L | 10 | yes | OpsRunner,ACLNNRunner,Operation |
| [rms_norm_backward](routing/rms_norm_backward.md) | train | S | 4 | no | OpsRunner,Operation |
| [rms_norm_with_stride](routing/rms_norm_with_stride.md) | infer | S | 4 | no | OpsRunner,Operation |

#### attention（注意力算子）— 11 个

| 算子 | 分类 | Tier | 文件数 | ACLNN | Runner 类型 |
|------|------|------|--------|-------|-----------|
| [faupdate](routing/faupdate.md) | infer | S | 4 | no | OpsRunner,Operation |
| [gen_attention_mask](routing/gen_attention_mask.md) | train | S | 4 | no | OpsRunner,Operation |
| [laser_attention](routing/laser_attention.md) | train | S | 4 | no | OpsRunner,Operation |
| [laser_attention_grad](routing/laser_attention_grad.md) | train | S | 4 | no | OpsRunner,Operation |
| [mla_preprocess](routing/mla_preprocess.md) | infer | L | 9 | yes | OpsRunner,ACLNNRunner,Operation |
| [multi_latent_attention](routing/multi_latent_attention.md) | infer | M | 9 | no | OpsRunner,Operation |
| [paged_attention](routing/paged_attention.md) | infer | L | 12 | yes | OpsRunner,ACLNNRunner,Operation |
| [razor_fusion_attention](routing/razor_fusion_attention.md) | infer | S | 4 | no | OpsRunner,Operation |
| [relay_attention](routing/relay_attention.md) | infer | S | 6 | no | OpsRunner,Operation |
| [ring_mla](routing/ring_mla.md) | infer | L | 7 | no | OpsRunner,Operation |
| [self_attention](routing/self_attention.md) | infer | XL | 27 | yes | OpsRunner,ACLNNRunner,Operation |

#### activation（激活/量化算子）— 12 个

| 算子 | 分类 | Tier | 文件数 | ACLNN | Runner 类型 |
|------|------|------|--------|-------|-----------|
| [activation](routing/activation.md) | infer | M | 10 | yes | OpsRunner,ACLNNRunner,Operation |
| [elewise](routing/elewise.md) | infer | L | 10 | yes | OpsRunner,ACLNNRunner,Operation |
| [fast_soft_max](routing/fast_soft_max.md) | train | S | 4 | no | OpsRunner,Operation |
| [fast_soft_max_grad](routing/fast_soft_max_grad.md) | train | S | 4 | no | OpsRunner,Operation |
| [gating](routing/gating.md) | infer | S | 4 | no | OpsRunner,Operation |
| [gmm_deq_swiglu_quant_gmm_deq](routing/gmm_deq_swiglu_quant_gmm_deq.md) | infer | M | 4 | no | OpsRunner,Operation |
| [mm_deq_swiglu_quant_mm_deq](routing/mm_deq_swiglu_quant_mm_deq.md) | infer | M | 4 | no | OpsRunner,Operation |
| [rope](routing/rope.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [rope_grad](routing/rope_grad.md) | train | S | 4 | no | OpsRunner,Operation |
| [rope_q_concat](routing/rope_q_concat.md) | infer | S | 4 | no | OpsRunner,Operation |
| [softmax](routing/softmax.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [swiglu_quant](routing/swiglu_quant.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |

#### communication（集合通信算子）— 11 个

| 算子 | 分类 | Tier | 文件数 | ACLNN | Runner 类型 |
|------|------|------|--------|-------|-----------|
| [all_gather](routing/all_gather.md) | infer | S | 6 | no | OpsRunner,Operation |
| [all_gatherv](routing/all_gatherv.md) | infer | S | 4 | no | OpsRunner,Operation |
| [all_reduce](routing/all_reduce.md) | infer | S | 6 | no | OpsRunner,Operation |
| [all_to_all](routing/all_to_all.md) | infer | S | 6 | no | OpsRunner,Operation |
| [all_to_allv](routing/all_to_allv.md) | infer | S | 4 | no | OpsRunner,Operation |
| [all_to_allvv2](routing/all_to_allvv2.md) | infer | S | 4 | no | OpsRunner,Operation |
| [broadcast](routing/broadcast.md) | infer | S | 6 | no | OpsRunner,Operation |
| [recv](routing/recv.md) | infer | S | 4 | no | OpsRunner,Operation |
| [reduce_scatter](routing/reduce_scatter.md) | infer | S | 6 | no | OpsRunner,Operation |
| [reduce_scatterv](routing/reduce_scatterv.md) | infer | S | 4 | no | OpsRunner,Operation |
| [send](routing/send.md) | infer | S | 4 | no | OpsRunner,Operation |

#### other（其他算子）— 40 个

| 算子 | 分类 | Tier | 文件数 | ACLNN | Runner 类型 |
|------|------|------|--------|-------|-----------|
| [as_strided](routing/as_strided.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [block_copy](routing/block_copy.md) | infer | S | 4 | no | OpsRunner,Operation |
| [concat](routing/concat.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [cumsum](routing/cumsum.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [dynamic_ntk](routing/dynamic_ntk.md) | infer | S | 4 | no | OpsRunner,Operation |
| [event_operation](routing/event_operation.md) | common | S | 4 | no |  |
| [fill](routing/fill.md) | infer | M | 8 | yes | OpsRunner,ACLNNRunner |
| [fused_add_topk_div](routing/fused_add_topk_div.md) | infer | M | 5 | no | OpsRunner,Operation |
| [gather](routing/gather.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [group_topk](routing/group_topk.md) | infer | S | 4 | no | OpsRunner,Operation |
| [grouped_matmul_inplace_add](routing/grouped_matmul_inplace_add.md) | infer | S | 4 | no | OpsRunner,Operation |
| [grouped_matmul_with_routing](routing/grouped_matmul_with_routing.md) | infer | M | 4 | no | OpsRunner,Operation |
| [index_add](routing/index_add.md) | infer | S | 4 | no | OpsRunner,Operation |
| [kv_cache](routing/kv_cache.md) | infer | S | 4 | no | OpsRunner,Operation |
| [linear](routing/linear.md) | infer | L | 10 | yes | OpsRunner,ACLNNRunner,Operation |
| [linear_parallel](routing/linear_parallel.md) | infer | L | 8 | yes | OpsRunner,ACLNNRunner,Operation,Kernel |
| [linear_sparse](routing/linear_sparse.md) | infer | S | 4 | no | OpsRunner,Operation |
| [multinomial](routing/multinomial.md) | infer | S | 4 | no | OpsRunner,Operation |
| [nonzero](routing/nonzero.md) | infer | S | 4 | no | OpsRunner,Operation |
| [onehot](routing/onehot.md) | infer | S | 4 | no | OpsRunner,Operation |
| [pad](routing/pad.md) | infer | S | 4 | no | OpsRunner,Operation |
| [pad_with_hidden_state](routing/pad_with_hidden_state.md) | train | S | 4 | no | OpsRunner,Operation |
| [paged_cache_load](routing/paged_cache_load.md) | infer | M | 5 | no | OpsRunner,Operation |
| [reduce](routing/reduce.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [repeat](routing/repeat.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [reshape_and_cache](routing/reshape_and_cache.md) | infer | L | 14 | yes | OpsRunner,ACLNNRunner,Operation |
| [reshape_and_cache_omni](routing/reshape_and_cache_omni.md) | infer | S | 4 | no | OpsRunner |
| [reshape_and_cache_with_stride](routing/reshape_and_cache_with_stride.md) | infer | S | 6 | no | OpsRunner,Operation |
| [scatter_elements_v2](routing/scatter_elements_v2.md) | infer | S | 4 | no | OpsRunner,Operation |
| [set_value](routing/set_value.md) | infer | S | 4 | no | OpsRunner,Operation |
| [slice](routing/slice.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [sort](routing/sort.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner |
| [split](routing/split.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [strided_batch_matmul](routing/strided_batch_matmul.md) | train | S | 4 | no | OpsRunner |
| [topk_topp_sampling](routing/topk_topp_sampling.md) | infer | S | 4 | no | OpsRunner |
| [transdata](routing/transdata.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [transpose](routing/transpose.md) | infer | M | 6 | yes | OpsRunner,ACLNNRunner,Operation |
| [unpad](routing/unpad.md) | infer | S | 4 | no | OpsRunner,Operation |
| [unpad_with_hidden_state](routing/unpad_with_hidden_state.md) | train | S | 4 | no | OpsRunner,Operation |
| [where](routing/where.md) | infer | S | 4 | no | OpsRunner,Operation |

### 2.2 按复杂度分层

#### XL — 1 个 — 复合算子（多 Op 组合，>18 文件或多 SoC variant）

| 算子 | 分类 | 文件数 | ACLNN | Dtype断点 |
|------|------|--------|-------|----------|
| [self_attention](routing/self_attention.md) | infer | 27 | yes | ⚠️ |

#### L — 8 个 — 大型算子（10-28 文件，多 Runner 路径，多 SoC variant）

| 算子 | 分类 | 文件数 | ACLNN | Dtype断点 |
|------|------|--------|-------|----------|
| [elewise](routing/elewise.md) | infer | 10 | yes | ⚠️ |
| [linear](routing/linear.md) | infer | 10 | yes | ⚠️ |
| [linear_parallel](routing/linear_parallel.md) | infer | 8 | yes | ⚠️ |
| [mla_preprocess](routing/mla_preprocess.md) | infer | 9 | yes | ⚠️ |
| [paged_attention](routing/paged_attention.md) | infer | 12 | yes | ⚠️ |
| [reshape_and_cache](routing/reshape_and_cache.md) | infer | 14 | yes | ⚠️ |
| [ring_mla](routing/ring_mla.md) | infer | 7 | no | ⚠️ |
| [rms_norm](routing/rms_norm.md) | infer | 10 | yes | ⚠️ |

#### M — 24 个 — 中型算子（6-15 文件，多 variant 或 ACLNN 路径）

| 算子 | 分类 | 文件数 | ACLNN | Dtype断点 |
|------|------|--------|-------|----------|
| [activation](routing/activation.md) | infer | 10 | yes |  |
| [as_strided](routing/as_strided.md) | infer | 6 | yes | ⚠️ |
| [concat](routing/concat.md) | infer | 6 | yes | ⚠️ |
| [cumsum](routing/cumsum.md) | infer | 6 | yes |  |
| [fill](routing/fill.md) | infer | 8 | yes | ⚠️ |
| [fused_add_topk_div](routing/fused_add_topk_div.md) | infer | 5 | no | ⚠️ |
| [gather](routing/gather.md) | infer | 6 | yes | ⚠️ |
| [gather_pre_rms_norm](routing/gather_pre_rms_norm.md) | infer | 4 | no | ⚠️ |
| [gmm_deq_swiglu_quant_gmm_deq](routing/gmm_deq_swiglu_quant_gmm_deq.md) | infer | 4 | no | ⚠️ |
| [grouped_matmul_with_routing](routing/grouped_matmul_with_routing.md) | infer | 4 | no | ⚠️ |
| [layer_norm](routing/layer_norm.md) | infer | 6 | yes | ⚠️ |
| [mm_deq_swiglu_quant_mm_deq](routing/mm_deq_swiglu_quant_mm_deq.md) | infer | 4 | no | ⚠️ |
| [multi_latent_attention](routing/multi_latent_attention.md) | infer | 9 | no | ⚠️ |
| [paged_cache_load](routing/paged_cache_load.md) | infer | 5 | no | ⚠️ |
| [reduce](routing/reduce.md) | infer | 6 | yes |  |
| [repeat](routing/repeat.md) | infer | 6 | yes | ⚠️ |
| [rope](routing/rope.md) | infer | 6 | yes |  |
| [slice](routing/slice.md) | infer | 6 | yes |  |
| [softmax](routing/softmax.md) | infer | 6 | yes |  |
| [sort](routing/sort.md) | infer | 6 | yes | ⚠️ |
| [split](routing/split.md) | infer | 6 | yes |  |
| [swiglu_quant](routing/swiglu_quant.md) | infer | 6 | yes | ⚠️ |
| [transdata](routing/transdata.md) | infer | 6 | yes | ⚠️ |
| [transpose](routing/transpose.md) | infer | 6 | yes |  |

#### S — 49 个 — 简单算子（4-8 文件，单一路径）

| 算子 | 分类 | 文件数 | ACLNN | Dtype断点 |
|------|------|--------|-------|----------|
| [event_operation](routing/event_operation.md) | common | 4 | no |  |
| [all_gather](routing/all_gather.md) | infer | 6 | no |  |
| [all_gatherv](routing/all_gatherv.md) | infer | 4 | no |  |
| [all_reduce](routing/all_reduce.md) | infer | 6 | no | ⚠️ |
| [all_to_all](routing/all_to_all.md) | infer | 6 | no |  |
| [all_to_allv](routing/all_to_allv.md) | infer | 4 | no |  |
| [all_to_allvv2](routing/all_to_allvv2.md) | infer | 4 | no |  |
| [block_copy](routing/block_copy.md) | infer | 4 | no | ⚠️ |
| [broadcast](routing/broadcast.md) | infer | 6 | no |  |
| [cohere_layernorm](routing/cohere_layernorm.md) | infer | 4 | no |  |
| [dynamic_ntk](routing/dynamic_ntk.md) | infer | 4 | no | ⚠️ |
| [faupdate](routing/faupdate.md) | infer | 4 | no |  |
| [gating](routing/gating.md) | infer | 4 | no |  |
| [group_topk](routing/group_topk.md) | infer | 4 | no |  |
| [grouped_matmul_inplace_add](routing/grouped_matmul_inplace_add.md) | infer | 4 | no |  |
| [index_add](routing/index_add.md) | infer | 4 | no | ⚠️ |
| [kv_cache](routing/kv_cache.md) | infer | 4 | no | ⚠️ |
| [layer_norm_with_stride](routing/layer_norm_with_stride.md) | infer | 4 | no |  |
| [linear_sparse](routing/linear_sparse.md) | infer | 4 | no |  |
| [multinomial](routing/multinomial.md) | infer | 4 | no | ⚠️ |
| [nonzero](routing/nonzero.md) | infer | 4 | no |  |
| [norm_rope_reshape](routing/norm_rope_reshape.md) | infer | 4 | no |  |
| [onehot](routing/onehot.md) | infer | 4 | no |  |
| [pad](routing/pad.md) | infer | 4 | no |  |
| [razor_fusion_attention](routing/razor_fusion_attention.md) | infer | 4 | no |  |
| [recv](routing/recv.md) | infer | 4 | no |  |
| [reduce_scatter](routing/reduce_scatter.md) | infer | 6 | no | ⚠️ |
| [reduce_scatterv](routing/reduce_scatterv.md) | infer | 4 | no |  |
| [relay_attention](routing/relay_attention.md) | infer | 6 | no |  |
| [reshape_and_cache_omni](routing/reshape_and_cache_omni.md) | infer | 4 | no |  |
| [reshape_and_cache_with_stride](routing/reshape_and_cache_with_stride.md) | infer | 6 | no |  |
| [rms_norm_with_stride](routing/rms_norm_with_stride.md) | infer | 4 | no |  |
| [rope_q_concat](routing/rope_q_concat.md) | infer | 4 | no |  |
| [scatter_elements_v2](routing/scatter_elements_v2.md) | infer | 4 | no |  |
| [send](routing/send.md) | infer | 4 | no |  |
| [set_value](routing/set_value.md) | infer | 4 | no |  |
| [topk_topp_sampling](routing/topk_topp_sampling.md) | infer | 4 | no | ⚠️ |
| [unpad](routing/unpad.md) | infer | 4 | no | ⚠️ |
| [where](routing/where.md) | infer | 4 | no | ⚠️ |
| [fast_soft_max](routing/fast_soft_max.md) | train | 4 | no |  |
| [fast_soft_max_grad](routing/fast_soft_max_grad.md) | train | 4 | no |  |
| [gen_attention_mask](routing/gen_attention_mask.md) | train | 4 | no | ⚠️ |
| [laser_attention](routing/laser_attention.md) | train | 4 | no |  |
| [laser_attention_grad](routing/laser_attention_grad.md) | train | 4 | no |  |
| [pad_with_hidden_state](routing/pad_with_hidden_state.md) | train | 4 | no |  |
| [rms_norm_backward](routing/rms_norm_backward.md) | train | 4 | no | ⚠️ |
| [rope_grad](routing/rope_grad.md) | train | 4 | no |  |
| [strided_batch_matmul](routing/strided_batch_matmul.md) | train | 4 | no | ⚠️ |
| [unpad_with_hidden_state](routing/unpad_with_hidden_state.md) | train | 4 | no |  |

### 2.3 按 ACLNN 支持

**有 ACLNN 路径**（25 个）— 走 CANN 标准 API:

- [activation](routing/activation.md) (infer, M, 10f)
- [as_strided](routing/as_strided.md) (infer, M, 6f)
- [concat](routing/concat.md) (infer, M, 6f)
- [cumsum](routing/cumsum.md) (infer, M, 6f)
- [elewise](routing/elewise.md) (infer, L, 10f)
- [fill](routing/fill.md) (infer, M, 8f)
- [gather](routing/gather.md) (infer, M, 6f)
- [layer_norm](routing/layer_norm.md) (infer, M, 6f)
- [linear](routing/linear.md) (infer, L, 10f)
- [linear_parallel](routing/linear_parallel.md) (infer, L, 8f)
- [mla_preprocess](routing/mla_preprocess.md) (infer, L, 9f)
- [paged_attention](routing/paged_attention.md) (infer, L, 12f)
- [reduce](routing/reduce.md) (infer, M, 6f)
- [repeat](routing/repeat.md) (infer, M, 6f)
- [reshape_and_cache](routing/reshape_and_cache.md) (infer, L, 14f)
- [rms_norm](routing/rms_norm.md) (infer, L, 10f)
- [rope](routing/rope.md) (infer, M, 6f)
- [self_attention](routing/self_attention.md) (infer, XL, 27f)
- [slice](routing/slice.md) (infer, M, 6f)
- [softmax](routing/softmax.md) (infer, M, 6f)
- [sort](routing/sort.md) (infer, M, 6f)
- [split](routing/split.md) (infer, M, 6f)
- [swiglu_quant](routing/swiglu_quant.md) (infer, M, 6f)
- [transdata](routing/transdata.md) (infer, M, 6f)
- [transpose](routing/transpose.md) (infer, M, 6f)

**纯 Ops 路径**（57 个）— 走原生 Ops API:

- [all_gather](routing/all_gather.md) (infer, S, 6f)
- [all_gatherv](routing/all_gatherv.md) (infer, S, 4f)
- [all_reduce](routing/all_reduce.md) (infer, S, 6f)
- [all_to_all](routing/all_to_all.md) (infer, S, 6f)
- [all_to_allv](routing/all_to_allv.md) (infer, S, 4f)
- [all_to_allvv2](routing/all_to_allvv2.md) (infer, S, 4f)
- [block_copy](routing/block_copy.md) (infer, S, 4f)
- [broadcast](routing/broadcast.md) (infer, S, 6f)
- [cohere_layernorm](routing/cohere_layernorm.md) (infer, S, 4f)
- [dynamic_ntk](routing/dynamic_ntk.md) (infer, S, 4f)
- [event_operation](routing/event_operation.md) (common, S, 4f)
- [fast_soft_max](routing/fast_soft_max.md) (train, S, 4f)
- [fast_soft_max_grad](routing/fast_soft_max_grad.md) (train, S, 4f)
- [faupdate](routing/faupdate.md) (infer, S, 4f)
- [fused_add_topk_div](routing/fused_add_topk_div.md) (infer, M, 5f)
- [gather_pre_rms_norm](routing/gather_pre_rms_norm.md) (infer, M, 4f)
- [gating](routing/gating.md) (infer, S, 4f)
- [gen_attention_mask](routing/gen_attention_mask.md) (train, S, 4f)
- [gmm_deq_swiglu_quant_gmm_deq](routing/gmm_deq_swiglu_quant_gmm_deq.md) (infer, M, 4f)
- [group_topk](routing/group_topk.md) (infer, S, 4f)
- [grouped_matmul_inplace_add](routing/grouped_matmul_inplace_add.md) (infer, S, 4f)
- [grouped_matmul_with_routing](routing/grouped_matmul_with_routing.md) (infer, M, 4f)
- [index_add](routing/index_add.md) (infer, S, 4f)
- [kv_cache](routing/kv_cache.md) (infer, S, 4f)
- [laser_attention](routing/laser_attention.md) (train, S, 4f)
- [laser_attention_grad](routing/laser_attention_grad.md) (train, S, 4f)
- [layer_norm_with_stride](routing/layer_norm_with_stride.md) (infer, S, 4f)
- [linear_sparse](routing/linear_sparse.md) (infer, S, 4f)
- [mm_deq_swiglu_quant_mm_deq](routing/mm_deq_swiglu_quant_mm_deq.md) (infer, M, 4f)
- [multi_latent_attention](routing/multi_latent_attention.md) (infer, M, 9f)
- [multinomial](routing/multinomial.md) (infer, S, 4f)
- [nonzero](routing/nonzero.md) (infer, S, 4f)
- [norm_rope_reshape](routing/norm_rope_reshape.md) (infer, S, 4f)
- [onehot](routing/onehot.md) (infer, S, 4f)
- [pad](routing/pad.md) (infer, S, 4f)
- [pad_with_hidden_state](routing/pad_with_hidden_state.md) (train, S, 4f)
- [paged_cache_load](routing/paged_cache_load.md) (infer, M, 5f)
- [razor_fusion_attention](routing/razor_fusion_attention.md) (infer, S, 4f)
- [recv](routing/recv.md) (infer, S, 4f)
- [reduce_scatter](routing/reduce_scatter.md) (infer, S, 6f)
- [reduce_scatterv](routing/reduce_scatterv.md) (infer, S, 4f)
- [relay_attention](routing/relay_attention.md) (infer, S, 6f)
- [reshape_and_cache_omni](routing/reshape_and_cache_omni.md) (infer, S, 4f)
- [reshape_and_cache_with_stride](routing/reshape_and_cache_with_stride.md) (infer, S, 6f)
- [ring_mla](routing/ring_mla.md) (infer, L, 7f)
- [rms_norm_backward](routing/rms_norm_backward.md) (train, S, 4f)
- [rms_norm_with_stride](routing/rms_norm_with_stride.md) (infer, S, 4f)
- [rope_grad](routing/rope_grad.md) (train, S, 4f)
- [rope_q_concat](routing/rope_q_concat.md) (infer, S, 4f)
- [scatter_elements_v2](routing/scatter_elements_v2.md) (infer, S, 4f)
- [send](routing/send.md) (infer, S, 4f)
- [set_value](routing/set_value.md) (infer, S, 4f)
- [strided_batch_matmul](routing/strided_batch_matmul.md) (train, S, 4f)
- [topk_topp_sampling](routing/topk_topp_sampling.md) (infer, S, 4f)
- [unpad](routing/unpad.md) (infer, S, 4f)
- [unpad_with_hidden_state](routing/unpad_with_hidden_state.md) (train, S, 4f)
- [where](routing/where.md) (infer, S, 4f)

## 3. 文件定位模式

Agent 可通过以下模式快速定位任意 Op 的源码文件，无需遍历搜索：

### 3.1 标准目录结构

```
src/ops/ops_{category}/{op_name}/
  ├── {op_name}_operation.h        # Operation 类声明
  ├── {op_name}_operation.cpp      # 参数校验、InferShape、CreateRunner
  ├── {op_name}_aclnn_runner.h     # ACLNN 封装（如有）
  ├── {op_name}_aclnn_runner.cpp   # ACLNN 实现（如有）
  ├── {op_name}_ops_runner.h       # 原生 Ops 封装
  └── {op_name}_ops_runner.cpp     # 原生 Ops 实现
```

### 3.2 参数头文件定位

| 算子类别 | 参数头文件 |
|---------|-----------|
| 推理算子 (infer) | `include/atb/infer_op_params.h` |
| 训练算子 (train) | `include/atb/train_op_params.h` |

### 3.3 Kernel 路径定位

| Kernel 类型 | 路径模式 |
|------------|---------|
| 标准 Kernel | `src/kernels/kernels/{subcategory}/{opname}/` |
| 融合 Kernel | `src/kernels/mixkernels/{op_name}/` |
| Laser Attention | `src/kernels/mixkernels/laser_attention/` |
| TBE 适配 | `src/kernels/tbe_adapter/` |

### 3.4 测试用例定位

| 类型 | 路径模式 |
|------|---------|
| 推理算子测试 | `atk_test/atk_cida_atb/atb/infer/{OpName}Operation/` |
| 训练算子测试 | `atk_test/atk_cida_atb/atb/train/{OpName}Operation/` |
| 公共 API | `atk_test/atk_cida_atb/atb/common/atb_base_api.py` |
| Golden 实现 | `atk_test/atk_cida_atb/atb/common/data_generation.py` |

## 4. Agent 使用指南

### 4.1 快速定位流程

```
1. 在本文档搜索 Op 名 → 获取 tier 和 category
2. 读 routing/{op_name}.md → 获取文件清单和阅读顺序
3. 按需读 ops/{category}/{op_name}/index.md → 获取完整知识
```

### 4.2 常见任务与最小阅读路径

| 任务 | 最少需读文件 | 预估耗时 |
|------|-------------|---------|
| 查看 Op 参数约束 | routing/{op}.md → index.md §3 | 1 min |
| 生成 ATK Golden | routing/{op}.md → index.md §4 Pipeline | 2-3 min |
| 定位 Kernel 源码 | routing/{op}.md §1 文件清单 | 30 sec |
| 排查执行错误 | index.md §7 Known Issues | 1 min |
| 理解 Runner 选择 | index.md §5 Execution Paths | 2 min |
| 完整理解新 Op | routing → index.md（全部 9 章）| 10-20 min |

## 5. 知识条目完成状态

- **已创建知识条目**: 9/82 (10%)
- **路由文件**: 82/82 (100%)
- **模板初稿**: 2 份（layer_norm, swiglu_quant）

### 5.1 按分类统计

| 分类 | Operator 总数 | 有路由 | 有知识条目 |
|------|-------------|--------|----------|
| norm | 8 | 8 | 1 |
| attention | 11 | 11 | 6 |
| activation | 12 | 12 | 1 |
| communication | 11 | 11 | 0 |
| other | 40 | 40 | 0 |
| **合计** | **82** | **82** | **9** |

## 6. 全部 Op 清单（字母序）

| # | Op | 分类 | Tier | 文件数 | ACLNN | Dtype断点 | 路由 | 知识条目 |
|---|-----|------|------|--------|-------|----------|------|---------|
| 1 | activation | infer | M | 10 | yes |  | [📄](routing/activation.md) | ⏳ |
| 2 | all_gather | infer | S | 6 | no |  | [📄](routing/all_gather.md) | ⏳ |
| 3 | all_gatherv | infer | S | 4 | no |  | [📄](routing/all_gatherv.md) | ⏳ |
| 4 | all_reduce | infer | S | 6 | no | ⚠️ | [📄](routing/all_reduce.md) | ⏳ |
| 5 | all_to_all | infer | S | 6 | no |  | [📄](routing/all_to_all.md) | ⏳ |
| 6 | all_to_allv | infer | S | 4 | no |  | [📄](routing/all_to_allv.md) | ⏳ |
| 7 | all_to_allvv2 | infer | S | 4 | no |  | [📄](routing/all_to_allvv2.md) | ⏳ |
| 8 | as_strided | infer | M | 6 | yes | ⚠️ | [📄](routing/as_strided.md) | ⏳ |
| 9 | block_copy | infer | S | 4 | no | ⚠️ | [📄](routing/block_copy.md) | ⏳ |
| 10 | broadcast | infer | S | 6 | no |  | [📄](routing/broadcast.md) | ⏳ |
| 11 | cohere_layernorm | infer | S | 4 | no |  | [📄](routing/cohere_layernorm.md) | ⏳ |
| 12 | concat | infer | M | 6 | yes | ⚠️ | [📄](routing/concat.md) | ⏳ |
| 13 | cumsum | infer | M | 6 | yes |  | [📄](routing/cumsum.md) | ⏳ |
| 14 | dynamic_ntk | infer | S | 4 | no | ⚠️ | [📄](routing/dynamic_ntk.md) | ⏳ |
| 15 | elewise | infer | L | 10 | yes | ⚠️ | [📄](routing/elewise.md) | ⏳ |
| 16 | event_operation | common | S | 4 | no |  | [📄](routing/event_operation.md) | ⏳ |
| 17 | fast_soft_max | train | S | 4 | no |  | [📄](routing/fast_soft_max.md) | ⏳ |
| 18 | fast_soft_max_grad | train | S | 4 | no |  | [📄](routing/fast_soft_max_grad.md) | ⏳ |
| 19 | faupdate | infer | S | 4 | no |  | [📄](routing/faupdate.md) | [✅](ops/attention/faupdate/index.md) |
| 20 | fill | infer | M | 8 | yes | ⚠️ | [📄](routing/fill.md) | ⏳ |
| 21 | fused_add_topk_div | infer | M | 5 | no | ⚠️ | [📄](routing/fused_add_topk_div.md) | ⏳ |
| 22 | gather | infer | M | 6 | yes | ⚠️ | [📄](routing/gather.md) | ⏳ |
| 23 | gather_pre_rms_norm | infer | M | 4 | no | ⚠️ | [📄](routing/gather_pre_rms_norm.md) | ⏳ |
| 24 | gating | infer | S | 4 | no |  | [📄](routing/gating.md) | ⏳ |
| 25 | gen_attention_mask | train | S | 4 | no | ⚠️ | [📄](routing/gen_attention_mask.md) | ⏳ |
| 26 | gmm_deq_swiglu_quant_gmm_deq | infer | M | 4 | no | ⚠️ | [📄](routing/gmm_deq_swiglu_quant_gmm_deq.md) | ⏳ |
| 27 | group_topk | infer | S | 4 | no |  | [📄](routing/group_topk.md) | ⏳ |
| 28 | grouped_matmul_inplace_add | infer | S | 4 | no |  | [📄](routing/grouped_matmul_inplace_add.md) | ⏳ |
| 29 | grouped_matmul_with_routing | infer | M | 4 | no | ⚠️ | [📄](routing/grouped_matmul_with_routing.md) | ⏳ |
| 30 | index_add | infer | S | 4 | no | ⚠️ | [📄](routing/index_add.md) | ⏳ |
| 31 | kv_cache | infer | S | 4 | no | ⚠️ | [📄](routing/kv_cache.md) | ⏳ |
| 32 | laser_attention | train | S | 4 | no |  | [📄](routing/laser_attention.md) | ⏳ |
| 33 | laser_attention_grad | train | S | 4 | no |  | [📄](routing/laser_attention_grad.md) | ⏳ |
| 34 | layer_norm | infer | M | 6 | yes | ⚠️ | [📄](routing/layer_norm.md) | [✅](ops/norm/layer_norm/index.md) |
| 35 | layer_norm_with_stride | infer | S | 4 | no |  | [📄](routing/layer_norm_with_stride.md) | ⏳ |
| 36 | linear | infer | L | 10 | yes | ⚠️ | [📄](routing/linear.md) | ⏳ |
| 37 | linear_parallel | infer | L | 8 | yes | ⚠️ | [📄](routing/linear_parallel.md) | ⏳ |
| 38 | linear_sparse | infer | S | 4 | no |  | [📄](routing/linear_sparse.md) | ⏳ |
| 39 | mla_preprocess | infer | L | 9 | yes | ⚠️ | [📄](routing/mla_preprocess.md) | ⏳ |
| 40 | mm_deq_swiglu_quant_mm_deq | infer | M | 4 | no | ⚠️ | [📄](routing/mm_deq_swiglu_quant_mm_deq.md) | ⏳ |
| 41 | multi_latent_attention | infer | M | 9 | no | ⚠️ | [📄](routing/multi_latent_attention.md) | [✅](ops/attention/multi_latent_attention/index.md) |
| 42 | multinomial | infer | S | 4 | no | ⚠️ | [📄](routing/multinomial.md) | ⏳ |
| 43 | nonzero | infer | S | 4 | no |  | [📄](routing/nonzero.md) | ⏳ |
| 44 | norm_rope_reshape | infer | S | 4 | no |  | [📄](routing/norm_rope_reshape.md) | ⏳ |
| 45 | onehot | infer | S | 4 | no |  | [📄](routing/onehot.md) | ⏳ |
| 46 | pad | infer | S | 4 | no |  | [📄](routing/pad.md) | ⏳ |
| 47 | pad_with_hidden_state | train | S | 4 | no |  | [📄](routing/pad_with_hidden_state.md) | ⏳ |
| 48 | paged_attention | infer | L | 12 | yes | ⚠️ | [📄](routing/paged_attention.md) | [✅](ops/attention/paged_attention/index.md) |
| 49 | paged_cache_load | infer | M | 5 | no | ⚠️ | [📄](routing/paged_cache_load.md) | ⏳ |
| 50 | razor_fusion_attention | infer | S | 4 | no |  | [📄](routing/razor_fusion_attention.md) | [✅](ops/attention/razor_fusion_attention/index.md) |
| 51 | recv | infer | S | 4 | no |  | [📄](routing/recv.md) | ⏳ |
| 52 | reduce | infer | M | 6 | yes |  | [📄](routing/reduce.md) | ⏳ |
| 53 | reduce_scatter | infer | S | 6 | no | ⚠️ | [📄](routing/reduce_scatter.md) | ⏳ |
| 54 | reduce_scatterv | infer | S | 4 | no |  | [📄](routing/reduce_scatterv.md) | ⏳ |
| 55 | relay_attention | infer | S | 6 | no |  | [📄](routing/relay_attention.md) | [✅](ops/attention/relay_attention/index.md) |
| 56 | repeat | infer | M | 6 | yes | ⚠️ | [📄](routing/repeat.md) | ⏳ |
| 57 | reshape_and_cache | infer | L | 14 | yes | ⚠️ | [📄](routing/reshape_and_cache.md) | ⏳ |
| 58 | reshape_and_cache_omni | infer | S | 4 | no |  | [📄](routing/reshape_and_cache_omni.md) | ⏳ |
| 59 | reshape_and_cache_with_stride | infer | S | 6 | no |  | [📄](routing/reshape_and_cache_with_stride.md) | ⏳ |
| 60 | ring_mla | infer | L | 7 | no | ⚠️ | [📄](routing/ring_mla.md) | ⏳ |
| 61 | rms_norm | infer | L | 10 | yes | ⚠️ | [📄](routing/rms_norm.md) | ⏳ |
| 62 | rms_norm_backward | train | S | 4 | no | ⚠️ | [📄](routing/rms_norm_backward.md) | ⏳ |
| 63 | rms_norm_with_stride | infer | S | 4 | no |  | [📄](routing/rms_norm_with_stride.md) | ⏳ |
| 64 | rope | infer | M | 6 | yes |  | [📄](routing/rope.md) | ⏳ |
| 65 | rope_grad | train | S | 4 | no |  | [📄](routing/rope_grad.md) | ⏳ |
| 66 | rope_q_concat | infer | S | 4 | no |  | [📄](routing/rope_q_concat.md) | ⏳ |
| 67 | scatter_elements_v2 | infer | S | 4 | no |  | [📄](routing/scatter_elements_v2.md) | ⏳ |
| 68 | self_attention | infer | XL | 27 | yes | ⚠️ | [📄](routing/self_attention.md) | [✅](ops/attention/self_attention/index.md) |
| 69 | send | infer | S | 4 | no |  | [📄](routing/send.md) | ⏳ |
| 70 | set_value | infer | S | 4 | no |  | [📄](routing/set_value.md) | ⏳ |
| 71 | slice | infer | M | 6 | yes |  | [📄](routing/slice.md) | ⏳ |
| 72 | softmax | infer | M | 6 | yes |  | [📄](routing/softmax.md) | ⏳ |
| 73 | sort | infer | M | 6 | yes | ⚠️ | [📄](routing/sort.md) | ⏳ |
| 74 | split | infer | M | 6 | yes |  | [📄](routing/split.md) | ⏳ |
| 75 | strided_batch_matmul | train | S | 4 | no | ⚠️ | [📄](routing/strided_batch_matmul.md) | ⏳ |
| 76 | swiglu_quant | infer | M | 6 | yes | ⚠️ | [📄](routing/swiglu_quant.md) | [✅](ops/activation/swiglu_quant/index.md) |
| 77 | topk_topp_sampling | infer | S | 4 | no | ⚠️ | [📄](routing/topk_topp_sampling.md) | ⏳ |
| 78 | transdata | infer | M | 6 | yes | ⚠️ | [📄](routing/transdata.md) | ⏳ |
| 79 | transpose | infer | M | 6 | yes |  | [📄](routing/transpose.md) | ⏳ |
| 80 | unpad | infer | S | 4 | no | ⚠️ | [📄](routing/unpad.md) | ⏳ |
| 81 | unpad_with_hidden_state | train | S | 4 | no |  | [📄](routing/unpad_with_hidden_state.md) | ⏳ |
| 82 | where | infer | S | 4 | no | ⚠️ | [📄](routing/where.md) | ⏳ |

---

> 总 Op 数: 82 | 路由文件: 82 | 知识条目: 9 | 生成时间: 2026-07-03
>
> 索引更新: 运行 `scripts/update_knowledge_index.py` 可自动刷新本文件