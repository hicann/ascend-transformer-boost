# gmm_deq_swiglu_quant_gmm_deq — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "gmm_deq_swiglu_quant_gmm_deq", category: "activation", tier: "XL", type: "composite"}
source: {repo_path: "src/ops/ops_infer/gmm_deq_swiglu_quant_gmm_deq/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 4 文件 — operation + ops_runner
## 2. 复合 Pipeline: GMM(Dequant) → SwiGLU → Quant → GMM(Dequant)（4 阶段串联）
## 3. dtype 流转: int8 → fp16 → fp16 → int8 → fp16，含 2 个 dtype 断点
## 4. Runner: GMMDeqSwiGLuQuantGMMDeqOpsRunner（单一 OpsRunner，无 ACLNN）
## 5. Related: `mm_deq_swiglu_quant_mm_deq` (MM 变体), `swiglu_quant` (单 SwiGLU+Quant)
