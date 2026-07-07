# mm_deq_swiglu_quant_mm_deq — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "mm_deq_swiglu_quant_mm_deq", category: "activation", tier: "M", type: "composite"}
source: {repo_path: "src/ops/ops_infer/mm_deq_swiglu_quant_mm_deq/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 4 文件 — operation + ops_runner
## 2. 复合 Pipeline: MatMul(Dequant) → SwiGLU → Quant → MatMul(Dequant)
## 3. GMM → MM 变体（GroupedMatMul → MatMul），Pipeline 结构同 `gmm_deq_swiglu_quant_gmm_deq`
## 4. Runner: MMDeqSwiGLuQuantMMDeqOpsRunner（无 ACLNN）
## 5. Related: `gmm_deq_swiglu_quant_gmm_deq` (GMM 变体), `swiglu_quant`
