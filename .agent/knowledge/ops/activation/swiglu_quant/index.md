# swiglu_quant — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "swiglu_quant", category: "activation", tier: "M", type: "fusion"}
source:
  repo_path: "src/ops/ops_infer/swiglu_quant/"
  param_header: "include/atb/infer_op_params.h"
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 6 文件 — operation + aclnn_runner + ops_runner
## 2. SwiGLU 激活 + 量化融合：SwiGLU(x) = x ⊙ SiLU(gate) → Quantize
## 3. Pipeline: fp16 input → SiLU(gate) → Mul(x) → int8 output (dtype 断点)
## 4. Runner: SwigluQuantAclnnRunner (950) / SwigluQuantOpsRunner (fallback)
## 5. Related: `activation` (通用激活), `gmm_deq_swiglu_quant_gmm_deq` (SwiGLU+Dequant+GMM)
