# activation — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "activation", category: "activation", tier: "M", type: "composite"}
source:
  repo_path: "src/ops/ops_infer/activation/"
  kernel_path: "src/kernels/kernels/activation/"
  param_header: "include/atb/infer_op_params.h"
knowledge: {status: "complete", last_extracted: "2026-07-06", extractor_version: "1.0.0"}
```
## 1. Source: 10 文件 — 含 3 个 ACLNN Runner（activation/gelu/swiglu_forward）
## 2. 复合 Op：包含 GELU、SiLU/SwiGLU、ReLU、Tanh、Sigmoid 等多种激活函数
## 3. Runner 决策: ActivationAclnnRunner / GeluAclnnRunner / SwigluForwardAclnnRunner / ActivationOpsRunner
## 4. Pipeline: 逐元素单阶段，部分支持量化和 dtype 转换
## 5. Related: `swiglu_quant` (激活+量化), `fast_gelu` (GELU 变体)
