# fast_soft_max_grad — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "fast_soft_max_grad", category: "activation", tier: "S", type: "single"}
source:
  repo_path: "src/ops/ops_train/fast_soft_max_grad/"
  param_header: "include/atb/train_op_params.h"
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 4 文件 — operation + ops_runner（FastSoftMaxGrad）
## 2. Fast Softmax 反向传播（梯度计算），仅 A2 推理产品
## 3. Param: `headNum` (int32), `qSeqLen` (vector<int32>, ≤ 32)
## 4. Pipeline: grad output → Softmax backward → grad input
## 5. Related: `fast_soft_max` (前向), `self_attention` (梯度流经)
