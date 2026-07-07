# fast_soft_max — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "fast_soft_max", category: "activation", tier: "S", type: "single"}
source:
  repo_path: "src/ops/ops_train/fast_soft_max/"
  param_header: "include/atb/train_op_params.h"
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 4 文件 — operation + ops_runner（FastSoftMaxOperation + FastSoftMaxOpsRunner）
## 2. 训练侧 Fast Softmax（Unpad QK^T 输入），仅 A2 推理产品
## 3. Param: `headNum` (int32), `qSeqLen` (vector<int32>, ≤ 32)
## 4. Pipeline: unpad QK scores → Softmax(fp32 acc) → output
## 5. Related: `fast_soft_max_grad` (反向), `self_attention` (含 Softmax 阶段)
