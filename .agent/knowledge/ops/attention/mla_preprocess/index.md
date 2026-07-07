# mla_preprocess — ATB Agent 知识条目

> **状态**: complete | **最后更新**: 2026-07-06

```yaml
op:
  name: "mla_preprocess"
  category: "attention"
  tier: "L"
  type: "fusion"

source:
  repo_path: "src/ops/ops_infer/mla_preprocess/"
  kernel_path: "src/kernels/mixkernels/mla_preprocess"
  param_header: "include/atb/infer_op_params.h"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  extractor_version: "1.0.0"
```

## 1. Source File Map

| # | 文件 | 角色 |
|---|------|------|
| 1 | `mla_preprocess_operation.h` | Operation 定义 |
| 2 | `mla_preprocess_operation.cpp` | CreateRunner（3 路）+ InferShape |
| 3 | `mla_preprocess_aclnn_runner.h/cpp` | ACLNN Runner |
| 4 | `mla_preprocess_ops_runner.h/cpp` | OpsRunner（标准） |
| 5 | `mla_preprocess_ops_runner_split.h/cpp` | OpsRunner（Split 变体） |
| 6 | `atb_acl_mla_preprocess.cpp` | ACL 辅助 |

## 2. Reading Order

```
[1] mla_preprocess_operation.h → 融合算子接口
[2] mla_preprocess_operation.cpp → CreateRunner 决策树
[3] mla_preprocess_aclnn_runner.cpp → ACLNN 路径
[4] mla_preprocess_ops_runner.cpp → Ops 标准路径
[5] mla_preprocess_ops_runner_split.cpp → Ops Split 变体
```

## 3. Parameter Constraints

```cpp
struct MlaPreprocessParam {
    uint32_t wdqDim = 0;          // matmul 后拆分 dim
    uint32_t qRopeDim = 0;        // Q 进入 RoPE 的 dim
    uint32_t kRopeDim = 0;        // K 进入 RoPE 的 dim
    float epsilon = 1e-5;         // Norm epsilon
    int32_t qRotaryCoeff = 2;     // Q 旋转系数（2/4/headDim）
    int32_t kRotaryCoeff = 2;     // K 旋转系数（2/4/headDim）
    bool transposeWdq = true;     // WDQ 是否转置
    bool transposeWuq = true;     // WUQ 是否转置
    bool transposeWuk = true;     // WUK 是否转置
    enum CacheMode { KVCACHE=0, KROPE_CTKV, INT8_NZCACHE, NZCACHE };
    CacheMode cacheMode = KVCACHE;
};
```

| 参数 | 约束 |
|------|------|
| `wdqDim` | matmul 后 Q 拆分维度 |
| `qRopeDim` / `kRopeDim` | RoPE 旋转维度 |
| `qRotaryCoeff` / `kRotaryCoeff` | 2/4 或 headDim |
| `cacheMode` | KVCACHE / KROPE_CTKV / INT8_NZCACHE / NZCACHE |

## 4. Computation Pipeline

```yaml
pipeline_type: multi_stage_fusion
stages:
  - stage: QKV 投影
    note: "x @ WDQ → Q, x @ WUK → K, x @ WUV → V"
  - stage: RoPE
    note: "Q[..., :qRopeDim] + K[..., :kRopeDim] 旋转位置编码"
  - stage: KV Cache
    note: "根据 cacheMode 将 K,V 写入指定 Cache 格式"
  - stage: RMS Norm
    note: "可选 Norm（epsilon 控制）"
```

## 5. Execution Paths

```
MlaPreprocessOperation::CreateRunner()
  │
  ├── [ACLNN 路径] → MlaPreprocessAclnnRunner
  └── [Ops 路径]
       ├── MlaPreprocessOpsRunner（标准）
       └── MlaPreprocessOpsRunnerSplit（Split 变体）
```

| 平台 | Runner | 支持 |
|------|--------|------|
| 910B | OpsRunner（标准/Split） | — |
| 950 | ACLNN Runner + OpsRunner | ACLNN 加速 |

## 6. Kernel Dependencies

| Kernel | 文件 |
|--------|------|
| `mla_preprocess` | `src/kernels/mixkernels/mla_preprocess/` |

## 7. Known Issues

| # | 问题 | 状态 |
|---|------|------|
| 1 | 融合 4 阶段（投影+RoPE+Cache+Norm），Golden 需匹配 Pipeline | 注意 |

## 9. Related Ops

| 算子 | 关系 |
|------|------|
| `multi_latent_attention` | 下游：使用 preprocess 输出做 MLA |
| `ring_mla` | 分布式 MLA：Ring 通信 + MLA，同样需要 preprocess |
| `rope` | 融合的 RoPE 阶段 |
