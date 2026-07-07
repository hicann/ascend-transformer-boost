# reduce_scatterv — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "reduce_scatterv", category: "communication", tier: "S", type: "single"}
source: {repo_path: "src/ops/ops_infer/reduce_scatterv/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 4 文件 — operation + hccl_runner (仅 HCCL)
## 2. Runner: ReduceScattervHcclRunner
## 3. Pipeline: 集合通信 — 可变长度 ReduceScatter。HCCL。
## 4. Related: `reduce_scatter` (等长版)
