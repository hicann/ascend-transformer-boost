# reduce_scatter — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "reduce_scatter", category: "communication", tier: "S", type: "single"}
source: {repo_path: "src/ops/ops_infer/reduce_scatter/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 6 文件 — operation + hccl_runner + lccl_runner
## 2. Runner: ReduceScatterHcclRunner / ReduceScatterLcclRunner
## 3. Pipeline: 集合通信 — Reduce 后 Scatter 到各卡。HCCL/LCCL。
## 4. Related: `all_reduce` (Reduce + Broadcast), `all_gather` (Gather 无 Reduce)
