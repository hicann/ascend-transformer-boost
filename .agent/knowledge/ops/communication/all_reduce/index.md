# all_reduce — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "all_reduce", category: "communication", tier: "S", type: "single"}
source: {repo_path: "src/ops/ops_infer/all_reduce/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 6 文件 — operation.h/cpp + hccl_runner.h/cpp + lccl_runner.h/cpp
## 2. Runner: AllReduceHcclRunner / AllReduceLcclRunner（双后端）
## 3. Pipeline: 集合通信 — AllReduce（sum/min/max）。HCCL/LCCL 通信库。Operation 参数控制 reduce 类型。
## 4. Related: `reduce_scatter` (部分 Reduce), `all_gather` (Gather 无 Reduce)
