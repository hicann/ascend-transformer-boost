# broadcast — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "broadcast", category: "communication", tier: "S", type: "single"}
source: {repo_path: "src/ops/ops_infer/broadcast/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 6 文件 — operation + hccl_runner + lccl_runner
## 2. Runner: BroadcastHcclRunner / BroadcastLcclRunner
## 3. Pipeline: 集合通信 — 单卡广播到所有卡。HCCL/LCCL。
## 4. Related: `all_gather` (多卡 Gather), `send`/`recv` (P2P)
