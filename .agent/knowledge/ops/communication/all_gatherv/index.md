# all_gatherv — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "all_gatherv", category: "communication", tier: "S", type: "single"}
source: {repo_path: "src/ops/ops_infer/all_gatherv/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: `all_gatherv_operation.h/cpp`, `all_gatherv_hccl_runner.h/cpp` (4 文件)
## 2. Runner: HcclRunner（仅 HCCL，无 LCCL 变体）
## 3. Pipeline: 集合通信 — 可变长度 AllGather。HCCL 通信库。
## 4. Related: `all_gather` (等长版), `all_gathervv2` (增强版)
