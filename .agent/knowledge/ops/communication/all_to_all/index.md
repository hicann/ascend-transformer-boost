# all_to_all — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "all_to_all", category: "communication", tier: "S", type: "single"}
source: {repo_path: "src/ops/ops_infer/all_to_all/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 6 文件 — operation + hccl_runner + lccl_runner
## 2. Runner: AllToAllHcclRunner / AllToAllLcclRunner（双后端）
## 3. Pipeline: 集合通信 — 各卡间数据全交换。HCCL/LCCL。
## 4. Related: `all_to_allv` (可变长度), `all_to_allvv2` (增强版)
