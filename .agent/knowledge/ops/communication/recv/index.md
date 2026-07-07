# recv — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "recv", category: "communication", tier: "S", type: "single"}
source: {repo_path: "src/ops/ops_infer/recv/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 4 文件 — operation + hccl_runner (仅 HCCL)
## 2. Runner: RecvHcclRunner
## 3. Pipeline: P2P 通信 — 点对点 Recv。HCCL。
## 4. Related: `send` (配对), `broadcast` (一对多)
