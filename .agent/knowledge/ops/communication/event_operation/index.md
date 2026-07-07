# event_operation — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "event_operation", category: "communication", tier: "S", type: "single"}
source: {repo_path: "src/ops/ops_infer/event_operation/"}
knowledge: {status: "complete", last_extracted: "2026-07-06"}
```
## 1. Source: 4 文件 — operation + event_runner
## 2. Runner: EventRunner（单一，同步原语）
## 3. Pipeline: 同步原语 — Event Record/Wait，无数据操作。
## 4. Related: 用于 Stream 同步，非数据通信 Op。
