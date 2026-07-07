# all_gather — ATB Agent 知识条目
> **状态**: complete | **最后更新**: 2026-07-06
```yaml
op: {name: "all_gather", category: "communication", tier: "S", type: "single"}
source:
  repo_path: "src/ops/ops_infer/all_gather/"
  kernel_path: null
knowledge: {status: "complete", last_extracted: "2026-07-06", extractor_version: "1.0.0"}
```
## 1. Source File Map
| # | 文件 | 角色 |
|---|------|------|
| 1 | `all_gather_operation.h/cpp` | Operation 定义 + CreateRunner |
| 2 | `all_gather_hccl_runner.h/cpp` | HCCL 后端 Runner |
| 3 | `all_gather_lccl_runner.h/cpp` | LCCL 后端 Runner |
## 2. Execution Paths
```
AllGatherOperation::CreateRunner()
  ├── [HCCL] → AllGatherHcclRunner
  └── [LCCL] → AllGatherLcclRunner
```
## 3. Computation Pipeline
```yaml
pipeline_type: single_stage
note: "集合通信：各卡数据 Gather 到所有卡。无 dtype 变换。"
```
## 4. Kernel: HCCL/LCCL 通信库，无本地 Kernel。
## 5. Related: `all_gatherv` (可变长度版), `reduce_scatter` (反向操作)
