/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ACLTRANSFORMER_GRAPH_RUNNER_H
#define ACLTRANSFORMER_GRAPH_RUNNER_H
#include <map>
#include <set>
#include <functional>
#include <memory>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/runner/runner.h"
#include "acltransformer/runner/runner_graph.h"

namespace AclTransformer {
class MemAllocationSolver;

class GraphRunner : public Runner {
public:
    GraphRunner(const std::string &name);
    ~GraphRunner();

protected:
    AsdOps::Status SetupImpl(const VariantPack &variantPack) override;
    uint64_t GetTilingBufferSizeImpl() override;
    void FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize) override;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    uint64_t GetIntermediateBufferSizeImpl() override;
    AsdOps::Status ExecuteImpl(Handle &handle, VariantPack &variantPack) override;

private:
    void Reset();
    bool IsInternalTensor(const AsdOps::Tensor *tensor);
    int64_t GetInTensorId(const AsdOps::Tensor *tensor);
    int64_t GetOutTensorId(const AsdOps::Tensor *tensor);
    AsdOps::Status PreparseNodeVariantPack();
    AsdOps::Status RunNodeInTensorViewFuncs(size_t nodeId, RunnerGraphNode &node);
    void InferShapeNode(size_t nodeId, RunnerGraphNode &node);
    AsdOps::Status SetupAllRunners();
    AsdOps::Status CopyHostTilingToDevice(Handle handle, VariantPack &variantPack);
    void UpdateVariantPackBuffer(VariantPack &variantPack);
    void UpdateVariantPackTensorData(VariantPack &variantPack);
    AsdOps::Status ExecuteAllRunner(Handle &handle, VariantPack &variantPack);
    void *GetInOrOutTensorData(AsdOps::Tensor *tensor, const VariantPack &variantPack);
    void CalcTilingBufferSize();
    void CalcIntermediateBufferSize();

protected:
    friend class PlanBuilder;
    friend class Operation;
    friend class GraphOperation;
    RunnerGraph runnerGraph_;
    uint64_t selfIntermediateBufferSize_ = 0;
    uint64_t totalTilingBufferSize_ = 0;
    AsdOps::SVector<uint64_t> tilingBufferSizes_;
    uint64_t maxWorkspaceBufferSize_ = 0;
    AsdOps::SVector<uint64_t> workspaceBufferSizes_;
    uint64_t maxIntermediateBufferSize_ = 0;
    AsdOps::SVector<uint64_t> intermediateBufferSizes_;
    std::unique_ptr<MemAllocationSolver> memAllocatinSolver_;
};
} // namespace AclTransformer
#endif