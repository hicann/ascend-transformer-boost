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
#ifndef ACLTRANSFORMER_PLAN_H
#define ACLTRANSFORMER_PLAN_H
#include <map>
#include <set>
#include <functional>
#include <memory>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/runner.h"
#include "acltransformer/operation.h"

namespace AclTransformer {
using RunnerGraphNodeViewFunc =
    std::function<void(const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)>;

struct RunnerGraphNode {
    const Operation *operation = nullptr;
    std::shared_ptr<Runner> runner;
    AsdOps::SVector<AsdOps::Tensor *> inTensors;
    AsdOps::SVector<AsdOps::Tensor *> outTensors;
    AsdOps::SVector<RunnerGraphNodeViewFunc> inTensorViewFuncs;
    RunnerVariantPack runnerVariantPack;
};

struct RunnerGraph {
    AsdOps::SVector<AsdOps::Tensor> inTensors;
    AsdOps::SVector<AsdOps::Tensor> outTensors;
    AsdOps::SVector<AsdOps::Tensor> internalTensors;
    AsdOps::SVector<RunnerGraphNode> nodes;
    std::string ToString() const;
};

class MemAllocationSolver;

class Plan {
public:
    Plan();
    ~Plan();
    AsdOps::Status Setup(Handle handle, const VariantPack &variantPack);
    uint64_t GetWorkspaceSize();
    AsdOps::Status Execute(Handle handle, VariantPack &variantPack);

protected:
    void InitTensorMaxNodeMap();

private:
    void Reset();
    bool IsInternalTensor(const AsdOps::Tensor *tensor);
    int64_t GetInTensorId(const AsdOps::Tensor *tensor);
    int64_t GetOutTensorId(const AsdOps::Tensor *tensor);
    AsdOps::Status PreparseNodeRunnerVariantPack();
    AsdOps::Status RunNodeInTensorViewFuncs(size_t nodeId, RunnerGraphNode &node);
    void InferShapeNode(size_t nodeId, RunnerGraphNode &node);
    AsdOps::Status SetupAllRunners();
    AsdOps::Status CopyHostTilingToDevice(Handle handle, VariantPack &variantPack);
    void UpdateRunnerVariantPackBuffer(VariantPack &variantPack);
    void UpdateRunnerVariantPackTensorData(VariantPack &variantPack);
    AsdOps::Status ExecuteAllRunner(Handle &handle, VariantPack &variantPack);

protected:
    friend class PlanBuilder;
    std::string name_;
    RunnerGraph runnerGraph_;
    std::map<AsdOps::Tensor *, uint64_t> tensorMaxNodeIdMap_;
    std::map<uint64_t, std::set<AsdOps::Tensor *>> maxNodeIdTensorMap_;
    uint64_t selfIntermediateBufferSize_ = 0;
    uint64_t totalTilingBufferSize_ = 0;
    AsdOps::SVector<uint64_t> tilingBufferSizes_;
    std::vector<char> totalHostTilingBuffer_;
    uint64_t maxWorkspaceBufferSize_ = 0;
    AsdOps::SVector<uint64_t> workspaceBufferSizes_;
    uint64_t maxIntermediateBufferSize_ = 0;
    AsdOps::SVector<uint64_t> intermediateBufferSizes_;
    std::unique_ptr<MemAllocationSolver> memAllocatinSolver_;
};
} // namespace AclTransformer
#endif