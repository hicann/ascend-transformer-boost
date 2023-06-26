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
#include <asdops/utils/svector/svector.h>
#include "acltransformer/runner.h"
#include "acltransformer/operation.h"

namespace AclTransformer {
using RunnerGraphNodeViewFunc =
    std::function<void(const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)>;

struct RunnerGraphNode {
    Operation *operation = nullptr;
    Runner *runner = nullptr;
    AsdOps::SVector<AsdOps::Tensor *> inTensors;
    AsdOps::SVector<AsdOps::Tensor *> outTensors;
    AsdOps::SVector<RunnerGraphNodeViewFunc> inTensorViewFuncs;
    VariantPack variantPack;
};

struct RunnerGraph {
    std::string name;
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
    void LogVariantPack(const VariantPack &variantPack);

protected:
    friend class PlanBuilder;
    RunnerGraph runnerGraph_;
    uint64_t totalWorkspaceSize_ = 0;
    AsdOps::SVector<uint64_t> workspaceSizes_;
    uint64_t intermediateSize_ = 0;
    MemAllocationSolver *memAllocatinSolver_ = nullptr;
    std::map<AsdOps::Tensor *, uint64_t> tensorMaxNodeIdMap_;
    std::map<uint64_t, std::set<AsdOps::Tensor *>> maxNodeIdTensorMap_;
};
} // namespace AclTransformer
#endif