/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "acltransformer/runner.h"
#include "acltransformer/operation.h"

namespace AclTransformer {
struct RunnerGraphNode {
    Operation *operation = nullptr;
    Runner *runner = nullptr;
    std::vector<AsdOps::Tensor *> inTensors;
    std::vector<AsdOps::Tensor *> outTensors;
    VariantPack variantPack;
};

struct RunnerGraph {
    std::string name;
    std::vector<AsdOps::Tensor> inTensors;
    std::vector<AsdOps::Tensor> outTensors;
    std::vector<AsdOps::Tensor> internalTensors;
    std::vector<RunnerGraphNode> nodes;
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
    uint64_t workspaceSize_ = 0;
    uint64_t intermediateSize_ = 0;
    MemAllocationSolver *memAllocatinSolver_ = nullptr;
    std::map<AsdOps::Tensor *, uint64_t> tensorMaxNodeIdMap_;
    std::map<uint64_t, std::set<AsdOps::Tensor *>> maxNodeIdTensorMap_;
};
} // namespace AclTransformer
#endif