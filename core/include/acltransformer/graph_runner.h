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
#include "acltransformer/runner.h"

namespace AclTransformer {
class MemAllocationSolver;
class Operation;

class GraphRunner : public Runner {
public:
    using NodeViewFunc =
        std::function<void(const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)>;

    enum TensorType {
        INTERMEDIATE_TENSOR = 0,
        NOT_INTERMEDIATE_TENSOR,
    };
    struct Node {
        std::shared_ptr<Operation> operation;
        std::shared_ptr<Runner> runner;
        AsdOps::SVector<AsdOps::Tensor *> inTensors;
        AsdOps::SVector<AsdOps::Tensor *> outTensors;
        AsdOps::SVector<NodeViewFunc> inTensorViewFuncs;
        RunnerVariantPack runnerVariantPack;
        bool useVariantPackParam = false;
        AsdOps::SVector<TensorType> inTensorTypes;
        AsdOps::SVector<TensorType> outTensorTypes;
    };

    struct Graph {
        AsdOps::SVector<AsdOps::Tensor> inTensors;
        AsdOps::SVector<AsdOps::Tensor> outTensors;
        AsdOps::SVector<AsdOps::Tensor> internalTensors;
        AsdOps::SVector<Node, 64> nodes;
        std::map<AsdOps::Tensor *, uint64_t> tensorMaxNodeIdMap;
        std::map<uint64_t, std::set<AsdOps::Tensor *>> maxNodeIdTensorMap;
        std::string ToString() const;
        void Init();

    private:
        void InitTensorMaxNodeMap();
        void InitTensorType();
        bool IsInternalTensor(const AsdOps::Tensor *tensor);
    };

    GraphRunner(const std::string &name);
    ~GraphRunner();
    Graph &GetGraph();

protected:
    AsdOps::Status SetupImpl(const RunnerVariantPack &runnerVariantPack) override;
    uint64_t GetTilingBufferSizeImpl() override;
    void FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize) override;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    uint64_t GetIntermediateBufferSizeImpl() override;
    AsdOps::Status ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack) override;

private:
    void Reset();
    AsdOps::Status PreparseNodeVariantPack();
    AsdOps::Status RunNodeInTensorViewFuncs(size_t nodeId, Node &node);
    void InferShapeNode(size_t nodeId, Node &node);
    AsdOps::Status SetupAllRunners(const RunnerVariantPack &runnerVariantPack);
    void CalcTilingBufferSize();
    void CalcIntermediateBufferSize();
    void UpdateVariantPackBuffer(RunnerVariantPack &runnerVariantPack);
    void UpdateVariantPackTensorData(RunnerVariantPack &runnerVariantPack);
    AsdOps::Status ExecuteAllRunner(Handle &handle, RunnerVariantPack &runnerVariantPack);

private:
    Graph runnerGraph_;
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