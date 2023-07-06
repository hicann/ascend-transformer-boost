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
#include "acltransformer/graph_runner.h"
#include <sys/stat.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/utils/mem_allocation_solver/best_mem_allocation_solver.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "acltransformer/statistic.h"
#include "acltransformer/operation.h"

namespace AclTransformer {
std::string GraphRunner::Graph::ToString() const
{
    std::stringstream ss;

    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " " << TensorUtil::AsdOpsTensorToString(inTensors.at(i))
           << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " "
           << TensorUtil::AsdOpsTensorToString(inTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto &node = nodes.at(i);
        ss << "node[" << i << "] opeation:" << node.operation.get()
           << ", operationName:" << (node.operation ? node.operation->GetName() : "null")
           << ", runner:" << node.runner.get() << ", runnerName:" << (node.runner ? node.runner->GetName() : "null")
           << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt << " " << TensorUtil::AsdOpsTensorToString(*tensorIt)
               << std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt << " " << TensorUtil::AsdOpsTensorToString(*tensorIt)
               << std::endl;
        }
    }
    return ss.str();
}

void GraphRunner::Graph::InitTensorMaxNodeMap()
{
    tensorMaxNodeIdMap.clear();
    maxNodeIdTensorMap.clear();

    for (size_t i = 0; i < internalTensors.size(); ++i) {
        AsdOps::Tensor &internalTensor = internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
            auto &node = nodes.at(nodeId);
            for (auto inTensorIt : node.inTensors) {
                if (&internalTensor == inTensorIt) {
                    maxNodeId = nodeId;
                    dependNodeCount++;
                }
            }
        }
        tensorMaxNodeIdMap[&internalTensor] = maxNodeId;
        ASD_LOG_IF(dependNodeCount == 0, WARN) << "internal tensor[" << i << "] dependNodeCount is 0, graph wrong";
        maxNodeIdTensorMap[maxNodeId].insert(&internalTensor);
    }
}

GraphRunner::GraphRunner(const std::string &name) : Runner(name)
{
    memAllocatinSolver_.reset(new BestMemAllocationSolver());
}

GraphRunner::~GraphRunner() {}

GraphRunner::Graph &GraphRunner::GetGraph() { return runnerGraph_; }

AsdOps::Status GraphRunner::SetupImpl(const RunnerVariantPack &runnerVariantPack) { return AsdOps::Status::OkStatus(); }

uint64_t GraphRunner::GetTilingBufferSizeImpl() { return 0; }

void GraphRunner::FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize) {}

uint64_t GraphRunner::GetWorkspaceBufferSizeImpl() { return 0; }

uint64_t GraphRunner::GetIntermediateBufferSizeImpl() { return 0; }

AsdOps::Status GraphRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer