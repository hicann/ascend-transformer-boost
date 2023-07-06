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
#include "acltransformer/graph_operation.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/graph_runner.h"

namespace AclTransformer {
static std::string JoinInts(const AsdOps::SVector<uint64_t> &ids)
{
    std::string ret;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i == 0) {
            ret.append(std::to_string(i));
        } else {
            ret.append(", " + std::to_string(i));
        }
    }
    return ret;
}

std::string GraphOperation::Graph::ToString() const
{
    std::stringstream ss;
    ss << "inTensorSize:" << inTensorSize << ", outTensorSize:" << outTensorSize
       << ", intermediateTensorSize:" << intermediateTensorSize;
    for (size_t i = 0; i < nodes.size(); ++i) {
        ss << "\nnode[" << i << "]: operation:" << nodes.at(i).operation << ", inTensorIds:["
           << JoinInts(nodes.at(i).inTensorIds) << "], outTensorIds:[" << JoinInts(nodes.at(i).outTensorIds) << "]";
    }
    return ss.str();
}

GraphOperation::GraphOperation(const std::string &name) : Operation(name) {}

GraphOperation::~GraphOperation() {}

uint64_t GraphOperation::GetInTensorCount() const { return opGraph_.inTensorSize; }

uint64_t GraphOperation::GetOutTensorCount() const { return opGraph_.outTensorSize; }

Runner *GraphOperation::CreateBestRunner() const
{
    GraphRunner *runner = new GraphRunner(GetName() + "Runner");
    if (runner == nullptr) {
        ASD_LOG(ERROR) << name_ << " new GraphRunner fail";
        return nullptr;
    }

    GraphRunner::Graph &runnerGraph = runner->GetGraph();
    runnerGraph.inTensors.resize(opGraph_.inTensorSize);
    runnerGraph.outTensors.resize(opGraph_.outTensorSize);
    runnerGraph.internalTensors.resize(opGraph_.intermediateTensorSize);
    runnerGraph.nodes.resize(opGraph_.nodes.size());

    const int totalTensorCount = 512;
    AsdOps::SVector<AsdOps::Tensor *, totalTensorCount> fullTensorPtrs(opGraph_.inTensorSize + opGraph_.outTensorSize +
                                                                       opGraph_.intermediateTensorSize);
    size_t offset = 0;
    for (size_t i = 0; i < runnerGraph.inTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &runnerGraph.inTensors.at(i);
    }
    for (size_t i = 0; i < runnerGraph.outTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &runnerGraph.outTensors.at(i);
    }
    for (size_t i = 0; i < runnerGraph.internalTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &runnerGraph.internalTensors.at(i);
    }

    for (size_t i = 0; i < runnerGraph.nodes.size(); ++i) {
        const auto &opNode = opGraph_.nodes.at(i);
        GraphRunner::Node &runnerNode = runnerGraph.nodes.at(i);
        runnerNode.operation = opNode.operation;
        runnerNode.runner.reset(opNode.operation->CreateBestRunner());
        runnerNode.inTensorViewFuncs = opNode.inTensorViewFuncs;
        runnerNode.inTensors.resize(opNode.inTensorIds.size());
        runnerNode.outTensors.resize(opNode.outTensorIds.size());
        for (size_t i = 0; i < opNode.inTensorIds.size(); ++i) {
            runnerNode.inTensors.at(i) = fullTensorPtrs.at(opNode.inTensorIds.at(i));
        }
        for (size_t i = 0; i < opNode.outTensorIds.size(); ++i) {
            runnerNode.outTensors.at(i) = fullTensorPtrs.at(opNode.outTensorIds.at(i));
        }
    }

    runnerGraph.InitTensorMaxNodeMap();
    return runner;
}

RunnerBuilder *GraphOperation::FindBestRunnerBuilder() const { return nullptr; }
} // namespace AclTransformer