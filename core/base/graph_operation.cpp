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
#include "acltransformer/plan.h"
#include "acltransformer/runner/graph_runner.h"

namespace AclTransformer {
GraphOperation::GraphOperation(const std::string &name) : Operation(name) {}

GraphOperation::~GraphOperation() {}

uint64_t GraphOperation::GetInTensorCount() const { return operationGraph_.inTensorSize; }

uint64_t GraphOperation::GetOutTensorCount() const { return operationGraph_.outTensorSize; }

Runner *GraphOperation::CreateBestRunner() const
{
    GraphRunner *runner = new GraphRunner(GetName() + "Runner");
    auto &runnerGraph = runner->runnerGraph_;
    runnerGraph.inTensors.resize(operationGraph_.inTensorSize);
    runnerGraph.outTensors.resize(operationGraph_.outTensorSize);
    runnerGraph.internalTensors.resize(operationGraph_.intermediateTensorSize);
    runnerGraph.nodes.resize(operationGraph_.nodes.size());

    AsdOps::SVector<AsdOps::Tensor *, 256> fullTensorPtrs(operationGraph_.inTensorSize + operationGraph_.outTensorSize +
                                                          operationGraph_.intermediateTensorSize);
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
        const auto &opNode = operationGraph_.nodes.at(i);
        RunnerGraphNode &runnerNode = runnerGraph.nodes.at(i);
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
} // namespace AclTransformer