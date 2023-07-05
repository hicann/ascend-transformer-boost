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
#include "acltransformer/runner/runner_graph.h"
#include <sstream>
#include <asdops/utils/log/log.h>

namespace AclTransformer {
std::string RunnerGraph::ToString() const
{
    std::stringstream ss;

    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << ", data:" << inTensors.at(i).data;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << ", data:" << outTensors.at(i).data;
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto &node = nodes.at(i);
        ss << "node[" << i << "] opeation:" << (void *)node.operation;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt;
        }
    }
    return ss.str();
}

void RunnerGraph::InitTensorMaxNodeMap()
{
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
} // namespace AclTransformer