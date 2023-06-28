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
#include "acltransformer/plan_builder.h"
#include "acltransformer/plan.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
AsdOps::Status PlanBuilder::Build(const std::string &planName, const OperationGraph &opGraph, Plan &plan)
{
    try {
        ASD_LOG(INFO) << "PlanBuilder::Build opGraph:" << opGraph.ToString();
        return BuildImpl(planName, opGraph, plan);
    } catch (const std::exception &e) {
        ASD_LOG(ERROR) << "PlanBuilder::Build fail, exception:" << e.what();
        return AsdOps::Status::FailStatus(1, e.what());
    }
}

AsdOps::Status PlanBuilder::Build(const std::string &planName, const Operation *op, Plan &plan)
{
    OperationGraph opGraph;
    opGraph.inTensorSize = op->GetInTensorCount();
    opGraph.outTensorSize = op->GetOutTensorCount();
    opGraph.intermediateTensorSize = 0;

    opGraph.nodes.resize(1);
    OperationGraphNode &node = opGraph.nodes.at(0);
    node.operation = op;
    node.inTensorIds.resize(opGraph.inTensorSize);
    node.outTensorIds.resize(opGraph.outTensorSize);
    uint64_t tensorId = 0;
    for (size_t i = 0; i < opGraph.inTensorSize; ++i, ++tensorId) {
        node.inTensorIds.at(i) = tensorId;
    }
    for (size_t i = 0; i < opGraph.outTensorSize; ++i, ++tensorId) {
        node.outTensorIds.at(i) = tensorId;
    }

    return Build(planName, opGraph, plan);
}

AsdOps::Status PlanBuilder::BuildImpl(const std::string &planName, const OperationGraph &opGraph, Plan &plan)
{
    VariantPack variantPack;
    plan.name_ = planName;
    plan.runnerGraph_.inTensors.resize(opGraph.inTensorSize);
    plan.runnerGraph_.outTensors.resize(opGraph.outTensorSize);
    plan.runnerGraph_.internalTensors.resize(opGraph.intermediateTensorSize);
    std::vector<AsdOps::Tensor *> fullTensorPtrs(opGraph.inTensorSize + opGraph.outTensorSize +
                                                 opGraph.intermediateTensorSize);
    size_t offset = 0;
    for (size_t i = 0; i < plan.runnerGraph_.inTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &plan.runnerGraph_.inTensors.at(i);
    }
    for (size_t i = 0; i < plan.runnerGraph_.outTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &plan.runnerGraph_.outTensors.at(i);
    }
    for (size_t i = 0; i < plan.runnerGraph_.internalTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &plan.runnerGraph_.internalTensors.at(i);
    }

    for (auto &node : opGraph.nodes) {
        RunnerGraphNode runnerNode;
        runnerNode.operation = node.operation;
        runnerNode.runner.reset(node.operation->CreateBestRunner());
        runnerNode.inTensorViewFuncs = node.inTensorViewFuncs;
        runnerNode.inTensors.resize(node.inTensorIds.size());
        runnerNode.outTensors.resize(node.outTensorIds.size());
        for (size_t i = 0; i < node.inTensorIds.size(); ++i) {
            runnerNode.inTensors.at(i) = fullTensorPtrs.at(node.inTensorIds.at(i));
        }
        for (size_t i = 0; i < node.outTensorIds.size(); ++i) {
            runnerNode.outTensors.at(i) = fullTensorPtrs.at(node.outTensorIds.at(i));
        }
        plan.runnerGraph_.nodes.push_back(runnerNode);
    }

    ASD_LOG(INFO) << "PlanBuilder::Build success, RunnerGraph:" << plan.runnerGraph_.ToString();
    plan.InitTensorMaxNodeMap();
    return AsdOps::Status::OkStatus();
}

void PlanBuilder::LogOperationGraph(const OperationGraph &opGraph)
{
    ASD_LOG(INFO) << "PlanBuilder OperationGraph: " << opGraph.ToString();
}

void PlanBuilder::LogRunnerGraph(const RunnerGraph &runnerGraph)
{
    ASD_LOG(INFO) << "PlanBuilder runnerGraph.inTensorSize:" << runnerGraph.inTensors.size()
                  << ", outTensorSize:" << runnerGraph.outTensors.size()
                  << ", intermediateTensorSize:" << runnerGraph.internalTensors.size();
    for (size_t i = 0; i < runnerGraph.inTensors.size(); ++i) {
        ASD_LOG(INFO) << "PlanBuilder runnerGraph.inTensors[" << i << "]:" << &runnerGraph.inTensors.at(i)
                      << ", data:" << runnerGraph.inTensors.at(i).data;
    }
    for (size_t i = 0; i < runnerGraph.outTensors.size(); ++i) {
        ASD_LOG(INFO) << "PlanBuilder runnerGraph.outTensors[" << i << "]:" << &runnerGraph.outTensors.at(i)
                      << ", data:" << runnerGraph.outTensors.at(i).data;
    }

    ASD_LOG(INFO) << "PlanBuilder runnerGraph.node size:" << runnerGraph.nodes.size();
    for (size_t i = 0; i < runnerGraph.nodes.size(); ++i) {
        auto &node = runnerGraph.nodes.at(i);
        ASD_LOG(INFO) << "PlanBuilder runnerGraph.node[" << i << "] opeation:" << (void *)node.operation;
        for (auto tensorIt : node.inTensors) {
            ASD_LOG(INFO) << "PlanBuilder runnerGraph.node[" << i << "] inTensor:" << tensorIt;
        }
        for (auto tensorIt : node.outTensors) {
            ASD_LOG(INFO) << "PlanBuilder runnerGraph.node[" << i << "] outTensor:" << tensorIt;
        }
    }
}
} // namespace AclTransformer