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
#include "acltransformer/plan_builder.h"
#include "acltransformer/plan.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
Plan *PlanBuilder::Build(const VariantPack &variantPack, const OperationGraph &opGraph)
{
    try {
        Plan *plan = BuildImpl(variantPack, opGraph);
        return plan;
    } catch (const std::exception &e) {
        ASD_LOG(ERROR) << "build plan exception:" << e.what();
        return nullptr;
    }
}

Plan *PlanBuilder::BuildImpl(const VariantPack &variantPack, const OperationGraph &opGraph)
{
    LogOperationGraph(opGraph);

    Plan *plan = new Plan();
    if (opGraph.inTensorSize != variantPack.inTensors.size() ||
        opGraph.outTensorSize != variantPack.outTensors.size()) {
        ASD_LOG(ERROR) << "PlanBuilder opGraph.inTensorSize:" << opGraph.inTensorSize
                       << ", variantPack.inTensors.size:" << variantPack.inTensors.size()
                       << ", opGraph.outTensorSize:" << opGraph.outTensorSize
                       << ", variantPack.outTensors.size():" << variantPack.outTensors.size();
    }
    for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
        ASD_LOG(INFO) << "PlanBuilder variantPack.inTensors[" << i
                      << "]: " << AsdOpsTensorToString(variantPack.inTensors.at(i));
    }
    for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
        ASD_LOG(INFO) << "PlanBuilder variantPack.outTensors[" << i
                      << "]: " << AsdOpsTensorToString(variantPack.outTensors.at(i));
    }

    plan->runnerGraph_.name = opGraph.name;
    plan->runnerGraph_.inTensors = variantPack.inTensors;
    plan->runnerGraph_.outTensors = variantPack.outTensors;
    plan->runnerGraph_.internalTensors.resize(opGraph.intermediateTensorSize);
    std::vector<AsdOps::Tensor *> fullTensorPtrs;
    for (size_t i = 0; i < plan->runnerGraph_.inTensors.size(); ++i) {
        fullTensorPtrs.push_back(&plan->runnerGraph_.inTensors.at(i));
    }
    for (size_t i = 0; i < plan->runnerGraph_.outTensors.size(); ++i) {
        fullTensorPtrs.push_back(&plan->runnerGraph_.outTensors.at(i));
    }
    for (size_t i = 0; i < plan->runnerGraph_.internalTensors.size(); ++i) {
        fullTensorPtrs.push_back(&plan->runnerGraph_.internalTensors.at(i));
    }

    for (auto &node : opGraph.nodes) {
        Operation *op = node.operation;
        RunnerGraphNode runnerNode;
        runnerNode.operation = node.operation;
        runnerNode.runner = op->CreateBestRunner(variantPack);
        runnerNode.inTensors.resize(node.inTensorIds.size());
        runnerNode.outTensors.resize(node.outTensorIds.size());

        for (size_t i = 0; i < node.inTensorIds.size(); ++i) {
            runnerNode.inTensors.at(i) = fullTensorPtrs.at(node.inTensorIds.at(i));
        }
        for (size_t i = 0; i < node.outTensorIds.size(); ++i) {
            runnerNode.outTensors.at(i) = fullTensorPtrs.at(node.outTensorIds.at(i));
        }
        plan->runnerGraph_.nodes.push_back(runnerNode);
    }

    LogRunnerGraph(plan->runnerGraph_);

    plan->InitTensorMaxNodeMap();

    return plan;
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