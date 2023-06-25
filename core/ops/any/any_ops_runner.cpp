

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
#include "any_ops_runner.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "op_desc_json.h"

namespace AclTransformer {
AnyOpsRunner::AnyOpsRunner(const AnyParam &param) : OpsRunner("AnyOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "AnyOpsRunner::AnyOpsRunner called";
    kernelGraph_.inTensors.resize(param_.kernelGraph["inTensors"].size());
    kernelGraph_.outTensors.resize(param_.kernelGraph["outTensors"].size());
    std::map<std::string, AsdOps::Tensor *> tensorMap;
    for (size_t i = 0; i < param_.kernelGraph["inTensors"].size(); ++i) {
        std::string tensorName = param_.kernelGraph["inTensors"].at(i).get<std::string>();
        tensorMap[tensorName] = &kernelGraph_.inTensors[i];
    }

    for (size_t i = 0; i < param_.kernelGraph["outTensors"].size(); ++i) {
        std::string tensorName = param_.kernelGraph["outTensors"].at(i).get<std::string>();
        tensorMap[tensorName] = &kernelGraph_.outTensors[i];
    }

    kernelGraph_.internalTensors.resize(param_.kernelGraph["internalTensors"].size());
    for (size_t i = 0; i < param_.kernelGraph["internalTensors"].size(); ++i) {
        std::string tensorName = param_.kernelGraph["internalTensors"].at(i).get<std::string>();
        tensorMap[tensorName] = &kernelGraph_.internalTensors[i];
    }
    for (auto it : tensorMap) {
        ASD_LOG(INFO) << "tensorMap.tensor:" << it.first << ", addr:" << it.second;
    }

    for (size_t i = 0; i < param_.kernelGraph["nodes"].size(); ++i) {
        const nlohmann::json jsonNode = param_.kernelGraph["nodes"][i];
        KernelGraphNode kernelNode;
        JsonToOpDesc(jsonNode, kernelNode.opDesc);

        const nlohmann::json jsonInTensors = jsonNode["inTensors"];
        int jsonInTensorSize = int(jsonInTensors.size());
        kernelNode.inTensors.resize(jsonInTensorSize);
        for (int i = 0; i < jsonInTensorSize; ++i) {
            std::string tensorName = jsonInTensors[i].get<std::string>();
            auto it = tensorMap.find(tensorName);
            if (it == tensorMap.end()) {
                ASD_LOG(ERROR) << "nodes[" << i << "] inTensor:" << tensorName << " not exist";
                return;
            }
            kernelNode.inTensors[i] = it->second;
        }

        const nlohmann::json jsonOutTensors = jsonNode["outTensors"];
        int jsonOutTensorSize = int(jsonOutTensors.size());
        kernelNode.outTensors.resize(jsonOutTensorSize);
        for (int i = 0; i < jsonOutTensorSize; ++i) {
            std::string tensorName = jsonOutTensors[i].get<std::string>();
            auto it = tensorMap.find(tensorName);
            if (it == tensorMap.end()) {
                ASD_LOG(ERROR) << "nodes[" << i << "] outTensor:" << tensorName << " not exist";
                return;
            }
            kernelNode.outTensors[i] = it->second;
        }

        kernelGraph_.nodes.push_back(kernelNode);
    }
}

AnyOpsRunner::~AnyOpsRunner() {}
} // namespace AclTransformer
