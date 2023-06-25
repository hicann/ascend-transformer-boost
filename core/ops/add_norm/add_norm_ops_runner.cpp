

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
#include "add_norm_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
AddNormOpsRunner::AddNormOpsRunner(const AddNormParam &param)
    : OpsRunner("AddNormOpsRunner", RUNNER_TYPE_ADD_NORM), param_(param)
{
    ASD_LOG(INFO) << "AddNormOpsRunner::AddNormOpsRunner";
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &yTensor = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors.at(3);
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);
    kernelGraph_.internalTensors.resize(3);
    AsdOps::Tensor &addNodeResultTensor = kernelGraph_.internalTensors.at(0);
    AsdOps::Tensor &layerNormMeanTensor = kernelGraph_.internalTensors.at(1);
    AsdOps::Tensor &layerNormVarianceTensor = kernelGraph_.internalTensors.at(2);

    kernelGraph_.nodes.resize(2);
    auto &addNode = kernelGraph_.nodes[0];
    auto &layerNormNode = kernelGraph_.nodes[1];

    addNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addNode.inTensors = {&xTensor, &yTensor};
    addNode.outTensors = {&addNodeResultTensor};

    AsdOps::OpParam::Norm normParam = {AsdOps::OpParam::Norm::NORM_LAYERNORM};
    normParam.begin_norm_axis = 1;
    normParam.begin_params_axis = 1;
    normParam.epsilon = param_.layerNormEps;
    ASD_LOG(INFO) << GetName() << " NormOperation opDesc normParam.begin_norm_axis:" << normParam.begin_norm_axis
                  << ", normParam.begin_params_axis:" << normParam.begin_params_axis
                  << ", normParam.epsilon:" << normParam.epsilon;
    layerNormNode.opDesc = {0, "NormOperation", normParam};
    layerNormNode.inTensors = {&addNodeResultTensor, &weightTensor, &biasTensor};
    layerNormNode.outTensors = {&resultTensor, &layerNormMeanTensor, &layerNormVarianceTensor};
}

AddNormOpsRunner::~AddNormOpsRunner() {}
} // namespace AclTransformer
