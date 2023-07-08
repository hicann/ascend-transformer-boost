

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
    AsdOps::Tensor &gammaTensor = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &betaTensor = kernelGraph_.inTensors.at(3);
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);
    kernelGraph_.nodes.resize(1);
    auto &addLayerNormNode = kernelGraph_.nodes[0];
    AsdOps::OpParam::Norm opCommonParam = {AsdOps::OpParam::Norm::NORM_POSTLAYERNORM};
    opCommonParam.ops_mode = 0;
    opCommonParam.epsilon = param_.layerNormEps;
    opCommonParam.zoom_scale_value = param_.zoom_scale;
    ASD_LOG(INFO) << GetName() << ", opCommonParam.epsilon:" << opCommonParam.epsilon;
    addLayerNormNode.opDesc = {0, "NormOperation", opCommonParam};
    addLayerNormNode.inTensors = {&xTensor, &yTensor, &gammaTensor, &betaTensor};
    addLayerNormNode.outTensors = {&resultTensor};
}

AddNormOpsRunner::~AddNormOpsRunner() {}
} // namespace AclTransformer