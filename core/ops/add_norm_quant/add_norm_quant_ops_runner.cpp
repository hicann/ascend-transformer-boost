

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
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"
#include "add_norm_quant_ops_runner.h"


namespace AclTransformer {
AddNormQuantOpsRunner::AddNormQuantOpsRunner(const AddNormQuantParam &param)
    : OpsRunner("AddNormQuantOpsRunner", RUNNER_TYPE_ADD_NORM_QUANT), param_(param)
{
    ASD_LOG(INFO) << "AddNormQuantOpsRunner::AddNormQuantOpsRunner";
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &gammaTensor = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &betaTensor = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &resinTensor = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors.resize(2);
    AsdOps::Tensor &zTensor = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &resTensor = kernelGraph_.outTensors.at(1);

    kernelGraph_.nodes.resize(1);
    auto &layerNormNode = kernelGraph_.nodes[0];
    AsdOps::OpParam::Norm normParam = {AsdOps::OpParam::Norm::NORM_POSTLAYERNORMQUANT};

    normParam.begin_norm_axis = 1;
    normParam.begin_params_axis = 1;
    normParam.epsilon = param_.layerNormEps;
    normParam.input_scale = param_.inputScale;
    normParam.input_offset = param_.inputOffset;
    normParam.input_alpha = param_.inputAlpha;
    normParam.ops_mode = 0;

    ASD_LOG(INFO) << GetName() << " NormOperation opDesc normParam.begin_norm_axis:" << normParam.begin_norm_axis
                  << ", normParam.begin_params_axis:" << normParam.begin_params_axis
                  << ", normParam.epsilon:" << normParam.epsilon;
    layerNormNode.opDesc = {0, "NormOperation", normParam};

    layerNormNode.inTensors = {&xTensor, &gammaTensor, &betaTensor, &resinTensor};
    layerNormNode.outTensors = {&zTensor, &resTensor};
}

AddNormQuantOpsRunner::~AddNormQuantOpsRunner() {}
} // namespace AclTransformer
