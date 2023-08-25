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
#include "rms_pre_norm_quant_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
RmsPreNormQuantOpsRunner::RmsPreNormQuantOpsRunner(const RmsPreNormQuantParam &param)
    : OpsRunner("RmsPreNormQuantOpsRunner", RUNNER_TYPE_RMS_PRE_NORM_QUANT), param_(param)
{
    ASD_LOG(INFO) << "RmsPreNormQuantOpsRunner::RmsPreNormQuantOpsRunner called";
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors[0];
    AsdOps::Tensor &resinTensor = kernelGraph_.inTensors[1];
    AsdOps::Tensor &gammaTensor = kernelGraph_.inTensors[2];
    AsdOps::Tensor &betaTensor = kernelGraph_.inTensors[3];

    kernelGraph_.outTensors.resize(2);
    AsdOps::Tensor &zTensor = kernelGraph_.outTensors[0];
    AsdOps::Tensor &resTensor = kernelGraph_.outTensors[1];

    kernelGraph_.nodes.resize(1);
    auto &normNode = kernelGraph_.nodes[0];
    AsdOps::OpParam::Norm normParam = {AsdOps::OpParam::Norm::NORM_RMSPRENORMQUANT};
    
    normParam.input_scale = param_.inputScale;
    normParam.input_offset = param_.inputOffset;
    normParam.epsilon = param_.rmsNormEps;

    normNode.opDesc = {0, "NormOperation", normParam};
    normNode.inTensors = {&xTensor, &resinTensor, &gammaTensor, &betaTensor};
    normNode.outTensors = {&zTensor, &resTensor};
}

RmsPreNormQuantOpsRunner::~RmsPreNormQuantOpsRunner() {}
} // namespace AclTransformer
