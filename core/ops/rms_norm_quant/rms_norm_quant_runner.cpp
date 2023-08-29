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
#include "rms_norm_quant_ops_runner.h"

namespace AclTransformer {
RmsNormQuantOpsRunner::RmsNormQuantOpsRunner(const RmsNormQuantParam &param)
    : OpsRunner("RmsNormQuantOpsRunner", RUNNER_TYPE_RMS_NORM_QUANT), param_(param)
{
    const std::size_t inTensorCount = 3;

    ASD_LOG(INFO) << "RmsNormQuantOpsRunner::RmsNormQuantOpsRunner called";
    kernelGraph_.inTensors.resize(inTensorCount);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &gammaTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &betaTensor = kernelGraph_.inTensors[inTensorNum++];

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &yTensor = kernelGraph_.outTensors[0];

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes[0];
    AsdOps::OpParam::Norm rmsNormQuantParam = {AsdOps::OpParam::Norm::NORM_RMSNORMQUANT};

    rmsNormQuantParam.input_scale = param_.inputScale;
    rmsNormQuantParam.input_offset = param_.inputOffset;
    rmsNormQuantParam.epsilon = param_.rmsNormEps;

    rmsNormNode.opDesc = {0, "NormOperation", rmsNormQuantParam};
    rmsNormNode.inTensors = {&xTensor, &gammaTensor, &betaTensor};
    rmsNormNode.outTensors = {&yTensor};
}

RmsNormQuantOpsRunner::~RmsNormQuantOpsRunner() {}
} // namespace AclTransformer
