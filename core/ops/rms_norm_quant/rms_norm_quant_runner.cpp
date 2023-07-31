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
    ASD_LOG(INFO) << "RmsNormQuantOpsRunner::RmsNormQuantOpsRunner";
    kernelGraph_.inTensors.resize(3);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &gammaTensor = kernelGraph_.inTensor.at(1);
    AsdOps::Tensor &betaTensor = kernelGraph_.inTensor.at(2);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &yTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormQuantNode = kernelGrpha_.nodes[0];
    AsdOps::OpParam::RmsNormQuant rmsNormQuantParam = {AsdOps::OpParam::RmsNormQuant::RMSNORM_QUANT};

    rmsNormQuantParam.input_scale = param_.inputScale;
    rmsNormQuantParam.input_offset = param_.inputOffset;

    ASD_LOG(INFO) << GetName() << " RmsNormQuantOperation opDesc rmsNormQuantParam.input_scale:" << rmsNormQuantParam.input_scale
                  << ", rmsNormQuantParam.input_offset:" << rmsNormQuantParam.input_offset;
    rmsNormQuantNode.opDesc = {0, "RmsNormQuantOperation", rmsNormQuantParam};

    rmsNormQuantNode.inTensors = {&xTensor, &gammaTensor, &betaTensor,};
    rmsNormQuantNode.outTensors = {&yTensor};
}

RmsNormQuantOpsRunner::~RmsNormQuantOpsRunner() {}
}