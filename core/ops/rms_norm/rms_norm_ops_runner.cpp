

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
#include "rms_norm_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
RmsNormOpsRunner::RmsNormOpsRunner(const RmsNormParam &param)
    : OpsRunner("RmsNormOpsRunner", RUNNER_TYPE_RMS_NORM), param_(param)
{
    ASD_LOG(INFO) << "RmsNormOpsRunner::RmsNormOpsRunner called";
    kernelGraph_.inTensors.resize(2);
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[0];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[1];

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors[0];

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Norm rmsNormParam = {AsdOps::OpParam::Norm::NORM_RMSNORM};
    rmsNormParam.epsilon = param_.rmsNormEps;
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputTensor, &weightTensor};
    rmsNormNode.outTensors = {&resultTensor};
}

RmsNormOpsRunner::~RmsNormOpsRunner() {}
} // namespace AclTransformer
