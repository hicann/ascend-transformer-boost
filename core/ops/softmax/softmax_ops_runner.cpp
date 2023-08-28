

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 *  * Licensed under the Apache License, Version 2.0 (the "License");
 *  * you may not use this file except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 *  * http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  */
#include "softmax_ops_runner.h"
#include "acltransformer/params/softmax.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
SoftmaxOpsRunner::SoftmaxOpsRunner(const SoftmaxParam &param) : OpsRunner("SoftmaxOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "SoftmaxOpsRunner::SoftmaxOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &sortNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Norm asdParam;
    asdParam.normType = AsdOps::OpParam::Norm::NormType::NORM_SOFTMAX;
    for (std::size_t i = 0; i < param_.axes.size(); ++i) {
        asdParam.axes.push_back(param_.axes[i]);
    }

    sortNode.opDesc = {0, "SoftmaxOperation", asdParam};
    sortNode.inTensors = {&xTensor};
    sortNode.outTensors = {&resultTensor};
}

SoftmaxOpsRunner::~SoftmaxOpsRunner() {}

} // namespace AclTransformer
