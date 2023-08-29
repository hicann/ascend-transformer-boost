

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
#include "gather_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
GatherOpsRunner::GatherOpsRunner(const GatherParam &param) : OpsRunner("GatherOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "GatherOpsRunner::GatherOpsRunner called";
    kernelGraph_.inTensors.resize(2);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &yTensor = kernelGraph_.inTensors.at(1);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &gatherNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Gather gatherNodeParam = {AsdOps::OpParam::Gather::GatherType::GATHER_V2, param_.batchDims, param_.axis};

    gatherNode.opDesc = {0, "GatherOperation", gatherNodeParam};
    gatherNode.inTensors = {&xTensor, &yTensor};
    gatherNode.outTensors = {&outTensor};
}

GatherOpsRunner::~GatherOpsRunner() {}

} // namespace AclTransformer
