

/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
AddNormOpsRunner::AddNormOpsRunner(const AddNormParam &param) : OpsRunner("AddNormOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "AddNormOperation::AddNormOperation called";
    kernelGraph_.inTensors.resize(3);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors[0];
    AsdOps::Tensor &yTensor = kernelGraph_.inTensors[1];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[2];
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[3];
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors[0];
    kernelGraph_.internalTensors.resize(1);
    AsdOps::Tensor &addNodeResultTensor = kernelGraph_.internalTensors[0];

    kernelGraph_.nodes.resize(2);
    auto &addNode = kernelGraph_.nodes[0];
    auto &layerNormNode = kernelGraph_.nodes[1];

    addNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addNode.inTensors = {&xTensor, &yTensor};
    addNode.outTensors = {&addNodeResultTensor};
    // addNode.opDesc = {0, ""} todo
    layerNormNode.inTensors = {&addNodeResultTensor, &weightTensor, &biasTensor};
    layerNormNode.outTensors = {&resultTensor};
}

AddNormOpsRunner::~AddNormOpsRunner() {}
} // namespace AclTransformer
