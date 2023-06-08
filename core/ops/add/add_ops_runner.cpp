

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
#include "add_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
AddOpsRunner::AddOpsRunner(const AddParam &param) : OpsRunner("AddOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "AddOperation::AddOperation called";
}

AddOpsRunner::~AddOpsRunner() {}

AsdOps::Status AddOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    if (param_.scale == 1) {
        kernelGraph_.inTensors = variantPack.inTensors;
        AsdOps::Tensor &aTensor = kernelGraph_.inTensors.at(0);
        AsdOps::Tensor &bTensor = kernelGraph_.inTensors.at(1);
        kernelGraph_.outTensors = variantPack.outTensors;
        AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

        kernelGraph_.nodes.resize(1);
        auto &addNode = kernelGraph_.nodes.at(0);

        addNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
        addNode.inTensors = {&aTensor, &bTensor};
        addNode.outTensors = {&operationOutTensor};
    } else {
        kernelGraph_.inTensors = variantPack.inTensors;
        AsdOps::Tensor &aTensor = kernelGraph_.inTensors.at(0);
        AsdOps::Tensor &bTensor = kernelGraph_.inTensors.at(1);
        kernelGraph_.outTensors = variantPack.outTensors;
        AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);
        kernelGraph_.internalTensors.resize(1);
        AsdOps::Tensor &mulsOutTensor = kernelGraph_.internalTensors.at(0);

        kernelGraph_.nodes.resize(2);
        auto &mulsNode = kernelGraph_.nodes.at(0);
        auto &addNode = kernelGraph_.nodes.at(1);

        mulsNode.opDesc = {0, "ElewiseOperation",
                           AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, param_.scale})};
        mulsNode.inTensors = {&aTensor};
        mulsNode.outTensors = {&mulsOutTensor};

        addNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
        addNode.inTensors = {&mulsOutTensor, &bTensor};
        addNode.outTensors = {&operationOutTensor};
    }
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
