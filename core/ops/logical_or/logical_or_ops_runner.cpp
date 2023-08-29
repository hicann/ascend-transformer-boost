

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
#include "logical_or_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
LogicalOrOpsRunner::LogicalOrOpsRunner() : OpsRunner("LogicalOrOpsRunner")
{
    ASD_LOG(INFO) << "LogicalOrOpsRunner::LogicalOrOpsRunner called";
    kernelGraph_.inTensors.resize(2);
    AsdOps::Tensor &aTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &bTensor = kernelGraph_.inTensors.at(1);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &logicalOrNode = kernelGraph_.nodes.at(0);

    logicalOrNode.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_LOGICAL_OR})};
    logicalOrNode.inTensors = {&aTensor, &bTensor};
    logicalOrNode.outTensors = {&resultTensor};

}

LogicalOrOpsRunner::~LogicalOrOpsRunner() {}

} // namespace AclTransformer
