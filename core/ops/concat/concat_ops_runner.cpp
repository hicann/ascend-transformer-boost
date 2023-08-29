

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
#include "concat_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
ConcatOpsRunner::ConcatOpsRunner(const ConcatParam &param) : OpsRunner("ConcatOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "ConcatOpsRunner::ConcatOpsRunner called";
    kernelGraph_.inTensors.resize(2);
    AsdOps::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &yTensor = kernelGraph_.inTensors.at(1);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &concatNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Concat concatNodeParam = {param_.concatDim};

    concatNode.opDesc = {0, "ConcatOperation", concatNodeParam};
    concatNode.inTensors = {&xTensor, &yTensor};
    concatNode.outTensors = {&outTensor};
}

ConcatOpsRunner::~ConcatOpsRunner() {}

} // namespace AclTransformer
