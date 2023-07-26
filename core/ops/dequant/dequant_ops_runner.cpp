

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
#include "dequant_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>

namespace AclTransformer {
DequantOpsRunner::DequantOpsRunner(const DequantParam &param) 
	: OpsRunner("DequantOpsRunner", RUNNER_TYPE_QUANT), param_(param)
{
    ASD_LOG(INFO) << "DequantOperation::DequantOperation called";

    kernelGraph_.inTensors.resize(2);
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &aTensor = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &bTensor = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &dequantNode = kernelGraph_.nodes.at(0);
    dequantNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise(
                            {AsdOps::OpParam::Elewise::ELEWISE_DEQUANT, 0, 0, 0, 0})};
    dequantNode.inTensors = {&aTensor, &bTensor};
    dequantNode.outTensors = {&operationOutTensor};
}

DequantOpsRunner::~DequantOpsRunner() {}
} // namespace AclTransformer
