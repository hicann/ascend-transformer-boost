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
#include "ffn_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/matmul.h>

namespace AclTransformer {
FfnOpsRunner::FfnOpsRunner(const FfnParam &param) : OpsRunner("FfnOpsRunner", RUNNER_TYPE_FFN), param_(param)
{
    ASD_LOG(INFO) << "FfnOpsRunner::FfnOpsRunner";
    kernelGraph_.inTensors.resize(3);
    AsdOps::Tensor &aTensor = kernelGraph_.inTensors[0];
    AsdOps::Tensor &bTensor = kernelGraph_.inTensors[1];
    AsdOps::Tensor &cTensor = kernelGraph_.inTensors[2];
    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[0];

    kernelGraph_.internalTensors.resize(2);
    AsdOps::Tensor &matmulOutTensor = kernelGraph_.internalTensors[0];
    AsdOps::Tensor &addOutTensor = kernelGraph_.internalTensors[1];

    kernelGraph_.nodes.resize(3);
    auto &matmulNode = kernelGraph_.nodes[0];
    auto &addNode = kernelGraph_.nodes[1];
    auto &geluNode = kernelGraph_.nodes[2];

    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB})};
    matmulNode.inTensors = {&aTensor, &bTensor};
    matmulNode.outTensors = {&matmulOutTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    addNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addNode.inTensors = {&matmulOutTensor, &cTensor};
    addNode.outTensors = {&addOutTensor};

    geluNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
    geluNode.inTensors = {&addOutTensor};
    geluNode.outTensors = {&operationOutTensor};
}

FfnOpsRunner::~FfnOpsRunner() {}
} // namespace AclTransformer