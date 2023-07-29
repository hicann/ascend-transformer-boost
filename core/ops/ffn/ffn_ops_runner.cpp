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
}

FfnOpsRunner::~FfnOpsRunner() {}

AsdOps::Status FfnOpsRunner::SetupKernelGraphWithBias(const RunnerVariantPack &runnerVariantPack) {
    kernelGraph_.inTensors.resize(3);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &aTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &bTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &cTensor = kernelGraph_.inTensors[inTensorNum++];
    
    kernelGraph_.outTensors.resize(1);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[outTensorNum++];

    kernelGraph_.internalTensors.resize(2);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &matmulOutTensor = kernelGraph_.internalTensors[internalTensorNum++];
    AsdOps::Tensor &addOutTensor = kernelGraph_.internalTensors[internalTensorNum++];

    kernelGraph_.nodes.resize(3);
    int64_t nodeNum = 0;
    auto &matmulNode = kernelGraph_.nodes[nodeNum++];
    auto &addNode = kernelGraph_.nodes[nodeNum++];
    auto &activateNode = kernelGraph_.nodes[nodeNum++];

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

    switch (param_.activationFuncType) {
    case FfnParam::ActivationFuncType::FAST_GELU:
        activateNode.opDesc = {0, "ElewiseOperation",
                               AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
        break;
    case FfnParam::ActivationFuncType::GELU:
        activateNode.opDesc = {0, "ActivationOperation",
                               AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_GELU})};
        break;
    case FfnParam::ActivationFuncType::RELU:
        activateNode.opDesc = {0, "ActivationOperation",
                               AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_RELU})};
        break;
    default:
        activateNode.opDesc = {0, "ElewiseOperation",
                               AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
        break;
    }

    activateNode.inTensors = {&addOutTensor};
    activateNode.outTensors = {&operationOutTensor};
    return AsdOps::Status::OkStatus();
}

AsdOps::Status FfnOpsRunner::SetupKernelGraphWithoutBias(const RunnerVariantPack &runnerVariantPack) {
    kernelGraph_.inTensors.resize(2);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &aTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &bTensor = kernelGraph_.inTensors[inTensorNum++];

    kernelGraph_.outTensors.resize(1);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &operationOutTensor = kernelGraph_.outTensors[outTensorNum++];

    kernelGraph_.internalTensors.resize(1);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &matmulOutTensor = kernelGraph_.internalTensors[internalTensorNum++];

    kernelGraph_.nodes.resize(2);
    int64_t nodeNum = 0;
    auto &matmulNode = kernelGraph_.nodes[nodeNum++];
    auto &activateNode = kernelGraph_.nodes[nodeNum++];

    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB})};
    matmulNode.inTensors = {&aTensor, &bTensor};
    matmulNode.outTensors = {&matmulOutTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    switch (param_.activationFuncType) {
    case FfnParam::ActivationFuncType::FAST_GELU:
        activateNode.opDesc = {0, "ElewiseOperation",
                               AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
        break;
    case FfnParam::ActivationFuncType::GELU:
        activateNode.opDesc = {0, "ActivationOperation",
                               AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_GELU})};
        break;
    case FfnParam::ActivationFuncType::RELU:
        activateNode.opDesc = {0, "ActivationOperation",
                               AsdOps::OpParam::Activation({AsdOps::OpParam::Activation::ACTIVATION_RELU})};
        break;
    default:
        activateNode.opDesc = {0, "ElewiseOperation",
                               AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_FASTGELU})};
        break;
    }

    activateNode.inTensors = {&matmulOutTensor};
    activateNode.outTensors = {&operationOutTensor};
    return AsdOps::Status::OkStatus();
}

AsdOps::Status FfnOpsRunner::SetupKernelGraph(const RunnerVariantPack &runnerVariantPack)
{
    if (param_.hasBias) {
        return SetupKernelGraphWithBias(runnerVariantPack);
    } else {
        return SetupKernelGraphWithoutBias(runnerVariantPack);
    }
}
} // namespace AclTransformer