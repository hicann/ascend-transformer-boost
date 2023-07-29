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
#include "linear_ops_runner_910b.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
LinearOpsRunner910B::LinearOpsRunner910B(LinearParam &param)
    : OpsRunner("LinearOpsRunner910B", RUNNER_TYPE_LINEAR), param_(param)
{
    ASD_LOG(INFO) << "LinearOpsRunner910B::LinearOpsRunner910B";
}

LinearOpsRunner910B::~LinearOpsRunner910B() {}

AsdOps::Status LinearOpsRunner910B::SetupKernelGraphWithBias(const RunnerVariantPack &runnerVariantPack) {
    const std::size_t nodeSize = 2;
    const std::size_t dim2 = 2;

    kernelGraph_.inTensors.resize(3);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[inTensorNum++];

    kernelGraph_.outTensors.resize(1);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors[outTensorNum++];

    kernelGraph_.internalTensors.resize(1);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &matmulResultTensor = kernelGraph_.internalTensors[internalTensorNum++];

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &matmulNode = kernelGraph_.nodes[nodeNum++];
    auto &addNode = kernelGraph_.nodes[nodeNum++];

    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB})};
    matmulNode.inTensors = {&inputTensor, &weightTensor};
    matmulNode.outTensors = {&matmulResultTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(dim2)};
    };

    addNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addNode.inTensors = {&matmulResultTensor, &biasTensor};
    addNode.outTensors = {&resultTensor};
    return AsdOps::Status::OkStatus();
}

AsdOps::Status LinearOpsRunner910B::SetupKernelGraphWithoutBias(const RunnerVariantPack &runnerVariantPack) {
    const std::size_t nodeSize = 1;
    const std::size_t dim2 = 2;

    kernelGraph_.inTensors.resize(2);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[inTensorNum++];

    kernelGraph_.outTensors.resize(1);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &matmulResultTensor = kernelGraph_.outTensors[outTensorNum++];

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &matmulNode = kernelGraph_.nodes[nodeNum++];

    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB})};
    matmulNode.inTensors = {&inputTensor, &weightTensor};
    matmulNode.outTensors = {&matmulResultTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(dim2)};
    };
    return AsdOps::Status::OkStatus();
}

AsdOps::Status LinearOpsRunner910B::SetupKernelGraph(const RunnerVariantPack &runnerVariantPack)
{
    if (param_.hasBias) {
        return SetupKernelGraphWithBias(runnerVariantPack);
    } else {
        return SetupKernelGraphWithoutBias(runnerVariantPack);
    }
}
} // namespace AclTransformer