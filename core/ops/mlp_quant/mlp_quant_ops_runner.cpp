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
#include "mlp_quant_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MlpQuantOpsRunner::MlpQuantOpsRunner(const MlpQuantParam &param) : OpsRunner("MlpQuantOpsRunner", RUNNER_TYPE_MLP_QUANT), param_(param)
{
    ASD_LOG(INFO) << "MlpQuantOpsRunner::MlpQuantOpsRunner called";
    const std::size_t inTensorSize = 10;
    const std::size_t outTensorSize = 1;
    const std::size_t internalTensorSize = 5;
    const std::size_t nodeSize = 6;

    kernelGraph_.inTensors.resize(inTensorSize);
    size_t inTensorId = 0;
    AsdOps::Tensor &hiddenStatus = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightGate = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &deqScaleGate = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &biasGate = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightDown = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &deqScaleDown = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &biasDown = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightUp = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &deqScaleUp = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &biasUp = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(outTensorSize);
    size_t outTensorId = 0;
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.internalTensors.resize(internalTensorSize);
    size_t internalTensorId = 0;
    AsdOps::Tensor &matmulGateOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &matmulUpOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &mulOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &swishOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &quantOut = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(nodeSize);
    size_t nodeId = 0;
    auto &matmulGateNode = kernelGraph_.nodes[nodeId++];
    auto &swishNode = kernelGraph_.nodes[nodeId++];
    auto &matmulUpNode = kernelGraph_.nodes[nodeId++];
    auto &mulNode = kernelGraph_.nodes[nodeId++];
    auto &quantNode = kernelGraph_.nodes[nodeId++];
    auto &matmulDownNode = kernelGraph_.nodes[nodeId++];

    matmulGateNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulGateNode.inTensors = {&hiddenStatus, &weightGate, &biasGate, &deqScaleGate};
    matmulGateNode.outTensors = {&matmulGateOut};
    matmulGateNode.inTensorViewFuncs.resize(matmulGateNode.inTensors.size());
    matmulGateNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims,
                                              AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    swishNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_SWISH})};
    swishNode.inTensors = {&matmulGateOut};
    swishNode.outTensors = {&swishOut};

    matmulUpNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulUpNode.inTensors = {&hiddenStatus, &weightUp, &biasUp, &deqScaleUp};
    matmulUpNode.outTensors = {&matmulUpOut};
    matmulUpNode.inTensorViewFuncs.resize(matmulUpNode.inTensors.size());
    matmulUpNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    mulNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mulNode.inTensors = {&swishOut, &matmulUpOut};
    mulNode.outTensors = {&mulOut};

    quantNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise(
                            {AsdOps::OpParam::Elewise::ELEWISE_QUANT, 0, 0, param_.inputScale, param_.inputOffset})};
    quantNode.inTensors = {&mulOut};
    quantNode.outTensors = {&quantOut};

    matmulDownNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulDownNode.inTensors = {&quantOut, &weightDown, &biasDown, &deqScaleDown};
    matmulDownNode.outTensors = {&resultTensor};
}

MlpQuantOpsRunner::~MlpQuantOpsRunner() {}
} // namespace AclTransformer
