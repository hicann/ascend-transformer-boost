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
    kernelGraph_.inTensors.resize(10);
    AsdOps::Tensor &hiddenStatus = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &weightGate = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &deqScaleGate = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &biasGate = kernelGraph_.inTensors.at(3);
    AsdOps::Tensor &weightDown = kernelGraph_.inTensors.at(4);
    AsdOps::Tensor &deqScaleDown = kernelGraph_.inTensors.at(5);
    AsdOps::Tensor &biasDown = kernelGraph_.inTensors.at(6);
    AsdOps::Tensor &weightUp = kernelGraph_.inTensors.at(7);
    AsdOps::Tensor &deqScaleUp = kernelGraph_.inTensors.at(8);
    AsdOps::Tensor &biasUp = kernelGraph_.inTensors.at(9);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.internalTensors.resize(5);
    AsdOps::Tensor &matmulGateOut = kernelGraph_.internalTensors.at(0);
    AsdOps::Tensor &matmulUpOut = kernelGraph_.internalTensors.at(1);
    AsdOps::Tensor &mulOut = kernelGraph_.internalTensors.at(2);
    AsdOps::Tensor &swishOut = kernelGraph_.internalTensors.at(3);
    AsdOps::Tensor &quantOut = kernelGraph_.internalTensors.at(4);

    kernelGraph_.nodes.resize(6);
    auto &matmulGateNode = kernelGraph_.nodes[0];
    auto &swishNode = kernelGraph_.nodes[1];
    auto &matmulUpNode = kernelGraph_.nodes[2];
    auto &mulNode = kernelGraph_.nodes[3];
    auto &quantNode = kernelGraph_.nodes[4];
    auto &matmulDownNode = kernelGraph_.nodes[5];

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
