
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
#include "mlp_ops_glm2_6b_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MlpOpsGlm2Runner::MlpOpsGlm2Runner(const MlpParam &param)
    : OpsRunner("MlpOpsGlm2Runner", RUNNER_TYPE_MLP), param_(param)
{
    ASD_LOG(INFO) << "MlpOpsGlm2Runner::MlpOpsGlm2Runner called";
    const std::size_t inTensorSize = 3;
    const std::size_t outTensorSize = 1;
    const std::size_t internalTensorSize = 5;
    const std::size_t nodeSize = 5;

    kernelGraph_.inTensors.resize(inTensorSize);
    size_t inTensorId = 0;
    AsdOps::Tensor &hiddenStates = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightUp = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightDown = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.internalTensors.resize(internalTensorSize);
    size_t internalTensorId = 0;
    AsdOps::Tensor &matmulUpOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &mulOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &swishOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &chunk0 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &chunk1 = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(nodeSize);
    size_t nodeId = 0;
    auto &matmulUpNode = kernelGraph_.nodes[nodeId++];
    auto &splitNode = kernelGraph_.nodes[nodeId++];
    auto &swishNode = kernelGraph_.nodes[nodeId++];
    auto &mulNode = kernelGraph_.nodes[nodeId++];
    auto &matmulDownNode = kernelGraph_.nodes[nodeId++];

    matmulUpNode.opDesc = { 0, "MatMulOperation", AsdOps::OpParam::MatMul({ false, true }) };
    matmulUpNode.inTensors = { &hiddenStates, &weightUp };
    matmulUpNode.outTensors = { &matmulUpOut };
    matmulUpNode.inTensorViewFuncs.resize(matmulUpNode.inTensors.size());
    matmulUpNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    splitNode.opDesc = { 0, "SplitOperation", AsdOps::OpParam::Split { 0, 2 } }; // splitDim,splitNum
    splitNode.inTensors = { &matmulUpOut };
    splitNode.outTensors = { &chunk0, &chunk1 };
    splitNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({ 0, "SplitOperation", AsdOps::OpParam::Split { int(dims.size()) - 1, 2 } });
    };

    swishNode.opDesc = { 0, "ElewiseOperation", AsdOps::OpParam::Elewise({
        AsdOps::OpParam::Elewise::ELEWISE_SWISH }) };
    swishNode.inTensors = { &chunk0 };
    swishNode.outTensors = { &swishOut };

    mulNode.opDesc = { 0, "BroadcastOperation", AsdOps::OpParam::Broadcast({
        AsdOps::OpParam::Broadcast::BROADCAST_MUL }) };
    mulNode.inTensors = { &swishOut, &chunk1 };
    mulNode.outTensors = { &mulOut };

    matmulDownNode.opDesc = { 0, "MatMulOperation", AsdOps::OpParam::MatMul({ false, true }) };
    matmulDownNode.inTensors = { &mulOut, &weightDown };
    matmulDownNode.outTensors = { &resultTensor };
    ASD_LOG(INFO) << "MlpOpsGlm2Runner::MlpOpsGlm2Runner END";
}

MlpOpsGlm2Runner::~MlpOpsGlm2Runner() {}
} // namespace AclTransformer
