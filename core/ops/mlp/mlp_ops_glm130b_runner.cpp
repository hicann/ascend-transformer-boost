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
#include "mlp_ops_glm130b_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MlpOpsGlm130bRunner::MlpOpsGlm130bRunner(const MlpParam &param)
    : OpsRunner("MlpOpsGlm130bRunner", RUNNER_TYPE_MLP), param_(param)
{
    ASD_LOG(INFO) << "MlpOpsGlm130bRunner::MlpOpsGlm130bRunner called";
    const std::size_t inTensorSize = 3;
    const std::size_t outTensorSize = 1;
    const std::size_t internalTensorSize = 5;
    const std::size_t nodeSize = 5;

    kernelGraph_.inTensors.resize(inTensorSize);
    size_t inTensorId = 0;
    AsdOps::Tensor &hiddenStates = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weight = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &bias = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.internalTensors.resize(internalTensorSize);
    size_t internalTensorId = 0;
    AsdOps::Tensor &matmulOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &addOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &chunkOutA = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &chunkOutB = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &geluOut = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(nodeSize);
    size_t nodeId = 0;
    auto &matmulNode = kernelGraph_.nodes[nodeId++];
    auto &addNode = kernelGraph_.nodes[nodeId++];
    auto &chunkNode = kernelGraph_.nodes[nodeId++];
    auto &geluNode = kernelGraph_.nodes[nodeId++];
    auto &mulNode = kernelGraph_.nodes[nodeId++];

    matmulNode.opDesc = { 0, "MatMulOperation", AsdOps::OpParam::MatMul({ false, true }) };
    matmulNode.inTensors = { &hiddenStates, &weight };
    matmulNode.outTensors = { &matmulOut };
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    addNode.opDesc = { 0, "BroadcastOperation", AsdOps::OpParam::Broadcast({
        AsdOps::OpParam::Broadcast::BROADCAST_ADD }) };
    addNode.inTensors = { &matmulOut, &bias };
    addNode.outTensors = { &addOut };

    chunkNode.opDesc = { 0, "SplitOperation", AsdOps::OpParam::Split { 1, 2 } }; // splitDim,splitNum
    chunkNode.inTensors = { &addOut };
    chunkNode.outTensors = { &chunkOutA, &chunkOutB };
    chunkNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({ 0, "SplitOperation", AsdOps::OpParam::Split { int(dims.size()) - 1, 2 } });
    };

    geluNode.opDesc = { 0, "ElewiseOperation", AsdOps::OpParam::Elewise({
        AsdOps::OpParam::Elewise::ELEWISE_FASTGELU }) };
    geluNode.inTensors = { &chunkOutB };
    geluNode.outTensors = { &geluOut };

    mulNode.opDesc = { 0, "BroadcastOperation", AsdOps::OpParam::Broadcast({
        AsdOps::OpParam::Broadcast::BROADCAST_MUL }) };
    mulNode.inTensors = { &chunkOutA, &geluOut };
    mulNode.outTensors = { &resultTensor };
}

MlpOpsGlm130bRunner::~MlpOpsGlm130bRunner() {}
} // namespace AclTransformer
