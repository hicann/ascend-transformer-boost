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
#include "mlp_ops_glm2_6b_runner_310p.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MlpOpsGlm2Runner310P::MlpOpsGlm2Runner310P(const MlpParam &param)
    : OpsRunner("MlpOpsGlm2Runner310P", RUNNER_TYPE_MLP), param_(param)
{
    ASD_LOG(INFO) << "MlpOpsGlm2Runner310P::MlpOpsGlm2Runner310P called";
    const std::size_t inTensorSize = 3;
    const std::size_t outTensorSize = 1;
    const std::size_t internalTensorSize = 9;
    const std::size_t nodeSize = 9;
    const std::size_t dimSize = 4;

    kernelGraph_.inTensors.resize(inTensorSize);
    size_t inTensorId = 0;
    AsdOps::Tensor &hiddenStates = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightUp = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightDown = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.internalTensors.resize(internalTensorSize);
    size_t internalTensorId = 0;
    AsdOps::Tensor &transdataA1Out = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &matmulUpOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &transdataC1Out = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &mulOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &swishOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &chunk0 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &chunk1 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &transdataA2Out = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &matmulDownOut = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(nodeSize);
    size_t nodeId = 0;
    auto &transdataA1Node = kernelGraph_.nodes[nodeId++];
    auto &matmulUpNode = kernelGraph_.nodes[nodeId++];
    auto &transdataC1Node = kernelGraph_.nodes[nodeId++];
    auto &splitNode = kernelGraph_.nodes[nodeId++];
    auto &swishNode = kernelGraph_.nodes[nodeId++];
    auto &mulNode = kernelGraph_.nodes[nodeId++];
    auto &transdataA2Node = kernelGraph_.nodes[nodeId++];
    auto &matmulDownNode = kernelGraph_.nodes[nodeId++];
    auto &transdataC2Node = kernelGraph_.nodes[nodeId++];

    ViewFunc Squeeze1 = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        oriADims_ = oldDims;
        if (oldDims.size() == 2) {
            oriSize_ = 2;
            newDims = {1, oldDims.at(0), oldDims.at(1)};
        } else {
            newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
        }
    };

    transdataA1Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataA1Node.inTensors = {&hiddenStates};
    transdataA1Node.outTensors = {&transdataA1Out};
    transdataA1Node.inTensorViewFuncs.resize(transdataA1Node.inTensors.size());
    transdataA1Node.inTensorViewFuncs.at(0) = Squeeze1;

    ViewFunc CheckDimB = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        int64_t bLastDim = oldDims.at(oldDims.size() - 1);
        const int64_t blockDim = 16;
        if (bLastDim != blockDim || oldDims.size() < dimSize) {
            newDims = {1, bLastDim / blockDim, oldDims.at(0), blockDim};
        }
        oriBDims_ = {newDims.at(2), newDims.at(1) * newDims.at(3)};
    };

    matmulUpNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulUpNode.inTensors = {&transdataA1Out, &weightUp};
    matmulUpNode.outTensors = {&matmulUpOut};
    matmulUpNode.inTensorViewFuncs.resize(matmulUpNode.inTensors.size());
    matmulUpNode.inTensorViewFuncs.at(1) = CheckDimB;
    matmulUpNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1, dim2;
        if (oriSize_ == 3) {
            dim0 = oriADims_.at(0) * oriADims_.at(1);
            dim1 = oriADims_.at(2);
        } else {
            dim0 = oriADims_.at(0);
            dim1 = oriADims_.at(1);
        }
        dim2 = oriBDims_.at(0);
        runInfo.SetOpDesc({0, "MatMulOperation",
                           AsdOps::OpParam::MatMul({false, true, {dim0, dim1, dim2}})});
    };

    transdataC1Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataC1Node.inTensors = {&matmulUpOut};
    transdataC1Node.outTensors = {&transdataC1Out};
    transdataC1Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1;
        if (oriSize_ == 3) {
            dim0 = oriADims_.at(0) * oriADims_.at(1);
        } else {
            dim0 = oriADims_.at(0);
        }
        dim1 = oriBDims_.at(0);
        runInfo.SetOpDesc({0, "TransdataOperation",
                           AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {dim0, dim1}})});
    };

    splitNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split {0, 2}}; 
    splitNode.inTensors = {&transdataC1Out};
    splitNode.outTensors = {&chunk0, &chunk1};
    splitNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({ 0, "SplitOperation", AsdOps::OpParam::Split {int(dims.size()) - 1, 2}});
    };

    swishNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({
        AsdOps::OpParam::Elewise::ELEWISE_SWISH })};
    swishNode.inTensors = {&chunk0};
    swishNode.outTensors = {&swishOut};

    mulNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({
        AsdOps::OpParam::Broadcast::BROADCAST_MUL })};
    mulNode.inTensors = {&swishOut, &chunk1};
    mulNode.outTensors = {&mulOut};

    transdataA2Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataA2Node.inTensors = {&mulOut};
    transdataA2Node.outTensors = {&transdataA2Out};
    transdataA2Node.inTensorViewFuncs.resize(transdataA2Node.inTensors.size());
    transdataA2Node.inTensorViewFuncs.at(0) = Squeeze1;

    matmulDownNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulDownNode.inTensors = {&transdataA2Out, &weightDown};
    matmulDownNode.outTensors = {&matmulDownOut};
    matmulDownNode.inTensorViewFuncs.resize(matmulDownNode.inTensors.size());
    matmulDownNode.inTensorViewFuncs.at(1) = CheckDimB;
    matmulDownNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1, dim2;
        if (oriSize_ == 3) {
            dim0 = oriADims_.at(0) * oriADims_.at(1);
            dim1 = oriADims_.at(2);
        } else {
            dim0 = oriADims_.at(0);
            dim1 = oriADims_.at(1);
        }
        dim2 = oriBDims_.at(0);
        runInfo.SetOpDesc({0, "MatMulOperation",
                           AsdOps::OpParam::MatMul({false, true, {dim0, dim1, dim2}})});
    };

    transdataC2Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataC2Node.inTensors = {&matmulDownOut};
    transdataC2Node.outTensors = {&resultTensor};
    transdataC2Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1;
        if (oriSize_ == 3) {
            dim0 = oriADims_.at(0) * oriADims_.at(1);
        } else {
            dim0 = oriADims_.at(0);
        }
        dim1 = oriBDims_.at(0);
        runInfo.SetOpDesc({0, "TransdataOperation",
                           AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {dim0, dim1}})});
    };


    ASD_LOG(INFO) << "MlpOpsGlm2Runner310P::MlpOpsGlm2Runner310P END";
}

MlpOpsGlm2Runner310P::~MlpOpsGlm2Runner310P() {}
} // namespace AclTransformer
