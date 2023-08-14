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
#include "mlp_ops_runner_910a.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MlpOpsRunner910A::MlpOpsRunner910A(const MlpParam &param) : OpsRunner("MlpOpsRunner910A", RUNNER_TYPE_MLP), param_(param)
{
    ASD_LOG(INFO) << "MlpOpsRunner910A::MlpOpsRunner910A called";

    static const uint64_t IN_TENSOR_COUNT = 4;
    static const uint64_t OUT_TENSOR_COUNT = 1;
    static const uint64_t INTERMEDIATE_TENSOR_COUNT = 9;
    static const uint64_t NODE_COUNT = 10;

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &hiddenStatus = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &weightGate = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &weightDown = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &weightUp = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &hiddenStatusNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &matmulGateOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &matmulGateOutND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &swishOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &matmulUpOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &matmulUpOutND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulOutNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &resultTensorNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    
    int64_t nodeNum = 0;
    auto &transdata0Node = kernelGraph_.nodes[nodeNum++];
    auto &matmulGateNode = kernelGraph_.nodes[nodeNum++];
    auto &transdata2Node = kernelGraph_.nodes[nodeNum++];
    auto &swishNode = kernelGraph_.nodes[nodeNum++];
    auto &matmulUpNode = kernelGraph_.nodes[nodeNum++];
    auto &transdata4Node = kernelGraph_.nodes[nodeNum++];
    auto &mulNode = kernelGraph_.nodes[nodeNum++];
    auto &transdata5Node = kernelGraph_.nodes[nodeNum++];
    auto &matmulDownNode = kernelGraph_.nodes[nodeNum++];
    auto &transdata7Node = kernelGraph_.nodes[nodeNum++];

    transdata0Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata0Node.inTensors = {&hiddenStatus};
    transdata0Node.outTensors = {&hiddenStatusNZ};
    transdata0Node.inTensorViewFuncs.resize(transdata0Node.inTensors.size());
    transdata0Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        hiddenstatusdims_ = runInfo.GetInTensor(0).desc.dims;
    };

    matmulGateNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulGateNode.inTensors = {&hiddenStatusNZ, &weightGate};
    matmulGateNode.outTensors = {&matmulGateOut};
    matmulGateNode.inTensorViewFuncs.resize(matmulGateNode.inTensors.size());
    matmulGateNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
            newDims = {1, oldDims.at(1)/16, oldDims.at(0), 16};
        };
    matmulGateNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        weightgatedims_ = runInfo.GetInTensor(1).desc.dims;
        runInfo.SetOpDesc({0, "MatmulOperation",
                            AsdOps::OpParam::MatMul({false, true, 
                                {hiddenstatusdims_.at(1), hiddenstatusdims_.at(2), weightgatedims_.at(2)}})});
    };

    transdata2Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata2Node.inTensors = {&matmulGateOut};
    transdata2Node.outTensors = {&matmulGateOutND};
    transdata2Node.inTensorViewFuncs.resize(transdata2Node.inTensors.size());
    transdata2Node.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND,
             {hiddenstatusdims_.at(1), weightgatedims_.at(2)}})});
    };

    swishNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_SWISH})};
    swishNode.inTensors = {&matmulGateOutND};
    swishNode.outTensors = {&swishOut};
    swishNode.inTensorViewFuncs.resize(swishNode.inTensors.size());

    matmulUpNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulUpNode.inTensors = {&hiddenStatusNZ, &weightUp};
    matmulUpNode.outTensors = {&matmulUpOut};
    matmulUpNode.inTensorViewFuncs.resize(matmulUpNode.inTensors.size());
    matmulUpNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
            newDims = {1, oldDims.at(1)/16, oldDims.at(0), 16};
        };
    matmulUpNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        weightupdims_ = runInfo.GetInTensor(1).desc.dims;
        runInfo.SetOpDesc({0, "MatmulOperation",
                            AsdOps::OpParam::MatMul({false, true, 
                                {hiddenstatusdims_.at(1), hiddenstatusdims_.at(2), weightupdims_.at(2)}})});
    };

    transdata4Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata4Node.inTensors = {&matmulUpOut};
    transdata4Node.outTensors = {&matmulUpOutND};
    transdata4Node.inTensorViewFuncs.resize(transdata4Node.inTensors.size());
    transdata4Node.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND,
             {hiddenstatusdims_.at(1), weightupdims_.at(2)}})});
    };

    mulNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mulNode.inTensors = {&swishOut, &matmulUpOutND};
    mulNode.outTensors = {&mulOut};
    mulNode.inTensorViewFuncs.resize(mulNode.inTensors.size());

    transdata5Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata5Node.inTensors = {&mulOut};
    transdata5Node.outTensors = {&mulOutNZ};
    transdata5Node.inTensorViewFuncs.resize(transdata5Node.inTensors.size());
    transdata5Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        muloutdims_ = runInfo.GetInTensor(0).desc.dims;
    };

    matmulDownNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulDownNode.inTensors = {&mulOutNZ, &weightDown};
    matmulDownNode.outTensors = {&resultTensorNZ};
    matmulDownNode.inTensorViewFuncs.resize(matmulDownNode.inTensors.size());
    matmulDownNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
            newDims = {1, oldDims.at(1)/16, oldDims.at(0), 16};
        };
    matmulDownNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        weightdowndims_ = runInfo.GetInTensor(1).desc.dims;
        runInfo.SetOpDesc({0, "MatmulOperation",
                            AsdOps::OpParam::MatMul({false, true, 
                                {muloutdims_.at(1), muloutdims_.at(2), weightdowndims_.at(2)}})});
    };

    transdata7Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata7Node.inTensors = {&resultTensorNZ}; 
    transdata7Node.outTensors = {&resultTensor}; 
    transdata7Node.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND,
             {muloutdims_.at(1),  weightdowndims_.at(2)}})});
    };
}

MlpOpsRunner910A::~MlpOpsRunner910A() {}
} // namespace AclTransformer