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
#include "mlp_ops_llama13b_runner_910a.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MlpOpsLlama13bRunner910A::MlpOpsLlama13bRunner910A(const MlpParam &param) : OpsRunner("MlpOpsLlama13bRunner910A", RUNNER_TYPE_MLP), param_(param)
{
    ASD_LOG(INFO) << "MlpOpsLlama13bRunner910A::MlpOpsLlama13bRunner910A called";

    const std::size_t inTensorSize = 3;
    const std::size_t outTensorSize = 1;
    const std::size_t internalTensorSize = 6;
    const std::size_t nodeSize = 7;

    kernelGraph_.inTensors.resize(inTensorSize);
    size_t inTensorId = 0;
    AsdOps::Tensor &hiddenStatus = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightGate = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &weightUp = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.internalTensors.resize(internalTensorSize);
    size_t internalTensorId = 0;
    AsdOps::Tensor &hiddenStatusNZ = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &matmulGateOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &matmulGateOutND = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &swishOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &matmulUpOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &matmulUpOutND = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(nodeSize);
    size_t nodeId = 0;
    auto &transdata0Node = kernelGraph_.nodes[nodeId++];
    auto &matmulGateNode = kernelGraph_.nodes[nodeId++];
    auto &transdata1Node = kernelGraph_.nodes[nodeId++];
    auto &swishNode = kernelGraph_.nodes[nodeId++];
    auto &matmulUpNode = kernelGraph_.nodes[nodeId++];
    auto &transdata2Node = kernelGraph_.nodes[nodeId++];
    auto &mulNode = kernelGraph_.nodes[nodeId++];

    ViewFunc Squeeze1 = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        hiddenStatusDims_ = oldDims;
        if (oldDims.size() == 2) {
            oriSize_ = 2;
            newDims = {1, oldDims.at(0), oldDims.at(1)};
        } else {
            newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
        }
    };

    transdata0Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata0Node.inTensors = {&hiddenStatus};
    transdata0Node.outTensors = {&hiddenStatusNZ};
    transdata0Node.inTensorViewFuncs.resize(transdata0Node.inTensors.size());
    transdata0Node.inTensorViewFuncs.at(0) = Squeeze1;

    matmulGateNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulGateNode.inTensors = {&hiddenStatusNZ, &weightGate};
    matmulGateNode.outTensors = {&matmulGateOut};
    matmulGateNode.inTensorViewFuncs.resize(matmulGateNode.inTensors.size());
    matmulGateNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
            weightGatedims_ = oldDims;
            newDims = {1, oldDims.at(1)/16, oldDims.at(0), 16};
        };
    matmulGateNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1, dim2;
        if (oriSize_ == 3) {
            dim0 = hiddenStatusDims_.at(0) * hiddenStatusDims_.at(1);
            dim1 = hiddenStatusDims_.at(2);
        } else {
            dim0 = hiddenStatusDims_.at(0);
            dim1 = hiddenStatusDims_.at(1);
        }
        runInfo.SetOpDesc({0, "MatmulOperation",
                            AsdOps::OpParam::MatMul({false, true, 
                                {dim0, dim1, weightGatedims_.at(0)}})});
    };

    transdata1Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata1Node.inTensors = {&matmulGateOut};
    transdata1Node.outTensors = {&matmulGateOutND};
    transdata1Node.inTensorViewFuncs.resize(transdata1Node.inTensors.size());
    transdata1Node.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1;
        if (oriSize_ == 3) {
            dim0 = hiddenStatusDims_.at(0) * hiddenStatusDims_.at(1);
        } else {
            dim0 = hiddenStatusDims_.at(0);
        }
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND,
             {dim0, weightGatedims_.at(0)}})});
    };

    swishNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_SWISH})};
    swishNode.inTensors = {&matmulGateOutND};
    swishNode.outTensors = {&swishOut};

    matmulUpNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulUpNode.inTensors = {&hiddenStatusNZ, &weightUp};
    matmulUpNode.outTensors = {&matmulUpOut};
    matmulUpNode.inTensorViewFuncs.resize(matmulUpNode.inTensors.size());
    matmulUpNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
            weightUpdims_ = oldDims;
            newDims = {1, oldDims.at(1)/16, oldDims.at(0), 16};
        };
    matmulUpNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1, dim2;
        if (oriSize_ == 3) {
            dim0 = hiddenStatusDims_.at(0) * hiddenStatusDims_.at(1);
            dim1 = hiddenStatusDims_.at(2);
        } else {
            dim0 = hiddenStatusDims_.at(0);
            dim1 = hiddenStatusDims_.at(1);
        }
        runInfo.SetOpDesc({0, "MatmulOperation",
                            AsdOps::OpParam::MatMul({false, true, 
                                {dim0, dim1, weightUpdims_.at(0)}})});
    };

    transdata2Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata2Node.inTensors = {&matmulUpOut};
    transdata2Node.outTensors = {&matmulUpOutND};
    transdata2Node.inTensorViewFuncs.resize(transdata2Node.inTensors.size());
    transdata2Node.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        int64_t dim0, dim1;
        if (oriSize_ == 3) {
            dim0 = hiddenStatusDims_.at(0) * hiddenStatusDims_.at(1);
        } else {
            dim0 = hiddenStatusDims_.at(0);
        }
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND,
             {dim0, weightUpdims_.at(0)}})});
    };

    mulNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mulNode.inTensors = {&swishOut, &matmulUpOutND};
    mulNode.outTensors = {&resultTensor};
}

MlpOpsLlama13bRunner910A::~MlpOpsLlama13bRunner910A() {}
} // namespace AclTransformer