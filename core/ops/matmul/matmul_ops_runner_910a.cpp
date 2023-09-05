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
#include "matmul_ops_runner_910a.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;
namespace AclTransformer {
MatmulOpsRunner910A::MatmulOpsRunner910A(const MatmulParam &param)
    : OpsRunner("MatmulOpsRunner910A", RUNNER_TYPE_MATMUL), param_(param)
{
    ASD_LOG(INFO) << "MatmulOpsRunner910A::MatmulOpsRunner910A called";

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &inputTensorNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &resultTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    
    int64_t outTensorNum = 0;
    AsdOps::Tensor &resultTensorND = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t nodeNum = 0;
    auto &transdata0Node = kernelGraph_.nodes[nodeNum++];
    auto &matmulNode = kernelGraph_.nodes[nodeNum++];
    auto &transdata1Node = kernelGraph_.nodes[nodeNum++];

    ViewFunc Unsqueeze0 = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        if (oldDims.size() == 2) {
            newDims.resize(oldDims.size() + 1);
            newDims.at(0) = 1;
            for (size_t i = 1; i < newDims.size(); i++) {
                newDims.at(i) = oldDims.at(i - 1);
            }
        } else {
            for (size_t i = 1; i < newDims.size(); i++) {
                newDims.at(i) = oldDims.at(i);
            }
        }
    };
    
    transdata0Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdata0Node.inTensors = {&inputTensor};
    transdata0Node.outTensors = {&inputTensorNZ};
    transdata0Node.inTensorViewFuncs.resize(transdata0Node.inTensors.size());
    transdata0Node.inTensorViewFuncs.at(0) = Unsqueeze0;
    transdata0Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        inputTensorDims_ = runInfo.GetInTensor(0).desc.dims;
    };

    matmulNode.opDesc = { 0, "MatMulOperation", AsdOps::OpParam::MatMul( { param_.transposeA, param_.transposeB } ) };
    matmulNode.inTensors = {&inputTensorNZ, &weightTensor};
    matmulNode.outTensors = {&resultTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(1) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
            newDims = {1, oldDims.at(1)/16, oldDims.at(0), 16};
        };
    matmulNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        weightdims_ = runInfo.GetInTensor(1).desc.dims;
        runInfo.SetOpDesc({0, "MatmulOperation",
                            AsdOps::OpParam::MatMul({param_.transposeA, param_.transposeB, 
                                {inputTensorDims_.at(1), inputTensorDims_.at(2), weightdims_.at(2)}})});
    };

    transdata1Node.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdata1Node.inTensors = {&resultTensor};
    transdata1Node.outTensors = {&resultTensorND};
    transdata1Node.inTensorViewFuncs.resize(transdata1Node.inTensors.size());
    transdata1Node.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND,
             {inputTensorDims_.at(1), weightdims_.at(2)}})});
    };
}

MatmulOpsRunner910A::~MatmulOpsRunner910A() {}
} // namespace AclTransformer
