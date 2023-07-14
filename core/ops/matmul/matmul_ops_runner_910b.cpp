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
#include "matmul_ops_runner_910b.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MatmulOpsRunner910B::MatmulOpsRunner910B(const MatmulParam &param)
    : OpsRunner("MatmulOpsRunner910B", RUNNER_TYPE_MATMUL), param_(param)
{
    ASD_LOG(INFO) << "MatmulOpsRunner910B::MatmulOpsRunner910B";
    const std::size_t inTensorSize = 2;
    const std::size_t outTensorSize = 1;
    const std::size_t nodeSize = 1;
    const std::size_t dimSize_2 = 2;
    const std::size_t dimSize_3 = 3;
    kernelGraph_.inTensors.resize(inTensorSize);
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[0];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[1];

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors[0];

    kernelGraph_.nodes.resize(nodeSize);
    auto &matmulNode = kernelGraph_.nodes[0];

    matmulNode.opDesc = { 0, "MatMulOperation", AsdOps::OpParam::MatMul( { param_.transposeA, param_.transposeB } ) };
    matmulNode.inTensors = { &inputTensor, &weightTensor };
    matmulNode.outTensors = { &resultTensor };
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    // matmul可以是2维*2维，或者3维*3维，如果是2维*3维，需要对第一个tensor做下合轴
    matmulNode.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims,
        AsdOps::SVector<int64_t> &newDims) {
        if (oldDims.size() == dimSize_3 && matmulNode.inTensors[1]->desc.dims.size() == dimSize_2) {
            newDims = { oldDims.at(0) * oldDims.at(1), oldDims.at(2) };
        } else {
            newDims = oldDims;
        }
    };
}

MatmulOpsRunner910B::~MatmulOpsRunner910B() {}
} // namespace AclTransformer
