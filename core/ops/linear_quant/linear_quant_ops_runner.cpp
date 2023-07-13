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
#include "linear_quant_ops_runner.h"
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
LinearQuantOpsRunner::LinearQuantOpsRunner(LinearQuantParam &param)
    : OpsRunner("LinearQuantOpsRunner", RUNNER_TYPE_LINEAR_QUANT), param_(param)
{
    ASD_LOG(INFO) << "LinearQuantOpsRunner::LinearQuantOpsRunner";
    const std::size_t nodeSize = 1;
    const std::size_t dim2 = 2;
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[0];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[1];
    AsdOps::Tensor &biasTensor = kernelGraph_.inTensors[2];
    AsdOps::Tensor &descaleTensor = kernelGraph_.inTensors[3];

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &resultTensor = kernelGraph_.outTensors[0];

    kernelGraph_.nodes.resize(nodeSize);
    auto &matmulNode = kernelGraph_.nodes[0];

    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({param_.transposeA, !param_.transposeB})};
    matmulNode.inTensors = {&inputTensor, &weightTensor, &biasTensor, &descaleTensor};
    matmulNode.outTensors = {&resultTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(dim2)};
    };
}

LinearQuantOpsRunner::~LinearQuantOpsRunner() {}
} // namespace AclTransformer
