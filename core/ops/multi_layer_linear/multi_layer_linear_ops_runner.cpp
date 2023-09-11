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
#include "multi_layer_linear_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
MultiLayerLinearOpsRunner::MultiLayerLinearOpsRunner(const MultiLayerLinearParam &param) : OpsRunner("MultiLayerLinearOpsRunner", RUNNER_TYPE_MULTI_LAYER_LINEAR), param_(param)
{
    ASD_LOG(INFO) << "MultiLayerLinearOpsRunner::MultiLayerLinearOpsRunner called";

    const std::size_t nodeSize = 2;

    kernelGraph_.inTensors.resize(2);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &inputTensor = kernelGraph_.inTensors[inTensorNum++];
    AsdOps::Tensor &weightTensor = kernelGraph_.inTensors[inTensorNum++];

    kernelGraph_.internalTensors.resize(1);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &LinearQKVResult = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.outTensors.resize(3);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &matmulResultQTensor = kernelGraph_.outTensors[outTensorNum++];
    AsdOps::Tensor &matmulResultKTensor = kernelGraph_.outTensors[outTensorNum++];
    AsdOps::Tensor &matmulResultVTensor = kernelGraph_.outTensors[outTensorNum++];

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &matmulNode = kernelGraph_.nodes[nodeNum++];
    auto &splitNode = kernelGraph_.nodes[nodeNum++];

    matmulNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, true})};
    matmulNode.inTensors = {&inputTensor, &weightTensor};
    matmulNode.outTensors = {&LinearQKVResult};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    matmulNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };

    splitNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split {2, 3}}; 
    splitNode.inTensors = {&LinearQKVResult};
    splitNode.outTensors = {&matmulResultQTensor, &matmulResultKTensor, &matmulResultVTensor};
    splitNode.inTensorViewFuncs.resize(splitNode.inTensors.size());
    splitNode.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims,
                                            AsdOps::SVector<int64_t> &newDims) {
        newDims = {1, oldDims.at(0), oldDims.at(1)};
    };
}

MultiLayerLinearOpsRunner::~MultiLayerLinearOpsRunner() {}
} // namespace AclTransformer
