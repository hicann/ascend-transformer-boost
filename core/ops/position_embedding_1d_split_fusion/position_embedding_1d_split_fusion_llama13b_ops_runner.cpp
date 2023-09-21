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
#include "position_embedding_1d_split_fusion_llama13b_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/rope.h>
#include "acltransformer/utils/tensor_util.h"

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 2;
namespace AclTransformer {
PositionEmbedding1dFusionLlama13bOpsRunner::PositionEmbedding1dFusionLlama13bOpsRunner(const PositionEmbedding1dFusionParam &param)
    : OpsRunner("PositionEmbedding1dFusionLlama13bOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_1D_FUSION), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbedding1dFusionLlama13bOpsRunner::PositionEmbedding1dFusionLlama13bOpsRunner called";

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &mixedQKV = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &cosEmbed = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &sinEmbed = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &mixedV = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &mixedQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mixedK = kernelGraph_.internalTensors.at(internalTensorNum++);

    int64_t nodeNum = 0;
    auto &splitQKVNode = kernelGraph_.nodes[nodeNum++];
    auto &ropeNode = kernelGraph_.nodes[nodeNum++];

    // split QKV
    splitQKVNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 3}};
    splitQKVNode.inTensors = {&mixedQKV};
    splitQKVNode.outTensors = {&mixedQ, &mixedK, &mixedV};
    splitQKVNode.inTensorViewFuncs.resize(splitQKVNode.inTensors.size());
    splitQKVNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    };

    ViewFunc ropeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims.resize(2);
        newDims.at(0) = oldDims.at(0) * oldDims.at(1);
        newDims.at(1) = oldDims.at(2); 
    }; 

    ropeNode.opDesc = {0, "RopeOperation", AsdOps::OpParam::Rope({AsdOps::OpParam::Rope::ROPE, 2})};
    ropeNode.inTensors = {&mixedQ, &mixedK, &cosEmbed, &sinEmbed, &seqLen};
    ropeNode.outTensors = {&qEmbedded, &kEmbedded};
    ropeNode.inTensorViewFuncs.resize(ropeNode.inTensors.size());
    ropeNode.inTensorViewFuncs.at(2) = ropeCosSinView;
    ropeNode.inTensorViewFuncs.at(3) = ropeCosSinView;
    ropeNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        runInfo.GetInTensor(4).desc.dtype = AsdOps::TENSOR_DTYPE_UINT32;
    };
}

PositionEmbedding1dFusionLlama13bOpsRunner::~PositionEmbedding1dFusionLlama13bOpsRunner() {}
} // namespace AclTransformer
