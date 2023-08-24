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
#include "position_embedding_1d_fusion_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/rope.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbedding1dMixedFusionOpsRunner::PositionEmbedding1dMixedFusionOpsRunner(const PositionEmbeddingParam &param)
    : OpsRunner("PositionEmbedding1dMixedFusionOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_1D_MIXED_FUSION), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbedding1dMixedFusionOpsRunner::PositionEmbedding1dMixedFusionOpsRunner called, headNum: " << param_.headNum;
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &cos = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &sin = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors.resize(3);
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(2);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(2);
    int64_t nodeNum = 0;
    auto &splitQkvNode = kernelGraph_.nodes[nodeNum++];
    auto &ropeNode = kernelGraph_.nodes[nodeNum++];

    // split QKV
    splitQkvNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 3}};
    splitQkvNode.inTensors = {&mixedQkv};
    splitQkvNode.outTensors = {&qLayer, &kLayer, &value};
    splitQkvNode.inTensorViewFuncs.resize(splitQkvNode.inTensors.size());
    splitQkvNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                             AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };
    splitQkvNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    };

    // QK Rope
    ViewFunc ropeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims.resize(2);
        newDims.at(0) = oldDims.at(0) * oldDims.at(1);
        newDims.at(1) = oldDims.at(2); 
    }; 

    ViewFunc ropeKqView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    const size_t ropeViewSize = 4;
    ropeNode.opDesc = {0, "RopeOperation", AsdOps::OpParam::Rope{AsdOps::OpParam::Rope::ROPE}};
    ropeNode.inTensors = {&qLayer, &kLayer, &cos, &sin, &seqLen};
    ropeNode.outTensors = {&qEmbedded, &kEmbedded};
    ropeNode.inTensorViewFuncs.resize(ropeViewSize);
    ropeNode.inTensorViewFuncs.at(0) = ropeKqView;
    ropeNode.inTensorViewFuncs.at(1) = ropeKqView;
    ropeNode.inTensorViewFuncs.at(2) = ropeCosSinView;
    ropeNode.inTensorViewFuncs.at(3) = ropeCosSinView;
    ropeNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        runInfo.GetInTensor(4).desc.dtype = AsdOps::TENSOR_DTYPE_UINT32;
    };
}

PositionEmbedding1dMixedFusionOpsRunner::~PositionEmbedding1dMixedFusionOpsRunner() {}
} // namespace AclTransformer
