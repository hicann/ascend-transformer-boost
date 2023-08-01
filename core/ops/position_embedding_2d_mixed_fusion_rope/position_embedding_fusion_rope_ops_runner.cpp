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
#include "position_embedding_fusion_rope_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/rope.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingFusionRopeOpsRunner::PositionEmbeddingFusionRopeOpsRunner(const PositionEmbeddingFusionParam &param)
    : OpsRunner("PositionEmbeddingFusionRopeOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_2D_MIXED_FUSION), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingFusionRopeOpsRunner::PositionEmbeddingFusionOpsRunner called, headNum: "
                  << param_.headNum;
    const size_t inTensorSize = 4;
    const size_t outTensorSize = 3;
    const size_t interTensorSize = 2;
    const size_t nodeSize = 2;
    const int32_t kqvSliceSize = 3;
    kernelGraph_.inTensors.resize(inTensorSize);
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &cos_sum = kernelGraph_.inTensors.at(index2);
    AsdOps::Tensor &sin_sum = kernelGraph_.inTensors.at(index3);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(index4);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(index2);  // V

    kernelGraph_.internalTensors.resize(interTensorSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &split0Node = kernelGraph_.nodes[nodeNum++];
    auto &rope0Node = kernelGraph_.nodes[nodeNum++];

    split0Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, kqvSliceSize}};
    split0Node.inTensors = {&mixedQkv};
    split0Node.outTensors = {&qLayer, &kLayer, &value};
    split0Node.inTensorViewFuncs.resize(split0Node.inTensors.size());
    split0Node.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                             AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(index2) / param_.headNum};
    };
    split0Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, kqvSliceSize}});
    };

    ViewFunc ropeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims.resize(2);
        newDims.at(0) = oldDims.at(0) * oldDims.at(1);
        newDims.at(1) = oldDims.at(2);
    };

    ViewFunc ropeKqView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    const size_t ropeViewSize = 4;
    rope0Node.opDesc = {0, "RopeOperation", AsdOps::OpParam::Rope{AsdOps::OpParam::Rope::ROPEND}};
    rope0Node.inTensors = {&qLayer, &kLayer, &cos_sum, &sin_sum, &seqLen};
    rope0Node.outTensors = {&qEmbedded, &kEmbedded};
    rope0Node.inTensorViewFuncs.resize(ropeViewSize);
    rope0Node.inTensorViewFuncs.at(0) = ropeKqView;
    rope0Node.inTensorViewFuncs.at(1) = ropeKqView;
    rope0Node.inTensorViewFuncs.at(index2) = ropeCosSinView;
    rope0Node.inTensorViewFuncs.at(index3) = ropeCosSinView;
}

PositionEmbeddingFusionRopeOpsRunner::~PositionEmbeddingFusionRopeOpsRunner() {}
} // namespace AclTransformer
