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
#include "position_embedding_fusion_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/rope.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingFusionOpsRunner::PositionEmbeddingFusionOpsRunner(const PositionEmbeddingFusionParam &param)
    : OpsRunner("PositionEmbeddingFusionOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_2D_MIXED_FUSION), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingFusionOpsRunner::PositionEmbeddingFusionOpsRunner called, headNum: "
                  << param_.headNum;
    const size_t inTensorSize = 5;
    const size_t outTensorSize = 3;
    const size_t interTensorSize = 10;
    const size_t nodeSize = 10;
    const int32_t kqvSliceSize = 3;
    kernelGraph_.inTensors.resize(inTensorSize);
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(index2);
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(index3);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(index4);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(index2);  // V

    kernelGraph_.internalTensors.resize(interTensorSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &positionIds0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &positionIds1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos_sum = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin_sum = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &split0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided1Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding0Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding1Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding2Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding3Node = kernelGraph_.nodes[nodeNum++];
    auto &concate4Node = kernelGraph_.nodes[nodeNum++];
    auto &concate5Node = kernelGraph_.nodes[nodeNum++];
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

    asStrided0Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided0Node.inTensors = {&positionIds};
    asStrided0Node.outTensors = {&positionIds0}; 
    asStrided0Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(2), dims.at(0)};
        AsdOps::SVector<int64_t> stride = {1, dims.at(1) * dims.at(2)};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, {0}}});
    };

    asStrided1Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided1Node.inTensors = {&positionIds};
    asStrided1Node.outTensors = {&positionIds1};
    asStrided1Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(2), dims.at(0)};
        AsdOps::SVector<int64_t> stride = {1, dims.at(1) * dims.at(2)};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, {dims.at(2)}}});
    };

    ViewFunc Squeeze1 = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        if (oldDims.at(1) == 1) {
            newDims.resize(oldDims.size() - 1);
            newDims.at(0) = oldDims.at(0);
            for (size_t i = 1; i < newDims.size(); i++) {
                newDims.at(i) = oldDims.at(i + 1);
            }
        } else {
            newDims = oldDims;
        }
    };

    embedding0Node.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embedding0Node.inTensors = {&cosTable, &positionIds0};
    embedding0Node.outTensors = {&cos0};
    embedding0Node.inTensorViewFuncs.resize(embedding0Node.inTensors.size());
    embedding0Node.inTensorViewFuncs.at(0) = Squeeze1;

    embedding1Node.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embedding1Node.inTensors = {&sinTable, &positionIds0};
    embedding1Node.outTensors = {&sin0};
    embedding1Node.inTensorViewFuncs.resize(embedding1Node.inTensors.size());
    embedding1Node.inTensorViewFuncs.at(0) = Squeeze1;

    embedding2Node.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embedding2Node.inTensors = {&cosTable, &positionIds1};
    embedding2Node.outTensors = {&cos1};
    embedding2Node.inTensorViewFuncs.resize(embedding2Node.inTensors.size());
    embedding2Node.inTensorViewFuncs.at(0) = Squeeze1;

    embedding3Node.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embedding3Node.inTensors = {&sinTable, &positionIds1};
    embedding3Node.outTensors = {&sin1};
    embedding3Node.inTensorViewFuncs.resize(embedding3Node.inTensors.size());
    embedding3Node.inTensorViewFuncs.at(0) = Squeeze1;

    InferShapePreFunc cat0InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "ConcatOperation", AsdOps::OpParam::Concat{int(dims.size()) - 1}});
    };
    
    concate4Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    concate4Node.inTensors = {&cos0, &cos1};
    concate4Node.outTensors = {&cos_sum};
    concate4Node.inferShapePreFunc = cat0InferShape;
    
    concate5Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    concate5Node.inTensors = {&sin0, &sin1};
    concate5Node.outTensors = {&sin_sum};
    concate5Node.inferShapePreFunc = cat0InferShape;

    ViewFunc ropeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims.resize(2);
        newDims.at(0) = oldDims.at(0) * oldDims.at(1);
        newDims.at(1) = oldDims.at(2); 
    }; 

    ViewFunc ropeKqView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

    const size_t ropeViewSize = 4;
    rope0Node.opDesc = {0, "RopeOperation", AsdOps::OpParam::Rope{AsdOps::OpParam::Rope::ROPE}};
    rope0Node.inTensors = {&qLayer, &kLayer, &cos_sum, &sin_sum, &seqLen};
    rope0Node.outTensors = {&qEmbedded, &kEmbedded};
    rope0Node.inTensorViewFuncs.resize(ropeViewSize);
    rope0Node.inTensorViewFuncs.at(0) = ropeKqView;
    rope0Node.inTensorViewFuncs.at(1) = ropeKqView;
    rope0Node.inTensorViewFuncs.at(index2) = ropeCosSinView;
    rope0Node.inTensorViewFuncs.at(index3) = ropeCosSinView;
    rope0Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        runInfo.GetInTensor(4).desc.dtype = AsdOps::TENSOR_DTYPE_UINT32;
    };
}

PositionEmbeddingFusionOpsRunner::~PositionEmbeddingFusionOpsRunner() {}
} // namespace AclTransformer
