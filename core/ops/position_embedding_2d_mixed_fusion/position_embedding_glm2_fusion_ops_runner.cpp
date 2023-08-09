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
#include "position_embedding_glm2_fusion_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/rope.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingGlm2FusionOpsRunner::PositionEmbeddingGlm2FusionOpsRunner(const PositionEmbeddingFusionParam &param)
    : OpsRunner("PositionEmbeddingGlm2FusionOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_2D_MIXED_FUSION), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingGlm2FusionOpsRunner::PositionEmbeddingGlm2FusionOpsRunner called, headNum: "
                  << param_.headNum;
    const size_t inTensorSize = 3;
    const size_t outTensorSize = 3;
    const size_t interTensorSize = 14;
    const size_t nodeSize = 13;
    kernelGraph_.inTensors.resize(inTensorSize);
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &ropeCache = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(2);

    kernelGraph_.outTensors.resize(outTensorSize);
    AsdOps::Tensor &qOut = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kOut = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(interTensorSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &valueIntermediate = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &ropeCacheSlice = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin1 = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &asStrided0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided1Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided2Node = kernelGraph_.nodes[nodeNum++];
    auto &muls0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided3Node = kernelGraph_.nodes[nodeNum++];
    auto &split0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided4Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided5Node = kernelGraph_.nodes[nodeNum++];
    auto &split1Node = kernelGraph_.nodes[nodeNum++];
    auto &split2Node = kernelGraph_.nodes[nodeNum++];
    auto &rope0Node = kernelGraph_.nodes[nodeNum++];
    auto &cat0Node = kernelGraph_.nodes[nodeNum++];
    auto &cat1Node = kernelGraph_.nodes[nodeNum++];

    int64_t qLayerDim = param_.numHeadsPerPartition * param_.hiddenSizePerHead;
    int64_t kLayerDim = param_.numGroupsPerPartition * param_.hiddenSizePerHead;
    int64_t hiddenSizePerHead = param_.hiddenSizePerHead;
    int64_t numGroupsPerPartition = param_.numGroupsPerPartition;

    // split qkv
    asStrided0Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided0Node.inTensors = {&mixedQkv};
    asStrided0Node.outTensors = {&qLayer};
    asStrided0Node.inferShapePreFunc = [qLayerDim](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split q";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), qLayerDim};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2), dims.at(2), 1};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    asStrided1Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided1Node.inTensors = {&mixedQkv};
    asStrided1Node.outTensors = {&kLayer};
    asStrided1Node.inferShapePreFunc = [qLayerDim, kLayerDim](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split k";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), kLayerDim};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2), dims.at(2), 1};
        AsdOps::SVector<int64_t> offset = {qLayerDim};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    asStrided2Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided2Node.inTensors = {&mixedQkv};
    asStrided2Node.outTensors = {&valueIntermediate};
    asStrided2Node.inferShapePreFunc = [qLayerDim, kLayerDim](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split v";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), kLayerDim};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2), dims.at(2), 1};
        AsdOps::SVector<int64_t> offset = {qLayerDim + kLayerDim};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    // view qkv before slice
    muls0Node.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, 1}};
    muls0Node.inTensors = {&valueIntermediate};
    muls0Node.outTensors = {&value};
    muls0Node.inTensorViewFuncs.resize(muls0Node.inTensors.size());
    muls0Node.inTensorViewFuncs.at(0) = [numGroupsPerPartition, hiddenSizePerHead](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), numGroupsPerPartition, hiddenSizePerHead};
    };

    InferShapePreFunc split1InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 2}});
    };
    
    // slice rope with sq
    asStrided3Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided3Node.inTensors = {&ropeCache};
    asStrided3Node.outTensors = {&ropeCacheSlice};
    asStrided3Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t sqLen = kernelGraph_.inTensors.at(0).desc.dims[0];
        ASD_LOG(INFO) << "slice rope with sq"<< sqLen;
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {sqLen, dims.at(1), dims.at(2), dims.at(3)};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3), 1};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    // split to first half (cos) and second half (sin)
    split0Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split0Node.inTensors = {&ropeCacheSlice};
    split0Node.outTensors = {&cos, &sin};
    split0Node.inferShapePreFunc = split1InferShape;

    // need to double cos and sin
    InferShapePreFunc sinCosInfer = [](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "double cos and sin";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), 2};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3), 0};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    asStrided4Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided4Node.inTensors = {&cos};
    asStrided4Node.outTensors = {&cos1};
    asStrided4Node.inferShapePreFunc = sinCosInfer;

    asStrided5Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided5Node.inTensors = {&sin};
    asStrided5Node.outTensors = {&sin1};
    asStrided5Node.inferShapePreFunc = sinCosInfer;

    // split q and k to half
    split1Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split1Node.inTensors = {&qLayer};
    split1Node.outTensors = {&qChunk0, &qChunk1};
    split1Node.inferShapePreFunc = split1InferShape;
    split1Node.inTensorViewFuncs.resize(split1Node.inTensors.size());
    split1Node.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        int64_t sqLen = kernelGraph_.inTensors.at(0).desc.dims[0];
        int64_t batchSize = kernelGraph_.inTensors.at(0).desc.dims[1];
        newDims = {sqLen, batchSize, param_.numHeadsPerPartition, param_.hiddenSizePerHead};
    };

    split2Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split2Node.inTensors = {&kLayer};
    split2Node.outTensors = {&kChunk0, &kChunk1};
    split2Node.inferShapePreFunc = split1InferShape;
    split2Node.inTensorViewFuncs.resize(split2Node.inTensors.size());
    split2Node.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        int64_t sqLen = kernelGraph_.inTensors.at(0).desc.dims[0];
        int64_t batchSize = kernelGraph_.inTensors.at(0).desc.dims[1];
        newDims = {sqLen, batchSize, param_.numGroupsPerPartition, param_.hiddenSizePerHead};
    };
    
    ViewFunc ropeView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    }; 

    int rotDim = hiddenSizePerHead / 2;
    rope0Node.opDesc = {0, "RopeOperation", AsdOps::OpParam::Rope{AsdOps::OpParam::Rope::ROPEND, rotDim}};
    rope0Node.inTensors = {&qChunk0, &kChunk0, &cos1, &sin1, &seqLen};
    rope0Node.outTensors = {&qEmbedded, &kEmbedded};
    rope0Node.inTensorViewFuncs = {ropeView, ropeView, ropeView, ropeView};
    rope0Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        runInfo.GetInTensor(4).desc.dtype = AsdOps::TENSOR_DTYPE_UINT32;
    };

    // concat
    InferShapePreFunc catInferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "ConcatOperation", AsdOps::OpParam::Concat{int(dims.size()) - 1}});
    };
    cat0Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat0Node.inTensors = {&qEmbedded, &qChunk1};
    cat0Node.outTensors = {&qOut};
    cat0Node.inTensorViewFuncs.resize(cat0Node.inTensors.size());
    cat0Node.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        int64_t sqLen = kernelGraph_.inTensors.at(0).desc.dims[0];
        int64_t batchSize = kernelGraph_.inTensors.at(0).desc.dims[1];
        int64_t lastDim = oldDims.at(1) / param_.numHeadsPerPartition;
        newDims = {sqLen, batchSize, param_.numHeadsPerPartition, lastDim};
    };
    cat0Node.inferShapePreFunc = catInferShape;

    cat1Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat1Node.inTensors = {&kEmbedded, &kChunk1};
    cat1Node.outTensors = {&kOut};
    cat1Node.inTensorViewFuncs.resize(cat1Node.inTensors.size());
    cat1Node.inTensorViewFuncs.at(0) = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        int64_t sqLen = kernelGraph_.inTensors.at(0).desc.dims[0];
        int64_t batchSize = kernelGraph_.inTensors.at(0).desc.dims[1];
        int64_t lastDim = oldDims.at(1) / param_.numGroupsPerPartition;
        newDims = {sqLen, batchSize, param_.numGroupsPerPartition, lastDim};
    };
    cat1Node.inferShapePreFunc = catInferShape;
}

PositionEmbeddingGlm2FusionOpsRunner::~PositionEmbeddingGlm2FusionOpsRunner() {}
} // namespace AclTransformer
