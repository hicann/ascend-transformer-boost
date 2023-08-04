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
#include "position_embedding_ops_glm2_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingOpsGlm2Runner::PositionEmbeddingOpsGlm2Runner(const PositionEmbeddingParam &param)
    : OpsRunner("PositionEmbeddingOpsGlm2Runner", RUNNER_TYPE_POSITION_EMBEDDING_2D_MIXED), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingOpsGlm2Runner::PositionEmbeddingOpsGlm2Runner called, numHeadPerPartition: " << param_.numHeadPerPartition;
    ASD_LOG(INFO) << "numHeadPerPartition: " << param_.hiddenSizePerHead << "numGroupsPerPartition: " << param_.numGroupsPerPartition;
    kernelGraph_.inTensors.resize(2);
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &ropeCache = kernelGraph_.inTensors.at(1);

    kernelGraph_.outTensors.resize(3);
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(2);



    kernelGraph_.internalTensors.resize(29);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &valueIntermediate = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &ropeCacheSlice = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &ropeCacheSlice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &ropeCacheSlice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &ropeCacheSlice1Neg = kernelGraph_.internalTensors.at(internalTensorNum++);
    // q
    AsdOps::Tensor &qChunk0Slice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk0Slice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded2 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded1Part1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded1Part2 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded1Part3 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded1Part4 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    // k
    AsdOps::Tensor &kChunk0Slice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk0Slice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded2 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded1Part2 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded1Part1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded1Part3 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded1Part4 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kOut = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(29);
    int64_t nodeNum = 0;
    auto &asStrided0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided1Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided2Node = kernelGraph_.nodes[nodeNum++];
    auto &muls0Node = kernelGraph_.nodes[nodeNum++];
    auto &split1Node = kernelGraph_.nodes[nodeNum++];
    auto &split2Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided3Node = kernelGraph_.nodes[nodeNum++];
    auto &split3Node = kernelGraph_.nodes[nodeNum++];
    auto &muls3Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided4Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided5Node = kernelGraph_.nodes[nodeNum++];
    auto &mul11Node = kernelGraph_.nodes[nodeNum++];
    auto &mul12Node = kernelGraph_.nodes[nodeNum++];
    auto &mul13Node = kernelGraph_.nodes[nodeNum++];
    auto &mul14Node = kernelGraph_.nodes[nodeNum++];
    auto &add0Node = kernelGraph_.nodes[nodeNum++];
    auto &add1Node = kernelGraph_.nodes[nodeNum++];
    auto &cat0Node = kernelGraph_.nodes[nodeNum++];
    auto &cat1Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided6Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided7Node = kernelGraph_.nodes[nodeNum++];
    auto &mul15Node = kernelGraph_.nodes[nodeNum++];
    auto &mul16Node = kernelGraph_.nodes[nodeNum++];
    auto &mul17Node = kernelGraph_.nodes[nodeNum++];
    auto &mul18Node = kernelGraph_.nodes[nodeNum++];
    auto &add2Node = kernelGraph_.nodes[nodeNum++];
    auto &add3Node = kernelGraph_.nodes[nodeNum++];
    auto &cat2Node = kernelGraph_.nodes[nodeNum++];
    auto &cat3Node = kernelGraph_.nodes[nodeNum++];

    int64_t qLayerDim = param_.numHeadPerPartition * param_.hiddenSizePerHead;
    int64_t kLayerDim = param_.numGroupsPerPartition * param_.hiddenSizePerHead;
    int64_t np = param_.numHeadPerPartition;
    int64_t hn = param_.hiddenSizePerHead;
    int64_t gp = param_.numGroupsPerPartition;

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
    muls0Node.inTensorViewFuncs.at(0) = [gp, hn](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), gp, hn};
    };

    InferShapePreFunc split1InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 2}});
    };
    split1Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split1Node.inTensors = {&qLayer};
    split1Node.outTensors = {&qChunk0, &qChunk1};
    split1Node.inferShapePreFunc = split1InferShape;
    split1Node.inTensorViewFuncs.resize(split1Node.inTensors.size());
    split1Node.inTensorViewFuncs.at(0) = [np, hn](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), np, hn};
    };


    split2Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split2Node.inTensors = {&kLayer};
    split2Node.outTensors = {&kChunk0, &kChunk1};
    split2Node.inferShapePreFunc = split1InferShape;
    split2Node.inTensorViewFuncs.resize(split2Node.inTensors.size());
    split2Node.inTensorViewFuncs.at(0) = [gp, hn](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), gp, hn};
    };

    // slice rope with sq
    asStrided3Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided3Node.inTensors = {&ropeCache};
    asStrided3Node.outTensors = {&ropeCacheSlice};
    asStrided3Node.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        int64_t sq_len = kernelGraph_.inTensors.at(0).desc.dims[0];
        ASD_LOG(INFO) << "slice rope with sq"<< sq_len;
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {sq_len, dims.at(1), dims.at(2), dims.at(3)};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3), 1};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    // split to first half (cos) and second half (sin)
    split3Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split3Node.inTensors = {&ropeCacheSlice};
    split3Node.outTensors = {&ropeCacheSlice0, &ropeCacheSlice1};
    split3Node.inferShapePreFunc = split1InferShape;

    muls3Node.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    muls3Node.inTensors = {&ropeCacheSlice1};
    muls3Node.outTensors = {&ropeCacheSlice1Neg};

    // split q
    asStrided4Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided4Node.inTensors = {&qChunk0};
    asStrided4Node.outTensors = {&qChunk0Slice0};
    asStrided4Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split q 1";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), dims.at(3) / 2};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3), 2};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    asStrided5Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided5Node.inTensors = {&qChunk0};
    asStrided5Node.outTensors = {&qChunk0Slice1};
    asStrided5Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split q 2";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), dims.at(3) / 2};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3), 2};
        AsdOps::SVector<int64_t> offset = {1};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    // q mul
    ViewFunc Unsqueeze = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(3) / 1, 1, oldDims.at(2)};
    };

    mul11Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul11Node.inTensors = {&qChunk0Slice0, &ropeCacheSlice0};
    mul11Node.outTensors = {&qEmbedded1Part1};
    mul11Node.inTensorViewFuncs.resize(mul11Node.inTensors.size());
    mul11Node.inTensorViewFuncs.at(1) = Unsqueeze;

    mul12Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul12Node.inTensors = {&qChunk0Slice1, &ropeCacheSlice1Neg};
    mul12Node.outTensors = {&qEmbedded1Part2};
    mul12Node.inTensorViewFuncs.resize(mul12Node.inTensors.size());
    mul12Node.inTensorViewFuncs.at(1) = Unsqueeze;

    mul13Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul13Node.inTensors = {&qChunk0Slice1, &ropeCacheSlice0};
    mul13Node.outTensors = {&qEmbedded1Part3};
    mul13Node.inTensorViewFuncs.resize(mul13Node.inTensors.size());
    mul13Node.inTensorViewFuncs.at(1) = Unsqueeze;

    mul14Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul14Node.inTensors = {&qChunk0Slice0, &ropeCacheSlice1};
    mul14Node.outTensors = {&qEmbedded1Part4};
    mul14Node.inTensorViewFuncs.resize(mul14Node.inTensors.size());
    mul14Node.inTensorViewFuncs.at(1) = Unsqueeze;

    // add 
    add0Node.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    add0Node.inTensors = {&qEmbedded1Part1, &qEmbedded1Part2};
    add0Node.outTensors = {&qEmbedded1};

    add1Node.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    add1Node.inTensors = {&qEmbedded1Part3, &qEmbedded1Part4};
    add1Node.outTensors = {&qEmbedded2};

    // concat
    InferShapePreFunc cat0InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "ConcatOperation", AsdOps::OpParam::Concat{int(dims.size()) - 1}});
    };
    ViewFunc Unsqueeze1 = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2), oldDims.at(3), 1};
    };
    cat0Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat0Node.inTensors = {&qEmbedded1, &qEmbedded2};
    cat0Node.outTensors = {&qOut};
    cat0Node.inTensorViewFuncs.resize(cat0Node.inTensors.size());
    cat0Node.inTensorViewFuncs.at(0) = Unsqueeze1;
    cat0Node.inTensorViewFuncs.at(1) = Unsqueeze1;
    cat0Node.inferShapePreFunc = cat0InferShape;

    cat1Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat1Node.inTensors = {&qOut, &qChunk1};
    cat1Node.outTensors = {&qEmbedded};
    cat1Node.inTensorViewFuncs.resize(cat1Node.inTensors.size());
    cat1Node.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2), oldDims.at(3) * 2};
    };
    cat1Node.inferShapePreFunc = cat0InferShape;
    // q emb end

    // split k
    asStrided6Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided6Node.inTensors = {&kChunk0};
    asStrided6Node.outTensors = {&kChunk0Slice0};
    asStrided6Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split k 1";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), dims.at(3) / 2};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3), 2};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    asStrided7Node.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    asStrided7Node.inTensors = {&kChunk0};
    asStrided7Node.outTensors = {&kChunk0Slice1};
    asStrided7Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        ASD_LOG(INFO) << "split k 2";
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), dims.at(3) / 2};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3), 2};
        AsdOps::SVector<int64_t> offset = {1};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };

    // k mul
    mul15Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul15Node.inTensors = {&kChunk0Slice0, &ropeCacheSlice0};
    mul15Node.outTensors = {&kEmbedded1Part1};
    mul15Node.inTensorViewFuncs.resize(mul15Node.inTensors.size());
    mul15Node.inTensorViewFuncs.at(1) = Unsqueeze;

    mul16Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul16Node.inTensors = {&kChunk0Slice1, &ropeCacheSlice1Neg};
    mul16Node.outTensors = {&kEmbedded1Part2};
    mul16Node.inTensorViewFuncs.resize(mul16Node.inTensors.size());
    mul16Node.inTensorViewFuncs.at(1) = Unsqueeze;

    mul17Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul17Node.inTensors = {&kChunk0Slice1, &ropeCacheSlice0};
    mul17Node.outTensors = {&kEmbedded1Part3};
    mul17Node.inTensorViewFuncs.resize(mul17Node.inTensors.size());
    mul17Node.inTensorViewFuncs.at(1) = Unsqueeze;

    mul18Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul18Node.inTensors = {&kChunk0Slice0, &ropeCacheSlice1};
    mul18Node.outTensors = {&kEmbedded1Part4};
    mul18Node.inTensorViewFuncs.resize(mul18Node.inTensors.size());
    mul18Node.inTensorViewFuncs.at(1) = Unsqueeze;

    // add 
    add2Node.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    add2Node.inTensors = {&kEmbedded1Part1, &kEmbedded1Part2};
    add2Node.outTensors = {&kEmbedded1};

    add3Node.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    add3Node.inTensors = {&kEmbedded1Part3, &kEmbedded1Part4};
    add3Node.outTensors = {&kEmbedded2};

    // concat
    cat2Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat2Node.inTensors = {&kEmbedded1, &kEmbedded2};
    cat2Node.outTensors = {&kOut};
    cat2Node.inTensorViewFuncs.resize(cat2Node.inTensors.size());
    cat2Node.inTensorViewFuncs.at(0) = Unsqueeze1;
    cat2Node.inTensorViewFuncs.at(1) = Unsqueeze1;
    cat2Node.inferShapePreFunc = cat0InferShape;

    cat3Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat3Node.inTensors = {&kOut, &kChunk1};
    cat3Node.outTensors = {&kEmbedded};
    cat3Node.inTensorViewFuncs.resize(cat3Node.inTensors.size());
    cat3Node.inTensorViewFuncs.at(0) = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2), oldDims.at(3) * 2};
    };
    cat3Node.inferShapePreFunc = cat0InferShape;
    // k emb end
}

PositionEmbeddingOpsGlm2Runner::~PositionEmbeddingOpsGlm2Runner() {}
} // namespace AclTransformer
