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
#include "position_embedding_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingOpsRunner::PositionEmbeddingOpsRunner(const PositionEmbeddingParam &param)
    : OpsRunner("PositionEmbeddingOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingOperation::PositionEmbeddingOperation called";
}

PositionEmbeddingOpsRunner::~PositionEmbeddingOpsRunner() {}

AsdOps::Status PositionEmbeddingOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " SetupKernelGraph start: "
                  << "headNum: " << param_.headNum;

    kernelGraph_.inTensors = variantPack.inTensors;
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors = variantPack.outTensors;
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(40);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &positionIds0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &positionIds1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk0Slice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk0Slice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk0Slice1Neg = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qRotate0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk1Slice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk1Slice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qChunk1Slice1Neg = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qRotate1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk0Slice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk0Slice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk0Slice1Neg = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kRotate0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk1Slice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk1Slice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kChunk1Slice1Neg = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kRotate1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded0Part0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded0Part1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded1Part0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded1Part1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qEmbedded1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded0Part0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded0Part1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded1Part0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded1Part1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kEmbedded1 = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(35);
    int64_t nodeNum = 0;
    auto &split0Node = kernelGraph_.nodes[nodeNum++];
    auto &split1Node = kernelGraph_.nodes[nodeNum++];
    auto &split2Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided0Node = kernelGraph_.nodes[nodeNum++];
    auto &asStrided1Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding0Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding1Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding2Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding3Node = kernelGraph_.nodes[nodeNum++];
    auto &split3Node = kernelGraph_.nodes[nodeNum++];
    auto &muls0Node = kernelGraph_.nodes[nodeNum++];
    auto &cat0Node = kernelGraph_.nodes[nodeNum++];
    auto &split4Node = kernelGraph_.nodes[nodeNum++];
    auto &muls1Node = kernelGraph_.nodes[nodeNum++];
    auto &cat1Node = kernelGraph_.nodes[nodeNum++];
    auto &split5Node = kernelGraph_.nodes[nodeNum++];
    auto &muls2Node = kernelGraph_.nodes[nodeNum++];
    auto &cat2Node = kernelGraph_.nodes[nodeNum++];
    auto &split6Node = kernelGraph_.nodes[nodeNum++];
    auto &muls3Node = kernelGraph_.nodes[nodeNum++];
    auto &cat3Node = kernelGraph_.nodes[nodeNum++];
    auto &mul00Node = kernelGraph_.nodes[nodeNum++];
    auto &mul01Node = kernelGraph_.nodes[nodeNum++];
    auto &add0Node = kernelGraph_.nodes[nodeNum++];
    auto &mul10Node = kernelGraph_.nodes[nodeNum++];
    auto &mul11Node = kernelGraph_.nodes[nodeNum++];
    auto &add1Node = kernelGraph_.nodes[nodeNum++];
    auto &mul20Node = kernelGraph_.nodes[nodeNum++];
    auto &mul21Node = kernelGraph_.nodes[nodeNum++];
    auto &add2Node = kernelGraph_.nodes[nodeNum++];
    auto &mul30Node = kernelGraph_.nodes[nodeNum++];
    auto &mul31Node = kernelGraph_.nodes[nodeNum++];
    auto &add3Node = kernelGraph_.nodes[nodeNum++];
    auto &cat4Node = kernelGraph_.nodes[nodeNum++];
    auto &cat5Node = kernelGraph_.nodes[nodeNum++];

    split0Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 3}};
    split0Node.inTensors = {&mixedQkv};
    split0Node.outTensors = {&qLayer, &kLayer, &value};
    split0Node.inTensorViewFuncs.resize(split0Node.inTensors.size());
    split0Node.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                             AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };
    split0Node.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    };

    InferShapePreFunc split1InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 2}});
    };
    split1Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split1Node.inTensors = {&qLayer};
    split1Node.outTensors = {&qChunk0, &qChunk1};
    split1Node.inferShapePreFunc = split1InferShape;
    split2Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split2Node.inTensors = {&kLayer};
    split2Node.outTensors = {&kChunk0, &kChunk1};
    split2Node.inferShapePreFunc = split1InferShape;

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

    split3Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split3Node.inTensors = {&qChunk0};
    split3Node.outTensors = {&qChunk0Slice0, &qChunk0Slice1};
    split3Node.inferShapePreFunc = split1InferShape;

    muls0Node.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    muls0Node.inTensors = {&qChunk0Slice1};
    muls0Node.outTensors = {&qChunk0Slice1Neg};
    InferShapePreFunc cat0InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "ConcatOperation", AsdOps::OpParam::Concat{int(dims.size()) - 1}});
    };
    cat0Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat0Node.inTensors = {&qChunk0Slice1Neg, &qChunk0Slice0};
    cat0Node.outTensors = {&qRotate0};
    cat0Node.inferShapePreFunc = cat0InferShape;

    split4Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split4Node.inTensors = {&qChunk1};
    split4Node.outTensors = {&qChunk1Slice0, &qChunk1Slice1};
    split4Node.inferShapePreFunc = split1InferShape;

    muls1Node.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    muls1Node.inTensors = {&qChunk1Slice1};
    muls1Node.outTensors = {&qChunk1Slice1Neg};
    cat1Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat1Node.inTensors = {&qChunk1Slice1Neg, &qChunk1Slice0};
    cat1Node.outTensors = {&qRotate1};
    cat1Node.inferShapePreFunc = cat0InferShape;

    split5Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split5Node.inTensors = {&kChunk0};
    split5Node.outTensors = {&kChunk0Slice0, &kChunk0Slice1};
    split5Node.inferShapePreFunc = split1InferShape;

    muls2Node.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    muls2Node.inTensors = {&kChunk0Slice1};
    muls2Node.outTensors = {&kChunk0Slice1Neg};
    cat2Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat2Node.inTensors = {&kChunk0Slice1Neg, &kChunk0Slice0};
    cat2Node.outTensors = {&kRotate0};
    cat2Node.inferShapePreFunc = cat0InferShape;

    split6Node.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    split6Node.inTensors = {&kChunk1};
    split6Node.outTensors = {&kChunk1Slice0, &kChunk1Slice1};
    split6Node.inferShapePreFunc = split1InferShape;

    muls3Node.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    muls3Node.inTensors = {&kChunk1Slice1};
    muls3Node.outTensors = {&kChunk1Slice1Neg};
    cat3Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat3Node.inTensors = {&kChunk1Slice1Neg, &kChunk1Slice0};
    cat3Node.outTensors = {&kRotate1};
    cat3Node.inferShapePreFunc = cat0InferShape;

    ViewFunc Unsqueeze2 = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims.resize(oldDims.size() + 1);
        newDims.at(0) = oldDims.at(0);
        newDims.at(1) = oldDims.at(1);
        newDims.at(2) = 1;
        for (size_t i = 3; i < newDims.size(); i++) {
            newDims.at(i) = oldDims.at(i - 1);
        }
    };

    mul00Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul00Node.inTensors = {&qChunk0, &cos0};
    mul00Node.outTensors = {&qEmbedded0Part0};
    mul00Node.inTensorViewFuncs.resize(mul00Node.inTensors.size());
    mul00Node.inTensorViewFuncs.at(1) = Unsqueeze2;
    mul01Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul01Node.inTensors = {&qRotate0, &sin0};
    mul01Node.outTensors = {&qEmbedded0Part1};
    mul01Node.inTensorViewFuncs.resize(mul01Node.inTensors.size());
    mul01Node.inTensorViewFuncs.at(1) = Unsqueeze2;
    add0Node.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    add0Node.inTensors = {&qEmbedded0Part0, &qEmbedded0Part1};
    add0Node.outTensors = {&qEmbedded0};

    mul10Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul10Node.inTensors = {&qChunk1, &cos1};
    mul10Node.outTensors = {&qEmbedded1Part0};
    mul10Node.inTensorViewFuncs.resize(mul10Node.inTensors.size());
    mul10Node.inTensorViewFuncs.at(1) = Unsqueeze2;
    mul11Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul11Node.inTensors = {&qRotate1, &sin1};
    mul11Node.outTensors = {&qEmbedded1Part1};
    mul11Node.inTensorViewFuncs.resize(mul11Node.inTensors.size());
    mul11Node.inTensorViewFuncs.at(1) = Unsqueeze2;
    add1Node.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    add1Node.inTensors = {&qEmbedded1Part0, &qEmbedded1Part1};
    add1Node.outTensors = {&qEmbedded1};

    mul20Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul20Node.inTensors = {&kChunk0, &cos0};
    mul20Node.outTensors = {&kEmbedded0Part0};
    mul20Node.inTensorViewFuncs.resize(mul20Node.inTensors.size());
    mul20Node.inTensorViewFuncs.at(1) = Unsqueeze2;
    mul21Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul21Node.inTensors = {&kRotate0, &sin0};
    mul21Node.outTensors = {&kEmbedded0Part1};
    mul21Node.inTensorViewFuncs.resize(mul21Node.inTensors.size());
    mul21Node.inTensorViewFuncs.at(1) = Unsqueeze2;
    add2Node.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    add2Node.inTensors = {&kEmbedded0Part0, &kEmbedded0Part1};
    add2Node.outTensors = {&kEmbedded0};

    mul30Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul30Node.inTensors = {&kChunk1, &cos1};
    mul30Node.outTensors = {&kEmbedded1Part0};
    mul30Node.inTensorViewFuncs.resize(mul30Node.inTensors.size());
    mul30Node.inTensorViewFuncs.at(1) = Unsqueeze2;
    mul31Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul31Node.inTensors = {&kRotate1, &sin1};
    mul31Node.outTensors = {&kEmbedded1Part1};
    mul31Node.inTensorViewFuncs.resize(mul31Node.inTensors.size());
    mul31Node.inTensorViewFuncs.at(1) = Unsqueeze2;
    add3Node.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    add3Node.inTensors = {&kEmbedded1Part0, &kEmbedded1Part1};
    add3Node.outTensors = {&kEmbedded1};

    cat4Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat4Node.inTensors = {&qEmbedded0, &qEmbedded1};
    cat4Node.outTensors = {&qEmbedded};
    cat4Node.inferShapePreFunc = cat0InferShape;
    cat5Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat5Node.inTensors = {&kEmbedded0, &kEmbedded1};
    cat5Node.outTensors = {&kEmbedded};
    cat5Node.inferShapePreFunc = cat0InferShape;

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
