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
#include "position_embedding_ops_gptneox20b_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingOpsGptNeox20bRunner::PositionEmbeddingOpsGptNeox20bRunner(const PositionEmbeddingParam &param)
    : OpsRunner("PositionEmbeddingOpsGptNeox20bRunner",  RUNNER_TYPE_POSITION_EMBEDDING_2D_MIXED), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingOpsGptNeox20bRunner::PositionEmbeddingOps20bRunner called, headNum: "
                  << param_.headNum;

    kernelGraph_.inTensors.resize(4);
    int64_t inTensorId = 0;
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(3);
    int64_t outTensorId = 0;
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(outTensorId++);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(outTensorId++);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.internalTensors.resize(22);
    int64_t internalTensorId = 0;
    AsdOps::Tensor &querySplit = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keySplit = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRot = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryPass = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRot = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyPass = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &cosEmbed = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &sinEmbed = kernelGraph_.internalTensors.at(internalTensorId++);

    AsdOps::Tensor &queryRotLeft = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRotRight = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRotRightNeg = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRotCat = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRotCosMul = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRotSinMul = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRotAdd = kernelGraph_.internalTensors.at(internalTensorId++);

    AsdOps::Tensor &keyRotLeft = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRotRight = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRotRightNeg = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRotCat = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRotCosMul = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRotSinMul = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRotAdd = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(21);
    int64_t nodeId = 0;
    auto &splitQKVNode = kernelGraph_.nodes[nodeId++];
    auto &qRotSliceNode = kernelGraph_.nodes[nodeId++];
    auto &qPassSliceNode = kernelGraph_.nodes[nodeId++];
    auto &kRotSliceNode = kernelGraph_.nodes[nodeId++];
    auto &kPassSliceNode = kernelGraph_.nodes[nodeId++];
    auto &cosEmbedNode = kernelGraph_.nodes[nodeId++];
    auto &sinEmbedNode = kernelGraph_.nodes[nodeId++];
    // do rotary embedding for q
    auto &qSliceNode = kernelGraph_.nodes[nodeId++];
    auto &qNegNode = kernelGraph_.nodes[nodeId++];
    auto &qCatNegNode = kernelGraph_.nodes[nodeId++];
    auto &qCosMulNode = kernelGraph_.nodes[nodeId++];
    auto &qSinMulNode = kernelGraph_.nodes[nodeId++];
    auto &qAddNode = kernelGraph_.nodes[nodeId++];
    // do rotary embedding for k
    auto &kSliceNode = kernelGraph_.nodes[nodeId++];
    auto &kNegNode = kernelGraph_.nodes[nodeId++];
    auto &kCatNegNode = kernelGraph_.nodes[nodeId++];
    auto &kCosMulNode = kernelGraph_.nodes[nodeId++];
    auto &kSinMulNode = kernelGraph_.nodes[nodeId++];
    auto &kAddNode = kernelGraph_.nodes[nodeId++];
    // do cat and output
    auto &qCatNode = kernelGraph_.nodes[nodeId++];
    auto &kCatNode = kernelGraph_.nodes[nodeId++];

    // split mixedQKV to q k v
    // [bs, sq, hn * 3 * hs] --> [bs, sq, hn, 3*hs] --> 3 of [bs, sq, hn, hs]
    splitQKVNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 3}};
    splitQKVNode.inTensors = {&mixedQkv};
    splitQKVNode.outTensors = {&querySplit, &keySplit, &value};
    splitQKVNode.inTensorViewFuncs.resize(splitQKVNode.inTensors.size());
    splitQKVNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                               AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };
    splitQKVNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    };

    // use asstride to split tensors
    int64_t rotaryNum = param_.dk * param_.rotaryPct;
    int64_t passNum = param_.dk - rotaryNum;
    ASD_LOG(INFO) << "Rotary num is " << rotaryNum << " pass num is " << passNum;
    InferShapePreFunc splitRotPreFunc = [=](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), rotaryNum};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3),
                                           1};
        int64_t offset = 0;
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, {offset}}});
    };
    InferShapePreFunc splitPassPreFunc = [=](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), passNum};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(3), dims.at(2) * dims.at(3), dims.at(3),
                                           1};
        int64_t offset = rotaryNum;
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, {offset}}});
    };

    // [bs, sq, hn, rd]
    qRotSliceNode.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    qRotSliceNode.inTensors = {&querySplit};
    qRotSliceNode.outTensors = {&queryRot};
    qRotSliceNode.inferShapePreFunc = splitRotPreFunc;

    qPassSliceNode.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    qPassSliceNode.inTensors = {&querySplit};
    qPassSliceNode.outTensors = {&queryPass};
    qPassSliceNode.inferShapePreFunc = splitPassPreFunc;

    kRotSliceNode.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    kRotSliceNode.inTensors = {&keySplit};
    kRotSliceNode.outTensors = {&keyRot};
    kRotSliceNode.inferShapePreFunc = splitRotPreFunc;

    kPassSliceNode.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    kPassSliceNode.inTensors = {&keySplit};
    kPassSliceNode.outTensors = {&keyPass};
    kPassSliceNode.inferShapePreFunc = splitPassPreFunc;

    // do embedding // [bs, sq, rd]
    cosEmbedNode.opDesc = {0, "GatherOperation",
                           AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    cosEmbedNode.inTensors = {&cosTable, &positionIds};
    cosEmbedNode.outTensors = {&cosEmbed};
    cosEmbedNode.inTensorViewFuncs.resize(cosEmbedNode.inTensors.size());

    sinEmbedNode.opDesc = {0, "GatherOperation",
                           AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    sinEmbedNode.inTensors = {&sinTable, &positionIds};
    sinEmbedNode.outTensors = {&sinEmbed};
    sinEmbedNode.inTensorViewFuncs.resize(sinEmbedNode.inTensors.size());

    // do query rotary embedding
    // 2 * [bs, sq, hn, rd/2]
    InferShapePreFunc splitHalfPreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 2}});
    };

    qSliceNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    qSliceNode.inTensors = {&queryRot};
    qSliceNode.outTensors = {&queryRotLeft, &queryRotRight};
    qSliceNode.inferShapePreFunc = splitHalfPreFunc;

    // [bs, sq, hb, rd/2]
    qNegNode.opDesc = {0, "ElewiseOperation",
                       AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    qNegNode.inTensors = {&queryRotRight};
    qNegNode.outTensors = {&queryRotRightNeg};

    // [bs, sq, hn, rd]
    InferShapePreFunc catHalfPreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "ConcatOperation", AsdOps::OpParam::Concat{int(dims.size()) - 1}});
    };

    qCatNegNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    qCatNegNode.inTensors = {&queryRotRightNeg, &queryRotLeft};
    qCatNegNode.outTensors = {&queryRotCat};
    qCatNegNode.inferShapePreFunc = catHalfPreFunc;

    // cos sin unsqueeze view
    // [bs, sq, 1, rd]
    ViewFunc unsqueezeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), 1, oldDims.at(2)};
    };

    // [bs, sq, hn, rd]
    qCosMulNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    qCosMulNode.inTensors = {&queryRot, &cosEmbed};
    qCosMulNode.outTensors = {&queryRotCosMul};
    qCosMulNode.inTensorViewFuncs.resize(qCosMulNode.inTensors.size());
    qCosMulNode.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    // [bs, sq, hn, rd]
    qSinMulNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    qSinMulNode.inTensors = {&queryRotCat, &sinEmbed};
    qSinMulNode.outTensors = {&queryRotSinMul};
    qSinMulNode.inTensorViewFuncs.resize(qSinMulNode.inTensors.size());
    qSinMulNode.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    // [bs, sq, hb, rd]
    qAddNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    qAddNode.inTensors = {&queryRotCosMul, &queryRotSinMul};
    qAddNode.outTensors = {&queryRotAdd};

    // do key rotary embedding
    kSliceNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    kSliceNode.inTensors = {&keyRot};
    kSliceNode.outTensors = {&keyRotLeft, &keyRotRight};
    kSliceNode.inferShapePreFunc = splitHalfPreFunc;

    // [bs, sq, hb, rd/2]
    kNegNode.opDesc = {0, "ElewiseOperation",
                       AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    kNegNode.inTensors = {&keyRotRight};
    kNegNode.outTensors = {&keyRotRightNeg};

    kCatNegNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    kCatNegNode.inTensors = {&keyRotRightNeg, &keyRotLeft};
    kCatNegNode.outTensors = {&keyRotCat};
    kCatNegNode.inferShapePreFunc = catHalfPreFunc;

    // [bs, sq, hn, rd]
    kCosMulNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    kCosMulNode.inTensors = {&keyRot, &cosEmbed};
    kCosMulNode.outTensors = {&keyRotCosMul};
    kCosMulNode.inTensorViewFuncs.resize(kCosMulNode.inTensors.size());
    kCosMulNode.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    // [bs, sq, hn, rd]
    kSinMulNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    kSinMulNode.inTensors = {&keyRotCat, &sinEmbed};
    kSinMulNode.outTensors = {&keyRotSinMul};
    kSinMulNode.inTensorViewFuncs.resize(kSinMulNode.inTensors.size());
    kSinMulNode.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    // [bs, sq, hb, rd]
    kAddNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    kAddNode.inTensors = {&keyRotCosMul, &keyRotSinMul};
    kAddNode.outTensors = {&keyRotAdd};

    // cat query and key
    InferShapePreFunc catLastDimInferShape = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "ConcatOperation", AsdOps::OpParam::Concat{int(dims.size()) - 1}});
    };

    // out [bs, sq, hb, hs]
    qCatNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    qCatNode.inTensors = {&queryRotAdd, &queryPass};
    qCatNode.outTensors = {&qEmbedded};
    qCatNode.inTensorViewFuncs.resize(qCatNode.inTensors.size());
    qCatNode.inferShapePreFunc = catLastDimInferShape;

    kCatNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    kCatNode.inTensors = {&keyRotAdd, &keyPass};
    kCatNode.outTensors = {&kEmbedded};
    kCatNode.inTensorViewFuncs.resize(kCatNode.inTensors.size());
    kCatNode.inferShapePreFunc = catLastDimInferShape;
}

PositionEmbeddingOpsGptNeox20bRunner::~PositionEmbeddingOpsGptNeox20bRunner() {}
} // namespace AclTransformer