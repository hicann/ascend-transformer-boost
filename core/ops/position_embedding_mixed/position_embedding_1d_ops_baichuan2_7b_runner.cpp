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
#include "position_embedding_1d_ops_baichuan2_7b_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbedding1dOpsBaichuan7bRunner::PositionEmbedding1dOpsBaichuan7bRunner(const PositionEmbeddingParam &param)
    : OpsRunner("PositionEmbedding1dOpsBaichuan7bRunner", RUNNER_TYPE_POSITION_EMBEDDING_1D_MIXED), param_(param)
{
    ASD_LOG(INFO) <<
        "PositionEmbedding1dOpsBaichuan7bRunner::PositionEmbedding1dOpsBaichuan7bRunner called, headNum: " <<
        param_.headNum;
    const int64_t IN_TENSOR_COUNT = 4;
    const int64_t OUT_TENSOR_COUNT = 3;
    const int64_t INTERNAL_TENSOR_COUNT = 14 + 6;
    const int64_t NODE_COUNT = 13 + 6;

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    int64_t inTensorId = 0;
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &cosEmbed = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &sinEmbed = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    int64_t outTensorId = 0;
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(outTensorId++);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(outTensorId++);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.internalTensors.resize(INTERNAL_TENSOR_COUNT);
    int64_t internalTensorId = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryfp32 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryfp32New = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyfp32 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyfp32New = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &qSlice0 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &qSlice1 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &qSlice1Neg = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &qRotate = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &qRotatefp32 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &kSlice0 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &kSlice1 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &kSice1Neg = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &kRotate = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &kRotatefp32 = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &qcos = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &rotqsin = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &kcos = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &rotksin = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(NODE_COUNT);
    int64_t nodeId = 0;
    auto &splitQkvNode = kernelGraph_.nodes[nodeId++];

    auto &castQfp32Node = kernelGraph_.nodes[nodeId++];
    auto &splitqNode = kernelGraph_.nodes[nodeId++];
    auto &mulsqNode = kernelGraph_.nodes[nodeId++];
    auto &catqNode = kernelGraph_.nodes[nodeId++];
    auto &castRQfp32Node = kernelGraph_.nodes[nodeId++];

    auto &castKfp32Node = kernelGraph_.nodes[nodeId++];
    auto &splitkNode = kernelGraph_.nodes[nodeId++];
    auto &mulskNode = kernelGraph_.nodes[nodeId++];
    auto &catkNode = kernelGraph_.nodes[nodeId++];
    auto &castRKfp32Node = kernelGraph_.nodes[nodeId++];

    auto &mulqcosNode = kernelGraph_.nodes[nodeId++];
    auto &mulrotqsinNode = kernelGraph_.nodes[nodeId++];
    auto &addqNode = kernelGraph_.nodes[nodeId++];
    auto &castQfp16Node = kernelGraph_.nodes[nodeId++];

    auto &mulkcosNode = kernelGraph_.nodes[nodeId++];
    auto &mulrotksinNode = kernelGraph_.nodes[nodeId++];
    auto &addkNode = kernelGraph_.nodes[nodeId++];
    auto &castKfp16Node = kernelGraph_.nodes[nodeId++];

    InferShapePreFunc split1InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({ 0, "SplitOperation", AsdOps::OpParam::Split{ int(dims.size()) - 1, 2 } });
    };

    // split QKV
    splitQkvNode.opDesc = { 0, "SplitOperation", AsdOps::OpParam::Split{ 0, 3 } };
    splitQkvNode.inTensors = { &mixedQkv };
    splitQkvNode.outTensors = { &qLayer, &kLayer, &value };
    splitQkvNode.inTensorViewFuncs.resize(splitQkvNode.inTensors.size());
    splitQkvNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({ 0, "SplitOperation", AsdOps::OpParam::Split{ int(dims.size()) - 1, 3 } });
    };

    castQfp32Node.opDesc = { 0, "ElewiseOperation",
        AsdOps::OpParam::Elewise{ AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_CAST } };
    castQfp32Node.inTensors = { &qLayer };
    castQfp32Node.outTensors = { &queryfp32 };

    // rot(q)
    splitqNode.opDesc = { 0, "SplitOperation", AsdOps::OpParam::Split{ 0, 2 } };
    splitqNode.inTensors = { &qLayer };
    splitqNode.outTensors = { &qSlice0, &qSlice1 };
    splitqNode.inTensorViewFuncs.resize(splitqNode.inTensors.size());
    splitqNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
        AsdOps::SVector<int64_t> &newDims) {
        newDims = { oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum };
    };
    splitqNode.inferShapePreFunc = split1InferShape;

    mulsqNode.opDesc = { 0, "ElewiseOperation",
        AsdOps::OpParam::Elewise{ AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1 } };
    mulsqNode.inTensors = { &qSlice1 };
    mulsqNode.outTensors = { &qSlice1Neg };

    InferShapePreFunc cat0InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({ 0, "ConcatOperation", AsdOps::OpParam::Concat{ int(dims.size()) - 1 } });
    };
    catqNode.opDesc = { 0, "ConcatOperation", AsdOps::OpParam::Concat{ 0 } };
    catqNode.inTensors = { &qSlice1Neg, &qSlice0 };
    catqNode.outTensors = { &qRotate };
    catqNode.inferShapePreFunc = cat0InferShape;

    castRQfp32Node.opDesc = { 0, "ElewiseOperation",
        AsdOps::OpParam::Elewise{ AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_CAST } };
    castRQfp32Node.inTensors = { &qRotate };
    castRQfp32Node.outTensors = { &qRotatefp32 };

    castKfp32Node.opDesc = { 0, "ElewiseOperation",
        AsdOps::OpParam::Elewise{ AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_CAST } };
    castKfp32Node.inTensors = { &kLayer };
    castKfp32Node.outTensors = { &keyfp32 };

    // rot(k)
    splitkNode.opDesc = { 0, "SplitOperation", AsdOps::OpParam::Split{ 0, 2 } };
    splitkNode.inTensors = { &kLayer };
    splitkNode.outTensors = { &kSlice0, &kSlice1 };
    splitkNode.inTensorViewFuncs.resize(splitkNode.inTensors.size());
    splitkNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
        AsdOps::SVector<int64_t> &newDims) {
        newDims = { oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum };
    };
    splitkNode.inferShapePreFunc = split1InferShape;

    mulskNode.opDesc = { 0, "ElewiseOperation",
        AsdOps::OpParam::Elewise{ AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1 } };
    mulskNode.inTensors = { &kSlice1 };
    mulskNode.outTensors = { &kSice1Neg };

    catkNode.opDesc = { 0, "ConcatOperation", AsdOps::OpParam::Concat{ 0 } };
    catkNode.inTensors = { &kSice1Neg, &kSlice0 };
    catkNode.outTensors = { &kRotate };
    catkNode.inferShapePreFunc = cat0InferShape;

    castRKfp32Node.opDesc = { 0, "ElewiseOperation",
        AsdOps::OpParam::Elewise{ AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_CAST } };
    castRKfp32Node.inTensors = { &kRotate };
    castRKfp32Node.outTensors = { &kRotatefp32 };

    // [bs, sq, 1, hs]
    ViewFunc unsqueezeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = { oldDims.at(0), oldDims.at(1), 1, oldDims.at(2) };
    };

    // q*cos + rot(q)*sin
    mulqcosNode.opDesc = { 0, "BroadcastOperation",
        AsdOps::OpParam::Broadcast{ AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL } };
    mulqcosNode.inTensors = { &queryfp32, &cosEmbed };
    mulqcosNode.outTensors = { &qcos };
    mulqcosNode.inTensorViewFuncs.resize(mulqcosNode.inTensors.size());
    mulqcosNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
        AsdOps::SVector<int64_t> &newDims) {
        newDims = { oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum };
    };
    mulqcosNode.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    mulrotqsinNode.opDesc = { 0, "BroadcastOperation",
        AsdOps::OpParam::Broadcast{ AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL } };
    mulrotqsinNode.inTensors = { &qRotatefp32, &sinEmbed };
    mulrotqsinNode.outTensors = { &rotqsin };
    mulrotqsinNode.inTensorViewFuncs.resize(mulrotqsinNode.inTensors.size());
    mulrotqsinNode.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    addqNode.opDesc = { 0, "BroadcastOperation",
        AsdOps::OpParam::Broadcast{ AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD } };
    addqNode.inTensors = { &qcos, &rotqsin };
    addqNode.outTensors = { &queryfp32New };

    castQfp16Node.opDesc = { 0, "ElewiseOperation",
        AsdOps::OpParam::Elewise{ AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_CAST } };
    castQfp16Node.inTensors = { &queryfp32New };
    castQfp16Node.outTensors = { &qEmbedded };

    // k*cos + rot(k)*sin
    mulkcosNode.opDesc = { 0, "BroadcastOperation",
        AsdOps::OpParam::Broadcast{ AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL } };
    mulkcosNode.inTensors = { &keyfp32, &cosEmbed };
    mulkcosNode.outTensors = { &kcos };
    mulkcosNode.inTensorViewFuncs.resize(mulkcosNode.inTensors.size());
    mulkcosNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
        AsdOps::SVector<int64_t> &newDims) {
        newDims = { oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum };
    };
    mulkcosNode.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    mulrotksinNode.opDesc = { 0, "BroadcastOperation",
        AsdOps::OpParam::Broadcast{ AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL } };
    mulrotksinNode.inTensors = { &kRotatefp32, &sinEmbed };
    mulrotksinNode.outTensors = { &rotksin };
    mulrotksinNode.inTensorViewFuncs.resize(mulrotksinNode.inTensors.size());
    mulrotksinNode.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    addkNode.opDesc = { 0, "BroadcastOperation",
        AsdOps::OpParam::Broadcast{ AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD } };
    addkNode.inTensors = { &kcos, &rotksin };
    addkNode.outTensors = { &keyfp32New };

    castKfp16Node.opDesc = { 0, "ElewiseOperation",
        AsdOps::OpParam::Elewise{ AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_CAST } };
    castKfp16Node.inTensors = { &keyfp32New };
    castKfp16Node.outTensors = { &kEmbedded };
}

PositionEmbedding1dOpsBaichuan7bRunner::~PositionEmbedding1dOpsBaichuan7bRunner() {}
} // namespace AclTransformer
