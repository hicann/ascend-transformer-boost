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
#include "position_embedding_1d_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbedding1dOpsRunner::PositionEmbedding1dOpsRunner(const PositionEmbeddingParam &param)
    : OpsRunner("PositionEmbedding1dOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_1D_MIXED), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbedding1dOpsRunner::PositionEmbedding1dOpsRunner called, headNum: " << param_.headNum;
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors.resize(3);
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(16);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qSlice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qSlice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qSlice1Neg = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qRotate = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kSlice0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kSlice1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kSice1Neg = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kRotate = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &qcos = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &rotqsin = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kcos = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &rotksin = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(15);
    int64_t nodeNum = 0;
    auto &splitQkvNode = kernelGraph_.nodes[nodeNum++];
    auto &embeddingCosNode = kernelGraph_.nodes[nodeNum++];
    auto &embeddingSinNode = kernelGraph_.nodes[nodeNum++];
    auto &splitqNode = kernelGraph_.nodes[nodeNum++];
    auto &mulsqNode = kernelGraph_.nodes[nodeNum++];
    auto &catqNode = kernelGraph_.nodes[nodeNum++];
    auto &splitkNode = kernelGraph_.nodes[nodeNum++];
    auto &mulskNode = kernelGraph_.nodes[nodeNum++];
    auto &catkNode = kernelGraph_.nodes[nodeNum++];
    auto &mulqcosNode = kernelGraph_.nodes[nodeNum++];
    auto &mulrotqsinNode = kernelGraph_.nodes[nodeNum++];
    auto &addqNode = kernelGraph_.nodes[nodeNum++];
    auto &mulkcosNode = kernelGraph_.nodes[nodeNum++];
    auto &mulrotksinNode = kernelGraph_.nodes[nodeNum++];
    auto &addkNode = kernelGraph_.nodes[nodeNum++];
    
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

    ViewFunc Unsqueeze2 = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims.resize(oldDims.size() + 1);
        newDims.at(0) = oldDims.at(0);
        newDims.at(1) = oldDims.at(1);
        newDims.at(2) = 1;
        for (size_t i = 3; i < newDims.size(); i++) {
            newDims.at(i) = oldDims.at(i - 1);
        }
    };

    InferShapePreFunc split1InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 2}});
    };

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

    // get cos
    embeddingCosNode.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embeddingCosNode.inTensors = {&cosTable, &positionIds};
    embeddingCosNode.outTensors = {&cos};
    embeddingCosNode.inTensorViewFuncs.resize(embeddingCosNode.inTensors.size());
    embeddingCosNode.inTensorViewFuncs.at(0) = Squeeze1;

    // get sin
    embeddingSinNode.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embeddingSinNode.inTensors = {&sinTable, &positionIds};
    embeddingSinNode.outTensors = {&sin};
    embeddingSinNode.inTensorViewFuncs.resize(embeddingSinNode.inTensors.size());
    embeddingSinNode.inTensorViewFuncs.at(0) = Squeeze1;

    // rot(q)
    splitqNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    splitqNode.inTensors = {&qLayer};
    splitqNode.outTensors = {&qSlice0, &qSlice1};
    splitqNode.inferShapePreFunc = split1InferShape;

    mulsqNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    mulsqNode.inTensors = {&qSlice1};
    mulsqNode.outTensors = {&qSlice1Neg};

    InferShapePreFunc cat0InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "ConcatOperation", AsdOps::OpParam::Concat{int(dims.size()) - 1}});
    };
    catqNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    catqNode.inTensors = {&qSlice1Neg, &qSlice0};
    catqNode.outTensors = {&qRotate};
    catqNode.inferShapePreFunc = cat0InferShape;

    // rot(k)
    splitkNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    splitkNode.inTensors = {&kLayer};
    splitkNode.outTensors = {&kSlice0, &kSlice1};
    splitkNode.inferShapePreFunc = split1InferShape;

    mulskNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    mulskNode.inTensors = {&kSlice1};
    mulskNode.outTensors = {&kSice1Neg};

    catkNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    catkNode.inTensors = {&kSice1Neg, &kSlice0};
    catkNode.outTensors = {&kRotate};
    catkNode.inferShapePreFunc = cat0InferShape;

    // q*cos + rot(q)*sin
    mulqcosNode.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mulqcosNode.inTensors = {&qLayer, &cos};
    mulqcosNode.outTensors = {&qcos};
    mulqcosNode.inTensorViewFuncs.resize(mulqcosNode.inTensors.size());
    mulqcosNode.inTensorViewFuncs.at(1) = Unsqueeze2;

    mulrotqsinNode.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mulrotqsinNode.inTensors = {&qRotate, &sin};
    mulrotqsinNode.outTensors = {&rotqsin};
    mulrotqsinNode.inTensorViewFuncs.resize(mulrotqsinNode.inTensors.size());
    mulrotqsinNode.inTensorViewFuncs.at(1) = Unsqueeze2;

    addqNode.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    addqNode.inTensors = {&qcos, &rotqsin};
    addqNode.outTensors = {&qEmbedded};

    // k*cos + rot(k)*sin
    mulkcosNode.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mulkcosNode.inTensors = {&kLayer, &cos};
    mulkcosNode.outTensors = {&kcos};
    mulkcosNode.inTensorViewFuncs.resize(mulkcosNode.inTensors.size());
    mulkcosNode.inTensorViewFuncs.at(1) = Unsqueeze2;

    mulrotksinNode.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mulrotksinNode.inTensors = {&kRotate, &sin};
    mulrotksinNode.outTensors = {&rotksin};
    mulrotksinNode.inTensorViewFuncs.resize(mulrotksinNode.inTensors.size());
    mulrotksinNode.inTensorViewFuncs.at(1) = Unsqueeze2;

    addkNode.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    addkNode.inTensors = {&kcos, &rotksin};
    addkNode.outTensors = {&kEmbedded};
}

PositionEmbedding1dOpsRunner::~PositionEmbedding1dOpsRunner() {}
} // namespace AclTransformer
