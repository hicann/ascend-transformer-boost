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
#include "position_embedding_1d_ops_llama7b_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbedding1dOpsLlama7bRunner::PositionEmbedding1dOpsLlama7bRunner(const PositionEmbeddingParam &param)
    : OpsRunner("PositionEmbedding1dOpsLlama7bRunner", RUNNER_TYPE_POSITION_EMBEDDING_1D_MIXED), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbedding1dOpsLlamaRunner::PositionEmbedding1dOpsLlamaRunner called, headNum: " << param_.headNum;
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors.resize(3);
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(20);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &interqLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &interkLayer = kernelGraph_.internalTensors.at(internalTensorNum++);
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
    AsdOps::Tensor &interqEmbedded = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &kcos = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &rotksin = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &interkEmbedded = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(19);
    int64_t nodeNum = 0;
    auto &splitQkvNode = kernelGraph_.nodes[nodeNum++];
    auto &qTransposeNode0 = kernelGraph_.nodes[nodeNum++];
    auto &kTransposeNode0 = kernelGraph_.nodes[nodeNum++];
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
    auto &qTransposeNode1 = kernelGraph_.nodes[nodeNum++];
    auto &mulkcosNode = kernelGraph_.nodes[nodeNum++];
    auto &mulrotksinNode = kernelGraph_.nodes[nodeNum++];
    auto &addkNode = kernelGraph_.nodes[nodeNum++];
    auto &kTransposeNode1 = kernelGraph_.nodes[nodeNum++];
    
    ViewFunc Squeeze1 = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        if (oldDims.at(0) == 1) {
            newDims.resize(oldDims.size() - 2);
            for (size_t i = 0; i < newDims.size(); i++) {
                newDims.at(i) = oldDims.at(i + 2);
            }
        } else {
            newDims = oldDims;
        }
    };

    InferShapePreFunc split1InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 2}});
    };

    // split QKV
    splitQkvNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 3}};
    splitQkvNode.inTensors = {&mixedQkv};
    splitQkvNode.outTensors = {&interqLayer, &interkLayer, &value};
    splitQkvNode.inTensorViewFuncs.resize(splitQkvNode.inTensors.size());
    splitQkvNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    };

    AsdOps::OpParam::Transpose qTransposeNode0Param = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    qTransposeNode0.opDesc = {0, "TransposeOperation", qTransposeNode0Param};
    qTransposeNode0.inTensors = {&interqLayer};
    qTransposeNode0.outTensors = {&qLayer};
    qTransposeNode0.inTensorViewFuncs.resize(qTransposeNode0.inTensors.size());
    qTransposeNode0.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };

    AsdOps::OpParam::Transpose kTransposeNode0Param = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    kTransposeNode0.opDesc = {0, "TransposeOperation", kTransposeNode0Param};
    kTransposeNode0.inTensors = {&interkLayer};
    kTransposeNode0.outTensors = {&kLayer};
    kTransposeNode0.inTensorViewFuncs.resize(kTransposeNode0.inTensors.size());
    kTransposeNode0.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
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

    mulrotqsinNode.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mulrotqsinNode.inTensors = {&qRotate, &sin};
    mulrotqsinNode.outTensors = {&rotqsin};
    mulrotqsinNode.inTensorViewFuncs.resize(mulrotqsinNode.inTensors.size());

    addqNode.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    addqNode.inTensors = {&qcos, &rotqsin};
    addqNode.outTensors = {&interqEmbedded};

    AsdOps::OpParam::Transpose qTransposeNode1Param = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {2, 0, 1, 3}};
    qTransposeNode1.opDesc = {0, "TransposeOperation", qTransposeNode1Param};
    qTransposeNode1.inTensors = {&interqEmbedded};
    qTransposeNode1.outTensors = {&qEmbedded};
    qTransposeNode1.inTensorViewFuncs.resize(qTransposeNode1.inTensors.size());

    // k*cos + rot(k)*sin
    mulkcosNode.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mulkcosNode.inTensors = {&kLayer, &cos};
    mulkcosNode.outTensors = {&kcos};
    mulkcosNode.inTensorViewFuncs.resize(mulkcosNode.inTensors.size());

    mulrotksinNode.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mulrotksinNode.inTensors = {&kRotate, &sin};
    mulrotksinNode.outTensors = {&rotksin};
    mulrotksinNode.inTensorViewFuncs.resize(mulrotksinNode.inTensors.size());

    addkNode.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    addkNode.inTensors = {&kcos, &rotksin};
    addkNode.outTensors = {&interkEmbedded};

    AsdOps::OpParam::Transpose kTransposeNode1Param = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {2, 0, 1, 3}};
    kTransposeNode1.opDesc = {0, "TransposeOperation", kTransposeNode1Param};
    kTransposeNode1.inTensors = {&interkEmbedded};
    kTransposeNode1.outTensors = {&kEmbedded};
    kTransposeNode1.inTensorViewFuncs.resize(kTransposeNode1.inTensors.size());
}

PositionEmbedding1dOpsLlama7bRunner::~PositionEmbedding1dOpsLlama7bRunner() {}
} // namespace AclTransformer
