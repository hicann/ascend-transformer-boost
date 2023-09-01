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
#include "position_embedding_1d_split_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include "acltransformer/utils/tensor_util.h"

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 10;
namespace AclTransformer {
PositionEmbedding1dSplitOpsRunner::PositionEmbedding1dSplitOpsRunner(const PositionEmbedding1dSplitParam &param)
    : OpsRunner("PositionEmbedding1dSplitOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_1D_SPLIT), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbedding1dSplitOpsRunner::PositionEmbedding1dSplitOpsRunner called";
    
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &input = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    AsdOps::Tensor &inputEmbeddedPermuted = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &inputTransposed = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &cos = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &inputTransposed0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &inputTransposed1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &inputTransposed1Neg = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &inputRotate = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mul0 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mul1 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &inputEmbedded = kernelGraph_.internalTensors.at(internalTensorNum++);

    int64_t nodeNum = 0;
    auto &inputTransposeNode = kernelGraph_.nodes[nodeNum++];
    auto &embedding0Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding1Node = kernelGraph_.nodes[nodeNum++];
    auto &sliceNode = kernelGraph_.nodes[nodeNum++];
    auto &negNode = kernelGraph_.nodes[nodeNum++];
    auto &cat0Node = kernelGraph_.nodes[nodeNum++];
    auto &mul0Node = kernelGraph_.nodes[nodeNum++];
    auto &mul1Node = kernelGraph_.nodes[nodeNum++];
    auto &addNode = kernelGraph_.nodes[nodeNum++];
    auto &permuteNode = kernelGraph_.nodes[nodeNum++];

    AsdOps::OpParam::Transpose inputTransposeNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    inputTransposeNode.opDesc = {0, "TransposeOperation", inputTransposeNodeParam};
    inputTransposeNode.inTensors = {&input};
    inputTransposeNode.outTensors = {&inputTransposed};
    inputTransposeNode.inTensorViewFuncs.resize(inputTransposeNode.inTensors.size());
    inputTransposeNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                AsdOps::SVector<int64_t> &newDims) {
            newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
        };

    ViewFunc Squeeze01 = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        if (oldDims.at(0) == 1) {
            newDims.resize(oldDims.size() - 2);
            for (size_t i = 0; i < newDims.size(); i++) {
                newDims.at(i) = oldDims.at(i + 2);
            }
        } else {
            newDims = oldDims;
        }
    };

    embedding0Node.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embedding0Node.inTensors = {&cosTable, &positionIds};
    embedding0Node.outTensors = {&cos};
    embedding0Node.inTensorViewFuncs.resize(embedding0Node.inTensors.size());
    embedding0Node.inTensorViewFuncs.at(0) = Squeeze01;

    embedding1Node.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embedding1Node.inTensors = {&sinTable, &positionIds};
    embedding1Node.outTensors = {&sin};
    embedding1Node.inTensorViewFuncs.resize(embedding1Node.inTensors.size());
    embedding1Node.inTensorViewFuncs.at(0) = Squeeze01;

    InferShapePreFunc split1InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 2}});
    };
    sliceNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 2}};
    sliceNode.inTensors = {&inputTransposed};
    sliceNode.outTensors = {&inputTransposed0, &inputTransposed1};
    sliceNode.inferShapePreFunc = split1InferShape;

    negNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise{AsdOps::OpParam::Elewise::ElewiseType::ELEWISE_MULS, -1}};
    negNode.inTensors = {&inputTransposed1};
    negNode.outTensors = {&inputTransposed1Neg};

    InferShapePreFunc cat0InferShape = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "ConcatOperation", AsdOps::OpParam::Concat{int(dims.size()) - 1}});
    };
    cat0Node.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    cat0Node.inTensors = {&inputTransposed1Neg, &inputTransposed0};
    cat0Node.outTensors = {&inputRotate};
    cat0Node.inferShapePreFunc = cat0InferShape;

    // [bs, sq, 1, rd]
    ViewFunc unsqueezeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), 1, oldDims.at(1), oldDims.at(2)};
    };

    mul0Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul0Node.inTensors = {&inputTransposed, &cos};
    mul0Node.outTensors = {&mul0};
    mul0Node.inTensorViewFuncs.resize(mul0Node.inTensors.size());
    mul0Node.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    mul1Node.opDesc = {0, "BroadcastOperation",
                        AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_MUL}};
    mul1Node.inTensors = {&inputRotate, &sin};
    mul1Node.outTensors = {&mul1};
    mul1Node.inTensorViewFuncs.resize(mul1Node.inTensors.size());
    mul1Node.inTensorViewFuncs.at(1) = unsqueezeCosSinView;

    addNode.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast{AsdOps::OpParam::Broadcast::BroadcastType::BROADCAST_ADD}};
    addNode.inTensors = {&mul0, &mul1};
    addNode.outTensors = {&inputEmbedded};

    AsdOps::OpParam::Transpose permuteNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {2, 0, 1, 3}};
    permuteNode.opDesc = {0, "TransposeOperation", permuteNodeParam};
    permuteNode.inTensors = {&inputEmbedded};
    permuteNode.outTensors = {&inputEmbeddedPermuted};
    permuteNode.inTensorViewFuncs.resize(permuteNode.inTensors.size());
}

PositionEmbedding1dSplitOpsRunner::~PositionEmbedding1dSplitOpsRunner() {}
} // namespace AclTransformer
