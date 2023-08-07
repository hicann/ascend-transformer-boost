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
#include "position_embedding_1d_split_fusion_ops_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/rope.h>
#include "acltransformer/utils/tensor_util.h"

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;
namespace AclTransformer {
PositionEmbedding1dFusionOpsRunner::PositionEmbedding1dFusionOpsRunner(const PositionEmbedding1dFusionParam &param)
    : OpsRunner("PositionEmbedding1dFusionOpsRunner", RUNNER_TYPE_POSITION_EMBEDDING_1D_FUSION), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbedding1dFusionOpsRunner::PositionEmbedding1dFusionOpsRunner called";

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &qLayer = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &kLayer = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &cos = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &sin = kernelGraph_.internalTensors.at(internalTensorNum++);

    int64_t nodeNum = 0;
    auto &embedding0Node = kernelGraph_.nodes[nodeNum++];
    auto &embedding1Node = kernelGraph_.nodes[nodeNum++];
    auto &ropeNode = kernelGraph_.nodes[nodeNum++];

    ViewFunc squeeze01ViewFunc = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
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
    embedding0Node.inTensorViewFuncs.at(0) = squeeze01ViewFunc;

    embedding1Node.opDesc = {0, "GatherOperation",
                             AsdOps::OpParam::Gather{AsdOps::OpParam::Gather::GatherType::GATHER_V2, 0, {0}}};
    embedding1Node.inTensors = {&sinTable, &positionIds};
    embedding1Node.outTensors = {&sin};
    embedding1Node.inTensorViewFuncs.resize(embedding1Node.inTensors.size());
    embedding1Node.inTensorViewFuncs.at(0) = squeeze01ViewFunc;

    ViewFunc ropeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims.resize(2);
        newDims.at(0) = oldDims.at(0) * oldDims.at(1);
        newDims.at(1) = oldDims.at(2); 
    }; 

    ropeNode.opDesc = {0, "RopeOperation", AsdOps::OpParam::Rope({AsdOps::OpParam::Rope::ROPEND, 2})};
    ropeNode.inTensors = {&qLayer, &kLayer, &cos, &sin, &seqLen};
    ropeNode.outTensors = {&qEmbedded, &kEmbedded};
    ropeNode.inTensorViewFuncs.resize(ropeNode.inTensors.size());
    ropeNode.inTensorViewFuncs.at(2) = ropeCosSinView;
    ropeNode.inTensorViewFuncs.at(3) = ropeCosSinView;
    ropeNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        runInfo.GetInTensor(4).desc.dtype = AsdOps::TENSOR_DTYPE_UINT32;
    };
}

PositionEmbedding1dFusionOpsRunner::~PositionEmbedding1dFusionOpsRunner() {}
} // namespace AclTransformer
