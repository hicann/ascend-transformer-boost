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
#include "position_embedding_fusion_ops_gptneox20b_runner.h"
#include <numeric>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/params/rope.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PositionEmbeddingFusionOpsGptNeox20bRunner::PositionEmbeddingFusionOpsGptNeox20bRunner(const PositionEmbeddingFusionParam &param)
    : OpsRunner("PositionEmbeddingFusionOpsGptNeox20bRunner", RUNNER_TYPE_POSITION_EMBEDDING_2D_MIXED_FUSION), param_(param)
{
    ASD_LOG(INFO) << "PositionEmbeddingFusionOpsGptNeox20bRunner::PositionEmbeddingFusionOpsGptNeox20bRunner called, headNum: "
                  << param_.headNum;
    kernelGraph_.inTensors.resize(5);
    int64_t inTensorId = 0;
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(inTensorId++);  // [bs, sq, 3 * all_hs]
    AsdOps::Tensor &positionIds = kernelGraph_.inTensors.at(inTensorId++);  // [bs, sq]
    AsdOps::Tensor &cosTable = kernelGraph_.inTensors.at(inTensorId++);  // [max_sq, rd]
    AsdOps::Tensor &sinTable = kernelGraph_.inTensors.at(inTensorId++);  // [max_sq, rd]
    AsdOps::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorId++);    // [sq]

    kernelGraph_.outTensors.resize(3);
    int64_t outTensorId = 0;
    AsdOps::Tensor &qEmbedded = kernelGraph_.outTensors.at(outTensorId++);
    AsdOps::Tensor &kEmbedded = kernelGraph_.outTensors.at(outTensorId++);
    AsdOps::Tensor &value = kernelGraph_.outTensors.at(outTensorId++);  // V

    kernelGraph_.internalTensors.resize(10);
    int64_t internalTensorId = 0;
    AsdOps::Tensor &querySplit = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keySplit = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRot = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryPass = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRot = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyPass = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &cosEmbed = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &sinEmbed = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &queryRotEmbed = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &keyRotEmbed = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(10);
    int64_t nodeId = 0;
    auto &splitQKVNode = kernelGraph_.nodes[nodeId++];
    auto &qRotSliceNode = kernelGraph_.nodes[nodeId++];
    auto &qPassSliceNode = kernelGraph_.nodes[nodeId++];
    auto &kRotSliceNode = kernelGraph_.nodes[nodeId++];
    auto &kPassSliceNode = kernelGraph_.nodes[nodeId++];
    auto &cosEmbedNode = kernelGraph_.nodes[nodeId++];
    auto &sinEmbedNode = kernelGraph_.nodes[nodeId++];
    auto &ropeNode = kernelGraph_.nodes[nodeId++];
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

    //
    ViewFunc ropeQKVView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        // [bs * sq, hn * hs]
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
    ViewFunc ropeCosSinView = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
    };
    // out [bs * sq, hn * rd]
    const size_t ropeViewSize = 4;
    AsdOps::OpParam::Rope ropeParam;
    ropeParam.ropeType = AsdOps::OpParam::Rope::ROPEND;
    ropeParam.rotaryCoeff = 2;
    ropeNode.opDesc = {0, "RopeOperation", ropeParam};
    ropeNode.inTensors = {&queryRot, &keyRot, &cosEmbed, &sinEmbed, &seqLen};
    ropeNode.outTensors = {&queryRotEmbed, &keyRotEmbed};
    ropeNode.inTensorViewFuncs.resize(ropeViewSize);
    ropeNode.inTensorViewFuncs.at(0) = ropeQKVView;
    ropeNode.inTensorViewFuncs.at(1) = ropeQKVView;
    ropeNode.inTensorViewFuncs.at(2) = ropeCosSinView;
    ropeNode.inTensorViewFuncs.at(3) = ropeCosSinView;
    ropeNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        runInfo.GetInTensor(4).desc.dtype = AsdOps::TENSOR_DTYPE_UINT32;
    };

    ViewFunc catQKVView = [&](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        // old [bs * sq, hn * rd]
        int64_t tmpBatchSize = kernelGraph_.inTensors.at(0).desc.dims[0];
        int64_t tmpSeqLen = kernelGraph_.inTensors.at(0).desc.dims[1];
        newDims = {tmpBatchSize, tmpSeqLen, param_.headNum, oldDims.at(1) / param_.headNum};
    };
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
    qCatNode.inTensors = {&queryRot, &queryPass};
    qCatNode.outTensors = {&qEmbedded};
    qCatNode.inTensorViewFuncs.resize(qCatNode.inTensors.size());
    qCatNode.inTensorViewFuncs.at(0) = catQKVView;
    qCatNode.inferShapePreFunc = catLastDimInferShape;

    kCatNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat{0}};
    kCatNode.inTensors = {&keyRot, &keyPass};
    kCatNode.outTensors = {&kEmbedded};
    kCatNode.inTensorViewFuncs.resize(kCatNode.inTensors.size());
    kCatNode.inTensorViewFuncs.at(0) = catQKVView;
    kCatNode.inferShapePreFunc = catLastDimInferShape;
}

PositionEmbeddingFusionOpsGptNeox20bRunner::~PositionEmbeddingFusionOpsGptNeox20bRunner() {}
} // namespace AclTransformer
