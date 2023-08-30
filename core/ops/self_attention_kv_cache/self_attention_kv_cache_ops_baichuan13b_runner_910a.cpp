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
#include "self_attention_kv_cache_ops_baichuan13b_runner_910a.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"
#include <asdops/utils/log/log.h>

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 25;
static const uint64_t NODE_COUNT = 26;
namespace AclTransformer {
SelfAttentionKvCacheOpsBaiChuan13bRunner910a::SelfAttentionKvCacheOpsBaiChuan13bRunner910a(const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsBaiChuan13bRunner910a", RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKvCacheOpsBaiChuan13bRunner910a::SelfAttentionKvCacheOpsBaiChuan13bRunner910a called, transKey:"
                  << param_.transKey << ", dk: " << param_.dk << ", headNum: " << param_.headNum
                  << ", layerId: " << param_.layerId;

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &mixedQkv = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastKey = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastValue = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    AsdOps::Tensor &contextOut = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &mixedQuery = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mixedKey = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mixedValue = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permuteQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permuteK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permuteV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutePastK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutePastV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &catKeyOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &catValueOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposeK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedQTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedKTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulsOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScoresF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &catValueOutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVoutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &contextTOut = kernelGraph_.internalTensors.at(internalTensorNum++);

    int64_t nodeId = 0;
    auto &splitQkvNode = kernelGraph_.nodes.at(nodeId++);
    auto &permuteQNode = kernelGraph_.nodes.at(nodeId++);
    auto &permuteKNode = kernelGraph_.nodes.at(nodeId++);
    auto &permuteVNode = kernelGraph_.nodes.at(nodeId++);
    auto &permutePastKNode = kernelGraph_.nodes.at(nodeId++);
    auto &permutePastVNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataKNode = kernelGraph_.nodes.at(nodeId++);
    auto &catKeyNode = kernelGraph_.nodes.at(nodeId++);
    auto &catValueNode = kernelGraph_.nodes.at(nodeId++);
    auto &permutePresentKNode = kernelGraph_.nodes.at(nodeId++);
    auto &permutePresentVNode = kernelGraph_.nodes.at(nodeId++);
    auto &transposePresentKNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataQNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataKNode = kernelGraph_.nodes.at(nodeId++);
    auto &bmmQKNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataQKNode = kernelGraph_.nodes.at(nodeId++);
    auto &mulsNode = kernelGraph_.nodes.at(nodeId++);
    auto &addMaskNode = kernelGraph_.nodes.at(nodeId++);
    auto &castInNode = kernelGraph_.nodes.at(nodeId++);
    auto &softmaxNode = kernelGraph_.nodes.at(nodeId++);
    auto &castOutNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataProbsNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataVNode = kernelGraph_.nodes.at(nodeId++);
    auto &bmmVNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataBmmVNode = kernelGraph_.nodes.at(nodeId++);
    auto &transposeContext1Node = kernelGraph_.nodes.at(nodeId++);
    auto &transposeContext2Node = kernelGraph_.nodes.at(nodeId++);

    // split QKV
    splitQkvNode.opDesc = {0, "SplitOperation", AsdOps::OpParam::Split{0, 3}};
    splitQkvNode.inTensors = {&mixedQkv};
    splitQkvNode.outTensors = {&mixedQuery, &mixedKey, &mixedValue};
    splitQkvNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        runInfo.SetOpDesc({0, "SplitOperation", AsdOps::OpParam::Split{int(dims.size()) - 1, 3}});
    };

    AsdOps::OpParam::Transpose permuteQParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    permuteQNode.opDesc = {0, "TransposeOperation", permuteQParam};
    permuteQNode.inTensors = {&mixedQuery};
    permuteQNode.outTensors = {&permutedQ};
    permuteQNode.inTensorViewFuncs.resize(permuteQNode.inTensors.size());
    permuteQNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };

    AsdOps::OpParam::Transpose permuteKParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    permuteKNode.opDesc = {0, "TransposeOperation", permuteKParam};
    permuteKNode.inTensors = {&mixedKey};
    permuteKNode.outTensors = {&permutedK};
    permuteKNode.inTensorViewFuncs.resize(permuteKNode.inTensors.size());
    permuteKNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };

    AsdOps::OpParam::Transpose permuteVParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    permuteVNode.opDesc = {0, "TransposeOperation", permuteVParam};
    permuteVNode.inTensors = {&mixedValue};
    permuteVNode.outTensors = {&permutedV};
    permuteVNode.inTensorViewFuncs.resize(permuteVNode.inTensors.size());
    permuteVNode.inTensorViewFuncs.at(0) = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), param_.headNum, oldDims.at(2) / param_.headNum};
    };

    AsdOps::OpParam::Transpose permutePastKParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 2, 0, 3}};
    permutePastKNode.opDesc = {0, "TransposeOperation", permutePastKParam};
    permutePastKNode.inTensors = {&pastKey};
    permutePastKNode.outTensors = {&permutedPastK};

    AsdOps::OpParam::Transpose permutePastVParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 2, 0, 3}};
    permutePastVNode.opDesc = {0, "TransposeOperation", permutePastVParam};
    permutePastVNode.inTensors = {&pastValue};
    permutePastVNode.outTensors = {&permutedPastV};

    catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({2})};
    catKeyNode.inTensors = {&permutedPastK, &permutedK};
    catKeyNode.outTensors = {&catKeyOut};

    AsdOps::OpParam::Transpose permutePresentKParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {2, 0, 1, 3}};
    permutePresentKNode.opDesc = {0, "TransposeOperation", permutePresentKParam};
    permutePresentKNode.inTensors = {&catKeyOut};
    permutePresentKNode.outTensors = {&presentKey};

    catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({2})};
    catValueNode.inTensors = {&permutedPastV, &permutedV};
    catValueNode.outTensors = {&catValueOut};

    AsdOps::OpParam::Transpose permutePresentVNParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {2, 0, 1, 3}};
    permutePresentVNode.opDesc = {0, "TransposeOperation", permutePresentVNParam};
    permutePresentVNode.inTensors = {&catValueOut};
    permutePresentVNode.outTensors = {&presentValue};

    AsdOps::OpParam::Transpose transposePresentKParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 1, 3, 2}};
    transposePresentKNode.opDesc = {0, "TransposeOperation", transposePresentKParam};
    transposePresentKNode.inTensors = {&catKeyOut};
    transposePresentKNode.outTensors = {&transposedK};

    transdataQNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&permutedQ};
    transdataQNode.outTensors = {&permutedQTransResult};
    transdataQNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgQDims_ = runInfo.GetInTensor(0).desc.dims;
    };
    transdataQNode.inTensorViewFuncs.resize(transdataQNode.inTensors.size());
    transdataQNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    transdataKNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataKNode.inTensors = {&transposedK};
    transdataKNode.outTensors = {&transposedKTransResult};
    transdataKNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgKDims_ = runInfo.GetInTensor(0).desc.dims;
    };
    transdataKNode.inTensorViewFuncs.resize(transdataKNode.inTensors.size());
    transdataKNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    bmmQKNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmQKNode.inTensors = {&permutedQTransResult, &transposedKTransResult};
    bmmQKNode.outTensors = {&bmmQkOut};
    bmmQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgQDims_.at(1), orgQDims_.at(2), orgKDims_.at(2)}})});
    };

    transdataQKNode.opDesc = {0, "TransdataOperation",
                              AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataQKNode.inTensors = {&bmmQkOut};
    transdataQKNode.outTensors = {&bmmQkOutTransResult};
    transdataQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transQKTargetDims = {orgQDims_.at(1), orgKDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transQKTargetDims})});
    };

    float varAttr = 1.0 / sqrt(param_.dk);
    mulsNode.opDesc = {0, "ElewiseOperation",
                       AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulsNode.inTensors = {&bmmQkOutTransResult};
    mulsNode.outTensors = {&mulsOut};

    addMaskNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addMaskNode.inTensors = {&attention_mask, &mulsOut};
    addMaskNode.outTensors = {&attentionScores};
    addMaskNode.inTensorViewFuncs.resize(addMaskNode.inTensors.size());
    addMaskNode.inTensorViewFuncs[1] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    castInNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castInNode.inTensors = {&attentionScores};
    castInNode.outTensors = {&attentionScoresF32};

    softMaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMaxNode.inTensors = {&attentionScoresF32};
    softMaxNode.outTensors = {&attentionProbsF32};

    castOutNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castOutNode.inTensors = {&attentionProbsF32};
    castOutNode.outTensors = {&attentionProbs};

    transdataProbsNode.opDesc = {0, "TransdataOperation",
                                 AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataProbsNode.inTensors = {&attentionProbs};
    transdataProbsNode.outTensors = {&attentionProbsTransResult};
    transdataProbsNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgProbsDims_ = runInfo.GetInTensor(0).desc.dims;
    };
    transdataProbsNode.inTensorViewFuncs.resize(transdataProbsNode.inTensors.size());
    transdataProbsNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    transdataVNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataVNode.inTensors = {&catValueOut};
    transdataVNode.outTensors = {&catValueOutTransResult};
    transdataVNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgVDims_ = runInfo.GetInTensor(0).desc.dims;
    };
    transdataVNode.inTensorViewFuncs.resize(transdataVNode.inTensors.size());
    transdataVNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbsTransResult, &catValueOutTransResult};
    bmmVNode.outTensors = {&bmmVOut};
    bmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgProbsDims_.at(1), orgProbsDims_.at(2), orgVDims_.at(2)}})});

    };

    transdataBmmVNode.opDesc = {0, "TransdataOperation",
                                AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataBmmVNode.inTensors = {&bmmVOut};
    transdataBmmVNode.outTensors = {&bmmVoutTransResult};
    transdataBmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transBmmTargetDims = {orgProbsDims_.at(1), orgVDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transBmmTargetDims})});
    };

    AsdOps::OpParam::Transpose transposeContext1Param = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeContext1Node.opDesc = {0, "TransposeOperation", transposeContext1Param};
    transposeContext1Node.inTensors = {&bmmVoutTransResult};
    transposeContext1Node.outTensors = {&contextTOut};
    transposeContext1Node.inTensorViewFuncs.resize(transposeContext1Node.inTensors.size());
    transposeContext1Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };
    
    AsdOps::OpParam::Transpose transposeContext2Param = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
    transposeContext2Node.opDesc = {0, "TransposeOperation", transposeContext2Param};
    transposeContext2Node.inTensors = {&contextTOut};
    transposeContext2Node.outTensors = {&contextOut};
    transposeContext2Node.inTensorViewFuncs.resize(transposeContext2Node.inTensors.size());
    transposeContext2Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
}

SelfAttentionKvCacheOpsBaiChuan13bRunner910a::~SelfAttentionKvCacheOpsBaiChuan13bRunner910a() {}
} // namespace AclTransformer