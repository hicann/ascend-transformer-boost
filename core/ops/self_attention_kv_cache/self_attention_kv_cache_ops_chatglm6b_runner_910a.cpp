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
#include "self_attention_kv_cache_ops_chatglm6b_runner_910a.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheOpsChatGlm6bRunner910a::SelfAttentionKvCacheOpsChatGlm6bRunner910a(
    const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsChatGlm6bRunner910a", RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKvCacheOpsChatGlm6bRunner910a::SelfAttentionKvCacheOpsChatGlm6bRunner910a called"
                  << "transKey: " << param_.transKey << ",dk: " << param_.dk << ",headNum: " << param_.headNum
                  << ",layerId: " << param_.layerId;
    const int inTensorSize = 6;
    kernelGraph_.inTensors.resize(inTensorSize);
    int64_t inTensorNum = 0;
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastKey = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastValue = kernelGraph_.inTensors.at(inTensorNum++);

    const int outTensorSize = 3;
    kernelGraph_.outTensors.resize(outTensorSize);
    int64_t outTensorNum = 0;
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(outTensorNum++);

    const int internalTensorSize = 17;
    kernelGraph_.internalTensors.resize(internalTensorSize);
    int64_t internalTensorNum = 0;
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedQTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedKTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &maskOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &maskOutF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScoresF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedVTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVout = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVoutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);

    const int nodeSize = 20;
    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &mulsQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &catKeyNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmQkNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &maskNode = kernelGraph_.nodes.at(nodeNum++);
    auto &castInNode = kernelGraph_.nodes.at(nodeNum++);
    auto &mulsMaskOutNode = kernelGraph_.nodes.at(nodeNum++);
    auto &softMaxNode = kernelGraph_.nodes.at(nodeNum++);
    auto &castOutNode = kernelGraph_.nodes.at(nodeNum++);
    auto &catValueNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataProbsNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataBmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteContextNode = kernelGraph_.nodes.at(nodeNum++);

    float varAttr = 1.0 / (sqrt(param_.dk) * (param_.layerId + 1));
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulsQNode.inTensors = {&mixedQuery};
    mulsQNode.outTensors = {&divOut};
    mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    AsdOps::OpParam::Transpose permuteQNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
    permuteQNode.opDesc = {0, "TransposeOperation", permuteQNodeParam};
    permuteQNode.inTensors = {&divOut};
    permuteQNode.outTensors = {&transposedQ};
    permuteQNode.inTensorViewFuncs.resize(permuteQNode.inTensors.size());
    permuteQNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };

    catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    catKeyNode.inTensors = {&pastKey, &mixedKey};
    catKeyNode.outTensors = {&presentKey};
    catKeyNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    AsdOps::OpParam::Transpose permuteKNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 2, 0}};
    permuteKNode.opDesc = {0, "TransposeOperation", permuteKNodeParam};
    permuteKNode.inTensors = {&presentKey};
    permuteKNode.outTensors = {&transposedK};
    permuteKNode.inTensorViewFuncs.resize(permuteKNode.inTensors.size());
    permuteKNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };

    transdataQNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&transposedQ};
    transdataQNode.outTensors = {&transposedQTransResult};
    transdataQNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgQDims_ = runInfo.GetInTensor(0).desc.dims;
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

    bmmQkNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmQkNode.inTensors = {&transposedQTransResult, &transposedKTransResult};
    bmmQkNode.outTensors = {&bmmQkOut};
    bmmQkNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
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

    float maskValue = -10000.0;
    maskNode.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MASKEDFILL, {maskValue}})};
    maskNode.inTensors = {&bmmQkOutTransResult, &attention_mask};
    maskNode.outTensors = {&maskOut};
    maskNode.inTensorViewFuncs.resize(maskNode.inTensors.size());
    maskNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    castInNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castInNode.inTensors = {&maskOut};
    castInNode.outTensors = {&maskOutF32};

    float scale = param_.layerId + 1.0;
    mulsMaskOutNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, scale})};
    mulsMaskOutNode.inTensors = {&maskOutF32};
    mulsMaskOutNode.outTensors = {&attentionScoresF32};

    softMaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMaxNode.inTensors = {&attentionScoresF32};
    softMaxNode.outTensors = {&attentionProbsF32};

    castOutNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castOutNode.inTensors = {&attentionProbsF32};
    castOutNode.outTensors = {&attentionProbs};

    catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    catValueNode.inTensors = {&pastValue, &mixedValue};
    catValueNode.outTensors = {&presentValue};
    catValueNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    AsdOps::OpParam::Transpose permuteVNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
    permuteVNode.opDesc = {0, "TransposeOperation", permuteVNodeParam};
    permuteVNode.inTensors = {&presentValue};
    permuteVNode.outTensors = {&transposedV};
    permuteVNode.inTensorViewFuncs.resize(permuteVNode.inTensors.size());
    permuteVNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };

    transdataProbsNode.opDesc = {0, "TransdataOperation",
                                 AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataProbsNode.inTensors = {&attentionProbs};
    transdataProbsNode.outTensors = {&attentionProbsTransResult};
    transdataProbsNode.inTensorViewFuncs.resize(transdataProbsNode.inTensors.size());
    transdataProbsNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims,
                                                 AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };
    transdataProbsNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgProbsDims_ = runInfo.GetInTensor(0).desc.dims;
    };

    transdataVNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataVNode.inTensors = {&transposedV};
    transdataVNode.outTensors = {&transposedVTransResult};
    transdataVNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgVDims_ = runInfo.GetInTensor(0).desc.dims;
    };

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbsTransResult, &transposedVTransResult};
    bmmVNode.outTensors = {&bmmVout};
    bmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgProbsDims_.at(1), orgProbsDims_.at(2), orgVDims_.at(2)}})});

    };

    transdataBmmVNode.opDesc = {0, "TransdataOperation",
                                AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataBmmVNode.inTensors = {&bmmVout};
    transdataBmmVNode.outTensors = {&bmmVoutTransResult};
    transdataBmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transBmmTargetDims = {orgProbsDims_.at(1), orgVDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transBmmTargetDims})});
    };

    AsdOps::OpParam::Transpose permuteContextNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE,
                                                          {2, 0, 1, 3}};
    permuteContextNode.opDesc = {0, "TransposeOperation", permuteContextNodeParam};
    permuteContextNode.inTensors = {&bmmVoutTransResult};
    permuteContextNode.outTensors = {&context};
    permuteContextNode.inTensorViewFuncs.resize(permuteContextNode.inTensors.size());
    permuteContextNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                  AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };
}

SelfAttentionKvCacheOpsChatGlm6bRunner910a::~SelfAttentionKvCacheOpsChatGlm6bRunner910a() {}
} // namespace AclTransformer
