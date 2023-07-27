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
SelfAttentionKvCacheOpsChatGlm6bRunner910a::SelfAttentionKvCacheOpsChatGlm6bRunner910a(const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsChatGlm6bRunner910a", RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKvCacheOpsChatGlm6bRunner910a::SelfAttentionKvCacheOpsChatGlm6bRunner910a called"
                  << "transKey: " << param_.transKey << ",dk: " << param_.dk << ",headNum: " << param_.headNum
                  << ",layerId: " << param_.layerId;
    kernelGraph_.inTensors.resize(6);
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(3);
    AsdOps::Tensor &pastKey = kernelGraph_.inTensors.at(4);
    AsdOps::Tensor &pastValue = kernelGraph_.inTensors.at(5);

    kernelGraph_.outTensors.resize(3);
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(17);
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(0);
    AsdOps::Tensor &transposedQ = kernelGraph_.internalTensors.at(1);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(2);
    AsdOps::Tensor &transposedQTransResult = kernelGraph_.internalTensors.at(3);
    AsdOps::Tensor &transposedKTransResult = kernelGraph_.internalTensors.at(4);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(5);
    AsdOps::Tensor &bmmQkOutTransResult = kernelGraph_.internalTensors.at(6);
    AsdOps::Tensor &maskOut = kernelGraph_.internalTensors.at(7);
    AsdOps::Tensor &maskOutF32 = kernelGraph_.internalTensors.at(8);
    AsdOps::Tensor &attentionScoresF32 = kernelGraph_.internalTensors.at(9);
    AsdOps::Tensor &attentionProbsF32 = kernelGraph_.internalTensors.at(10);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(11);
    AsdOps::Tensor &transposedV = kernelGraph_.internalTensors.at(12);
    AsdOps::Tensor &attentionProbsTransResult = kernelGraph_.internalTensors.at(13);
    AsdOps::Tensor &transposedVTransResult = kernelGraph_.internalTensors.at(14);
    AsdOps::Tensor &bmmVout = kernelGraph_.internalTensors.at(15);
    AsdOps::Tensor &bmmVoutTransResult = kernelGraph_.internalTensors.at(16);

    kernelGraph_.nodes.resize(20);
    auto &mulsQNode = kernelGraph_.nodes.at(0);
    auto &permuteQNode = kernelGraph_.nodes.at(1);
    auto &catKeyNode = kernelGraph_.nodes.at(2);
    auto &permuteKNode = kernelGraph_.nodes.at(3);
    auto &transdataQNode = kernelGraph_.nodes.at(4);
    auto &transdataKNode = kernelGraph_.nodes.at(5);
    auto &bmmQkNode = kernelGraph_.nodes.at(6);
    auto &transdataQKNode = kernelGraph_.nodes.at(7);
    auto &maskNode = kernelGraph_.nodes.at(8);
    auto &castInNode = kernelGraph_.nodes.at(9);
    auto &mulsMaskOutNode = kernelGraph_.nodes.at(10);
    auto &softMaxNode = kernelGraph_.nodes.at(11);
    auto &castOutNode = kernelGraph_.nodes.at(12);
    auto &catValueNode = kernelGraph_.nodes.at(13);
    auto &permuteVNode = kernelGraph_.nodes.at(14);
    auto &transdataProbsNode = kernelGraph_.nodes.at(15);
    auto &transdataVNode = kernelGraph_.nodes.at(16);
    auto &bmmVNode = kernelGraph_.nodes.at(17);
    auto &transdataBmmVNode = kernelGraph_.nodes.at(18);
    auto &permuteContextNode = kernelGraph_.nodes.at(19);

    AsdOps::SVector<int64_t> orgQDims;
    AsdOps::SVector<int64_t> orgKDims;
    AsdOps::SVector<int64_t> orgProbsDims;
    AsdOps::SVector<int64_t> orgVDims;


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
        orgQDims = runInfo.GetInTensor(0).desc.dims;
    };

    transdataKNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataKNode.inTensors = {&transposedK};
    transdataKNode.outTensors = {&transposedKTransResult};
    transdataKNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgKDims = runInfo.GetInTensor(0).desc.dims;
    };

    bmmQkNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmQkNode.inTensors = {&transposedQTransResult, &transposedKTransResult};
    bmmQkNode.outTensors = {&bmmQkOut};
    bmmQkNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc(0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgQDims.at(0), orgQDims.at(1), orgQDims.at(2), orgKDims.at(2)}}));
    };

    transdataQKNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataQKNode.inTensors = {&bmmQkOut};
    transdataQKNode.outTensors = {&bmmQkOutTransResult};
    transdataQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transQKTargetDims = {orgQDims.at(0), orgQDims.at(1), orgKDims.at(2)};
        runInfo.SetOpDesc(0, "TransdataOperation", 
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND , transQKTargetDims}));
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
    transdataProbsNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgProbsDims = runInfo.GetInTensor(0).desc.dims;
    };

    transdataVNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataVNode.inTensors = {&transposedV};
    transdataVNode.outTensors = {&transposedVTransResult};
    transdataVNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgVDims = runInfo.GetInTensor(0).desc.dims;
    };

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbsTransResult, &transposedVTransResult};
    bmmVNode.outTensors = {&bmmVout};
    bmmVNode.inTensorViewFuncs.resize(bmmVNode.inTensors.size());
    bmmVNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };
    bmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc(0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgProbsDims.at(0), orgProbsDims.at(1), orgProbsDims.at(2), orgVDims.at(2)}}));

    };

    transdataBmmVNode.opDesc = {
        0, "TransdataOperation",
        AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataBmmVNode.inTensors = {&bmmVout};
    transdataBmmVNode.outTensors = {&bmmVoutTransResult};
    transdataBmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transBmmTargetDims = {orgProbsDims.at(0), orgProbsDims.at(1), orgVDims.at(2)};
        runInfo.SetOpDesc(0, "TransdataOperation", 
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND , transBmmTargetDims}));
    };

    AsdOps::OpParam::Transpose permuteContextNodeParam =
        {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {2, 0, 1, 3}};
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
