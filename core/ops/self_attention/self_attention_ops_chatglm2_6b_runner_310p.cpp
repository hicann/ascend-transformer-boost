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

#include "self_attention_ops_chatglm2_6b_runner_310p.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionOpsChatglm26bRunner310P::SelfAttentionOpsChatglm26bRunner310P(const SelfAttentionParam &param)
: OpsRunner("SelfAttentionOpsChatglm26bRunner310P", RUNNER_TYPE_SELF_ATTENTION), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionOpsChatglm26bRunner310P::SelfAttentionOpsChatglm26bRunner310P called";
    
    const int internalTensorSize = 17;
    const int nodeSize = 18;
    float maskValue = -65504;
    
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);

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
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedVTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVout = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVoutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &keyExpand = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &valueExpand = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(nodeSize);
    int64_t nodeNum = 0;
    auto &mulsQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &expandNode0 = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmQkNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &maskNode = kernelGraph_.nodes.at(nodeNum++);
    auto &mulsMaskOutNode = kernelGraph_.nodes.at(nodeNum++);
    auto &softMaxNode = kernelGraph_.nodes.at(nodeNum++);
    auto &expandNode1 = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataProbsNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataBmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteContextNode = kernelGraph_.nodes.at(nodeNum++);

    float varAttr = 1.0 / (sqrt(param_.dk) * param_.preScale);
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

    int64_t np = param_.numHeadsPerPartition;
    int64_t hn = param_.hiddenSizePerHead;
    int64_t gp = param_.numGroupsPerPartition;
    InferShapePreFunc expandInferShape = [np, gp](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> dims = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> asStridedDims = {dims.at(0), dims.at(1), dims.at(2), np / gp, dims.at(4)};
        AsdOps::SVector<int64_t> stride = {dims.at(1) * dims.at(2) * dims.at(4), dims.at(2) * dims.at(4), dims.at(4), 0, 1};
        AsdOps::SVector<int64_t> offset = {0};
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided{asStridedDims, stride, offset}});
    };
    expandNode0.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    expandNode0.inTensors = {&mixedKey};
    expandNode0.outTensors = {&keyExpand};
    expandNode0.inTensorViewFuncs.resize(expandNode0.inTensors.size());
    expandNode0.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2), 1, oldDims.at(3)};
    };
    expandNode0.inferShapePreFunc = expandInferShape;

    AsdOps::OpParam::Transpose permuteKNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 2, 0}};
    permuteKNode.opDesc = {0, "TransposeOperation", permuteKNodeParam};
    permuteKNode.inTensors = {&keyExpand};
    permuteKNode.outTensors = {&transposedK};
    permuteKNode.inTensorViewFuncs.resize(permuteKNode.inTensors.size());
    permuteKNode.inTensorViewFuncs[0] = [np, hn](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * np, hn};
    };

    transdataQNode.opDesc = {0, "TransdataOperation",
                            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&transposedQ};
    transdataQNode.outTensors = {&transposedQTransResult};
    transdataQNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        oriQDims_ = runInfo.GetInTensor(0).desc.dims;
    };

    transdataKNode.opDesc = {0, "TransdataOperation",
                            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataKNode.inTensors = {&transposedK};
    transdataKNode.outTensors = {&transposedKTransResult};
    transdataKNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        oriKDims_ = runInfo.GetInTensor(0).desc.dims;
    };

    bmmQkNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmQkNode.inTensors = {&transposedQTransResult, &transposedKTransResult};
    bmmQkNode.outTensors = {&bmmQkOut};
    bmmQkNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {oriQDims_.at(1), oriQDims_.at(2), oriKDims_.at(2)}})});
    };

    transdataQKNode.opDesc = {0, "TransdataOperation",
                            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataQKNode.inTensors = {&bmmQkOut};
    transdataQKNode.outTensors = {&bmmQkOutTransResult};
    transdataQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transQKTargetDims = {oriQDims_.at(1), oriKDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transQKTargetDims})});
    };

    maskNode.opDesc = {0, "BroadcastOperation",
                    AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MASKEDFILL, {maskValue}})};
    maskNode.inTensors = {&bmmQkOutTransResult, &attention_mask};
    maskNode.outTensors = {&maskOut};
    maskNode.inTensorViewFuncs.resize(maskNode.inTensors.size());
    maskNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    float scale = param_.postScale;
    mulsMaskOutNode.opDesc = {0, "ElewiseOperation",
                            AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, scale})};
    mulsMaskOutNode.inTensors = {&maskOut};
    mulsMaskOutNode.outTensors = {&attentionScores};

    softMaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMaxNode.inTensors = {&attentionScores};
    softMaxNode.outTensors = {&attentionProbs};

    expandNode1.opDesc = {0, "AsStridedOperation", AsdOps::OpParam::AsStrided{{}, {}, {}}};
    expandNode1.inTensors = {&mixedValue};
    expandNode1.outTensors = {&valueExpand};
    expandNode1.inTensorViewFuncs.resize(expandNode1.inTensors.size());
    expandNode1.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2), 1, oldDims.at(3)};
    };
    expandNode1.inferShapePreFunc = expandInferShape;

    AsdOps::OpParam::Transpose permuteVNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {1, 0, 2}};
    permuteVNode.opDesc = {0, "TransposeOperation", permuteVNodeParam};
    permuteVNode.inTensors = {&valueExpand};
    permuteVNode.outTensors = {&transposedV};
    permuteVNode.inTensorViewFuncs.resize(permuteVNode.inTensors.size());
    permuteVNode.inTensorViewFuncs[0] = [np, hn](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * np, hn};
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
        oriProbsDims_ = runInfo.GetInTensor(0).desc.dims;
    };

    transdataVNode.opDesc = {0, "TransdataOperation",
                            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataVNode.inTensors = {&transposedV};
    transdataVNode.outTensors = {&transposedVTransResult};
    transdataVNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        oriVDims_ = runInfo.GetInTensor(0).desc.dims;
    };

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbsTransResult, &transposedVTransResult};
    bmmVNode.outTensors = {&bmmVout};
    bmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {oriProbsDims_.at(1), oriProbsDims_.at(2), oriVDims_.at(2)}})});
    };

    transdataBmmVNode.opDesc = {0, "TransdataOperation",
                                AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataBmmVNode.inTensors = {&bmmVout};
    transdataBmmVNode.outTensors = {&bmmVoutTransResult};
    transdataBmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transBmmTargetDims = {oriProbsDims_.at(1), oriVDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transBmmTargetDims})});
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

SelfAttentionOpsChatglm26bRunner310P::~SelfAttentionOpsChatglm26bRunner310P() {}
} // namespace AclTransformer