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
#include "self_attention_cross_ops_llama7badapter_runner_adapter_310p.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"
#include <asdops/utils/log/log.h>

static const uint64_t IN_TENSOR_COUNT = 9;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 32;
static const uint64_t NODE_COUNT = 35;
namespace AclTransformer {
SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p::SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p(const SelfAttentionCrossParam &param)
    : OpsRunner("SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p", RUNNER_TYPE_SELF_ATTENTION_CROSS), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p::SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p called "
                  << ", dk: " << param_.dk << ", headNum: " << param_.headNum << ", model: " << param_.model;

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &xq = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &xk = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &xv = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &keys = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &values = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &adapterV = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &adapterK = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &gateTanh = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    AsdOps::Tensor &contextOut = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &curKeysOut = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &curValuesOut = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &permutedQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedKeys = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedValues = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedAdapterV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedAdapterK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedQNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedKNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOutND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulsOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScoresF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &valueOutNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVOutND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &adapterKNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmASkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmASkOutND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulsOutAS = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulsOutASF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulsOutASProbsF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulsOutASProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &adapterScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &adapterScoresNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &adapterVNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmASVOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmASVOutND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &svAddAsv = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &svAddAsvView = kernelGraph_.internalTensors.at(internalTensorNum++);

    int64_t nodeNum = 0;
    auto &catKeyNode = kernelGraph_.nodes.at(nodeNum++); //0
    auto &catValueNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteKeyNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteValuesNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteAdapterVNode = kernelGraph_.nodes.at(nodeNum++);//5
    auto &permuteAdapterKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQNode = kernelGraph_.nodes.at(nodeNum++);//7
    auto &transdataKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmQKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &muls1Node = kernelGraph_.nodes.at(nodeNum++);
    auto &addMaskNode = kernelGraph_.nodes.at(nodeNum++);
    auto &castIn1Node = kernelGraph_.nodes.at(nodeNum++);
    auto &softMax1Node = kernelGraph_.nodes.at(nodeNum++);
    auto &castOut1Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataProbsNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataBmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataAKNode = kernelGraph_.nodes.at(nodeNum++);//20
    auto &bmmASNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataASNode = kernelGraph_.nodes.at(nodeNum++);
    auto &muls2Node = kernelGraph_.nodes.at(nodeNum++);
    auto &castIn2Node = kernelGraph_.nodes.at(nodeNum++);
    auto &softMax2Node = kernelGraph_.nodes.at(nodeNum++);
    auto &castOut2Node = kernelGraph_.nodes.at(nodeNum++);
    auto &mulsTanhNode = kernelGraph_.nodes.at(nodeNum++);//27
    auto &transdataAdapterProbsNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataAdapterVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmASVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataBmmASVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &addASVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeOutNode = kernelGraph_.nodes.at(nodeNum++);
    auto &reshapeOutNode = kernelGraph_.nodes.at(nodeNum++);

    catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({1})};
    catKeyNode.inTensors = {&keys, &xk};
    catKeyNode.outTensors = {&curKeysOut};

    catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({1})};
    catValueNode.inTensors = {&values, &xv};
    catValueNode.outTensors = {&curValuesOut};

    AsdOps::OpParam::Transpose permuteQParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    permuteQNode.opDesc = {0, "TransposeOperation", permuteQParam};
    permuteQNode.inTensors = {&xq};
    permuteQNode.outTensors = {&permutedQ};
    
    AsdOps::OpParam::Transpose permuteKParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 3, 1}};
    permuteKeyNode.opDesc = {0, "TransposeOperation", permuteKParam};
    permuteKeyNode.inTensors = {&curKeysOut};
    permuteKeyNode.outTensors = {&permutedKeys};

    AsdOps::OpParam::Transpose permuteVParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    permuteValuesNode.opDesc = {0, "TransposeOperation", permuteVParam};
    permuteValuesNode.inTensors = {&curValuesOut};
    permuteValuesNode.outTensors = {&permutedValues};

    AsdOps::OpParam::Transpose permuteAdapterVParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    permuteAdapterVNode.opDesc = {0, "TransposeOperation", permuteAdapterVParam};
    permuteAdapterVNode.inTensors = {&adapterV};
    permuteAdapterVNode.outTensors = {&permutedAdapterV};
    
    AsdOps::OpParam::Transpose permuteAdapterKParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 3, 1}};
    permuteAdapterKNode.opDesc = {0, "TransposeOperation", permuteAdapterKParam};
    permuteAdapterKNode.inTensors = {&adapterK};
    permuteAdapterKNode.outTensors = {&permutedAdapterK};

    transdataQNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&permutedQ};
    transdataQNode.outTensors = {&permutedQNZ};
    transdataQNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        orgQDims_ = runInfo.GetInTensor(0).desc.dims;
    };
    transdataQNode.inTensorViewFuncs.resize(transdataQNode.inTensors.size());
    transdataQNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    transdataKNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataKNode.inTensors = {&permutedKeys};
    transdataKNode.outTensors = {&transposedKNZ};
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
    bmmQKNode.inTensors = {&permutedQNZ, &transposedKNZ};
    bmmQKNode.outTensors = {&bmmQkOut};
    bmmQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgQDims_.at(1), orgQDims_.at(2), orgKDims_.at(2)}})});
    };

    transdataQKNode.opDesc = {0, "TransdataOperation",
                              AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataQKNode.inTensors = {&bmmQkOut};
    transdataQKNode.outTensors = {&bmmQkOutND};
    transdataQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transQKTargetDims = {orgQDims_.at(1), orgKDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transQKTargetDims})});
    };

    float varAttr = 1.0 / sqrt(param_.dk);
    muls1Node.opDesc = {0, "ElewiseOperation",
                       AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    muls1Node.inTensors = {&bmmQkOutND};
    muls1Node.outTensors = {&mulsOut};

    addMaskNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addMaskNode.inTensors = {&attention_mask, &mulsOut};
    addMaskNode.outTensors = {&attentionScores};
    addMaskNode.inTensorViewFuncs.resize(addMaskNode.inTensors.size());
    addMaskNode.inTensorViewFuncs[1] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    castIn1Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castIn1Node.inTensors = {&attentionScores};
    castIn1Node.outTensors = {&attentionScoresF32};

    softMax1Node.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMax1Node.inTensors = {&attentionScoresF32};
    softMax1Node.outTensors = {&attentionProbsF32};

    castOut1Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castOut1Node.inTensors = {&attentionProbsF32};
    castOut1Node.outTensors = {&attentionProbs};

    transdataProbsNode.opDesc = {0, "TransdataOperation",
                                 AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataProbsNode.inTensors = {&attentionProbs};
    transdataProbsNode.outTensors = {&attentionProbsNZ};
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
    transdataVNode.inTensors = {&permutedValues};
    transdataVNode.outTensors = {&valueOutNZ};
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
    bmmVNode.inTensors = {&attentionProbsNZ, &valueOutNZ};
    bmmVNode.outTensors = {&bmmVOut};
    bmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgProbsDims_.at(1), orgProbsDims_.at(2), orgVDims_.at(2)}})});

    };

    transdataBmmVNode.opDesc = {0, "TransdataOperation",
                                AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataBmmVNode.inTensors = {&bmmVOut};
    transdataBmmVNode.outTensors = {&bmmVOutND};
    transdataBmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transBmmTargetDims = {orgProbsDims_.at(1), orgVDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transBmmTargetDims})});
    };

    transdataAKNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataAKNode.inTensors = {&permutedAdapterK};
    transdataAKNode.outTensors = {&adapterKNZ};
    transdataAKNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgKDims_ = runInfo.GetInTensor(0).desc.dims;
    };
    transdataAKNode.inTensorViewFuncs.resize(transdataKNode.inTensors.size());
    transdataAKNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    bmmASNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmASNode.inTensors = {&permutedQNZ, &adapterKNZ};
    bmmASNode.outTensors = {&bmmASkOut};
    bmmASNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgQDims_.at(1), orgQDims_.at(2), orgKDims_.at(2)}})});
    };

    transdataASNode.opDesc = {0, "TransdataOperation",
                              AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataASNode.inTensors = {&bmmASkOut};
    transdataASNode.outTensors = {&bmmASkOutND};
    transdataASNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transQKTargetDims = {orgQDims_.at(1), orgKDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transQKTargetDims})});
    };

    muls2Node.opDesc = {0, "ElewiseOperation",
                       AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    muls2Node.inTensors = {&bmmASkOutND};
    muls2Node.outTensors = {&mulsOutAS};

    castIn2Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castIn2Node.inTensors = {&mulsOutAS};
    castIn2Node.outTensors = {&mulsOutASF32};

    softMax2Node.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMax2Node.inTensors = {&mulsOutASF32};
    softMax2Node.outTensors = {&mulsOutASProbsF32};

    castOut2Node.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castOut2Node.inTensors = {&mulsOutASProbsF32};
    castOut2Node.outTensors = {&mulsOutASProbs};

    mulsTanhNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MUL})};
    mulsTanhNode.inTensors = {&gateTanh, &mulsOutASProbs}; //[]
    mulsTanhNode.outTensors = {&adapterScores};
    mulsTanhNode.inTensorViewFuncs.resize(mulsTanhNode.inTensors.size());
    mulsTanhNode.inTensorViewFuncs[1] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    transdataAdapterProbsNode.opDesc = {0, "TransdataOperation",
                                 AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataAdapterProbsNode.inTensors = {&adapterScores};
    transdataAdapterProbsNode.outTensors = {&adapterScoresNZ};
    transdataAdapterProbsNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgProbsDims_ = runInfo.GetInTensor(0).desc.dims;
    };
    transdataAdapterProbsNode.inTensorViewFuncs.resize(transdataAdapterProbsNode.inTensors.size());
    transdataAdapterProbsNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    transdataAdapterVNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataAdapterVNode.inTensors = {&permutedAdapterV};
    transdataAdapterVNode.outTensors = {&adapterVNZ};
    transdataAdapterVNode.inferShapePreFunc = [&](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        orgVDims_ = runInfo.GetInTensor(0).desc.dims;
    };
    transdataAdapterVNode.inTensorViewFuncs.resize(transdataAdapterVNode.inTensors.size());
    transdataAdapterVNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    bmmASVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmASVNode.inTensors = {&adapterScoresNZ, &adapterVNZ};
    bmmASVNode.outTensors = {&bmmASVOut};
    bmmASVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgProbsDims_.at(1), orgProbsDims_.at(2), orgVDims_.at(2)}})});

    };

    transdataBmmASVNode.opDesc = {0, "TransdataOperation",
                                AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataBmmASVNode.inTensors = {&bmmASVOut};
    transdataBmmASVNode.outTensors = {&bmmASVOutND};
    transdataBmmASVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transBmmTargetDims = {orgProbsDims_.at(1), orgVDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transBmmTargetDims})});
    };

    addASVNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addASVNode.inTensors = {&bmmVOutND, &bmmASVOutND};
    addASVNode.outTensors = {&svAddAsv};

    AsdOps::OpParam::Transpose transposeOutParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeOutNode.opDesc = {0, "TransposeOperation", transposeOutParam};
    transposeOutNode.inTensors = {&svAddAsv};
    transposeOutNode.outTensors = {&svAddAsvView};
    transposeOutNode.inTensorViewFuncs.resize(transposeOutNode.inTensors.size());
    transposeOutNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    AsdOps::OpParam::Transpose reshapeOutParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 1, 2}};
    reshapeOutNode.opDesc = {0, "TransposeOperation", reshapeOutParam};
    reshapeOutNode.inTensors = {&svAddAsvView};
    reshapeOutNode.outTensors = {&contextOut};
    reshapeOutNode.inTensorViewFuncs.resize(reshapeOutNode.inTensors.size());
    reshapeOutNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

}

SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p::~SelfAttentionCrossOpsLlama7bAdapterRunnerAdapter310p() {}
} // namespace AclTransformer