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
#include "self_attention_cross_ops_llama7badapter_runner_310p.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"
#include <asdops/utils/log/log.h>

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 17;
static const uint64_t NODE_COUNT = 20;
namespace AclTransformer {
SelfAttentionCrossOpsLlama7bAdapterRunner310p::SelfAttentionCrossOpsLlama7bAdapterRunner310p(const SelfAttentionCrossParam &param)
    : OpsRunner("SelfAttentionCrossOpsLlama7bAdapterRunner310p", RUNNER_TYPE_SELF_ATTENTION_CROSS), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionCrossOpsLlama7bAdapterRunner310p::SelfAttentionCrossOpsLlama7bAdapterRunner310p called "
                  << ", dk: " << param_.dk << ", headNum: " << param_.headNum;


    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &xq = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &xk = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &xv = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &Keys = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &Values = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    AsdOps::Tensor &contextOut = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &curKeysOut = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &curValuesOut = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &permutedQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedKeys = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedValues = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedQNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedKNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOutNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulsOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScoresF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &valuesNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVoutND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &contextTOut = kernelGraph_.internalTensors.at(internalTensorNum++);

    int64_t nodeNum = 0;
    auto &catKeyNode = kernelGraph_.nodes.at(nodeNum++);
    auto &catValueNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteKeysNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteValuesNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmQKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataQKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &mulsNode = kernelGraph_.nodes.at(nodeNum++);
    auto &addMaskNode = kernelGraph_.nodes.at(nodeNum++);
    auto &castInNode = kernelGraph_.nodes.at(nodeNum++);
    auto &softMaxNode = kernelGraph_.nodes.at(nodeNum++);
    auto &castOutNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataProbsNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transdataBmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeOutputNode = kernelGraph_.nodes.at(nodeNum++);
    auto &reshapeOutputNode = kernelGraph_.nodes.at(nodeNum++);

    catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({1})};
    catKeyNode.inTensors = {&Keys, &xk};
    catKeyNode.outTensors = {&curKeysOut};

    catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({1})};
    catValueNode.inTensors = {&Values, &xv};
    catValueNode.outTensors = {&curValuesOut};

    AsdOps::OpParam::Transpose permuteQParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    permuteQNode.opDesc = {0, "TransposeOperation", permuteQParam};
    permuteQNode.inTensors = {&xq};
    permuteQNode.outTensors = {&permutedQ}; 
    permuteQNode.inTensorViewFuncs.resize(permuteQNode.inTensors.size());

    AsdOps::OpParam::Transpose permutePastParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 3, 1}};
    permuteKeysNode.opDesc = {0, "TransposeOperation", permutePastParam};
    permuteKeysNode.inTensors = {&curKeysOut};
    permuteKeysNode.outTensors = {&permutedKeys};
    permuteKeysNode.inTensorViewFuncs.resize(permuteKeysNode.inTensors.size());

    AsdOps::OpParam::Transpose permuteValuesNParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    permuteValuesNode.opDesc = {0, "TransposeOperation", permuteValuesNParam};
    permuteValuesNode.inTensors = {&curValuesOut};
    permuteValuesNode.outTensors = {&permutedValues};
    permuteValuesNode.inTensorViewFuncs.resize(permuteValuesNode.inTensors.size());

    transdataQNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&permutedQ};
    transdataQNode.outTensors = {&permutedQNZ};
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
    transdataQKNode.outTensors = {&bmmQkOutNZ};
    transdataQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transQKTargetDims = {orgQDims_.at(1), orgKDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transQKTargetDims})});
    };

    float varAttr = 1.0 / sqrt(param_.dk);
    mulsNode.opDesc = {0, "ElewiseOperation",
                       AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulsNode.inTensors = {&bmmQkOutNZ};
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
    transdataProbsNode.outTensors = {&attentionProbsND};
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
    transdataVNode.outTensors = {&valuesNZ};
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
    bmmVNode.inTensors = {&attentionProbsND, &valuesNZ};
    bmmVNode.outTensors = {&bmmVOut};
    bmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation", 
                            AsdOps::OpParam::MatMul({false, false, 
                                                        {orgProbsDims_.at(1), orgProbsDims_.at(2), orgVDims_.at(2)}})});

    };

    transdataBmmVNode.opDesc = {0, "TransdataOperation",
                                AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataBmmVNode.inTensors = {&bmmVOut};
    transdataBmmVNode.outTensors = {&bmmVoutND};
    transdataBmmVNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transBmmTargetDims = {orgProbsDims_.at(1), orgVDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transBmmTargetDims})});
    };

    AsdOps::OpParam::Transpose transposeOutputParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 2, 1, 3}};
    transposeOutputNode.opDesc = {0, "TransposeOperation", transposeOutputParam};
    transposeOutputNode.inTensors = {&bmmVoutND};
    transposeOutputNode.outTensors = {&contextTOut};
    transposeOutputNode.inTensorViewFuncs.resize(transposeOutputNode.inTensors.size());
    transposeOutputNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };
    
    AsdOps::OpParam::Transpose reshapeOutputParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0, 1, 2}};
    reshapeOutputNode.opDesc = {0, "TransposeOperation", reshapeOutputParam};
    reshapeOutputNode.inTensors = {&contextTOut};
    reshapeOutputNode.outTensors = {&contextOut};
    reshapeOutputNode.inTensorViewFuncs.resize(reshapeOutputNode.inTensors.size());
    reshapeOutputNode.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };

}

SelfAttentionCrossOpsLlama7bAdapterRunner310p::~SelfAttentionCrossOpsLlama7bAdapterRunner310p() {}

void SelfAttentionCrossOpsLlama7bAdapterRunner310p::AsStrideKernelInferShapeSet(const AsdOps::SVector<int64_t> &sequence,
                                                                       KernelGraphNode &node)
{
    node.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> inputShapeOrig = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> inputStrideOrig;
        AsdOps::SVector<int64_t> inputShape;
        AsdOps::SVector<int64_t> inputStride;

        uint64_t size = inputShapeOrig.size();
        if (sequence.size() != size) {
            ASD_LOG(ERROR) << "AsStride config size error: " << size << " -> " << sequence.size();
            return;
        }
        inputStrideOrig.resize(size);
        uint64_t stride = 1;
        for (size_t i = 0; i < size; i++) {
            inputStrideOrig.at(size - i - 1) = stride;
            stride *= inputShapeOrig.at(size - i - 1);
        }
        for (size_t i = 0; i < size; i++) {
            inputShape.push_back(inputShapeOrig[sequence[i]]);
            inputStride.push_back(inputStrideOrig[sequence[i]]);
        }
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided({inputShape, inputStride, {0}})});
    };
}
} // namespace AclTransformer