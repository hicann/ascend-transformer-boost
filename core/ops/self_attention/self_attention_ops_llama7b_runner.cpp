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
#include "self_attention_ops_llama7b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionOpsLlama7bRunner::SelfAttentionOpsLlama7bRunner(const SelfAttentionParam &param)
    : OpsRunner("SelfAttentionOpsLlama7bRunner", RUNNER_TYPE_SELF_ATTENTION), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionOpsLlama7bRunner::SelfAttentionOpsLlama7bRunner called, transKey:"
                  << param_.transKey << ", dk: " << param_.dk << ", headNum: " << param_.headNum
                  << ", layerId: " << param_.layerId;
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors.resize(3);
    AsdOps::Tensor &contextOut = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(10);
    AsdOps::Tensor &permutedQ = kernelGraph_.internalTensors.at(0);
    AsdOps::Tensor &permutedK = kernelGraph_.internalTensors.at(1);
    AsdOps::Tensor &permutedV = kernelGraph_.internalTensors.at(2);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(3);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(4);
    AsdOps::Tensor &mulsOut = kernelGraph_.internalTensors.at(5);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(6);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(7);
    AsdOps::Tensor &bmmVOut = kernelGraph_.internalTensors.at(8);
    AsdOps::Tensor &contextTOut = kernelGraph_.internalTensors.at(9);

    kernelGraph_.nodes.resize(13);
    auto &permuteQNode = kernelGraph_.nodes.at(0);
    auto &permuteKNode = kernelGraph_.nodes.at(1);
    auto &permuteVNode = kernelGraph_.nodes.at(2);
    auto &permutePresentKNode = kernelGraph_.nodes.at(3);
    auto &permutePresentVNode = kernelGraph_.nodes.at(4);
    auto &transposePresentKNode = kernelGraph_.nodes.at(5);
    auto &bmmQKNode = kernelGraph_.nodes.at(6);
    auto &mulsNode = kernelGraph_.nodes.at(7);
    auto &addMaskNode = kernelGraph_.nodes.at(8);
    auto &softMaxNode = kernelGraph_.nodes.at(9);
    auto &bmmVNode = kernelGraph_.nodes.at(10);
    auto &transposeContext1Node = kernelGraph_.nodes.at(11);
    auto &transposeContext2Node = kernelGraph_.nodes.at(12);

    /* permute inputs */
    permuteQNode.opDesc = {0, "AsStridedOperation"};
    permuteQNode.inTensors = {&mixedQuery};
    permuteQNode.outTensors = {&permutedQ};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({1, 2, 0, 3}), permuteQNode);

    permuteKNode.opDesc = {0, "AsStridedOperation"};
    permuteKNode.inTensors = {&mixedKey};
    permuteKNode.outTensors = {&permutedK};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({1, 2, 0, 3}), permuteKNode);

    permuteVNode.opDesc = {0, "AsStridedOperation"};
    permuteVNode.inTensors = {&mixedValue};
    permuteVNode.outTensors = {&permutedV};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({1, 2, 0, 3}), permuteVNode);

    permutePresentKNode.opDesc = {0, "AsStridedOperation"};
    permutePresentKNode.inTensors = {&permutedK};
    permutePresentKNode.outTensors = {&presentKey};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({2, 0, 1, 3}), permutePresentKNode);

    permutePresentVNode.opDesc = {0, "AsStridedOperation"};
    permutePresentVNode.inTensors = {&permutedV};
    permutePresentVNode.outTensors = {&presentValue};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({2, 0, 1, 3}), permutePresentVNode);

    transposePresentKNode.opDesc = {0, "AsStridedOperation"};
    transposePresentKNode.inTensors = {&permutedK};
    transposePresentKNode.outTensors = {&transposedK};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({0, 1, 3, 2}), transposePresentKNode);

    bmmQKNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmQKNode.inTensors = {&permutedQ, &transposedK};
    bmmQKNode.outTensors = {&bmmQkOut};
    bmmQKNode.inTensorViewFuncs.resize(bmmQKNode.inTensors.size());
    bmmQKNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };
    bmmQKNode.inTensorViewFuncs[1] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    float varAttr = 1.0 / sqrt(param_.dk);
    mulsNode.opDesc = {0, "ElewiseOperation",
                       AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulsNode.inTensors = {&bmmQkOut};
    mulsNode.outTensors = {&mulsOut};

    addMaskNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addMaskNode.inTensors = {&attention_mask, &mulsOut};
    addMaskNode.outTensors = {&attentionScores};
    addMaskNode.inTensorViewFuncs.resize(addMaskNode.inTensors.size());
    addMaskNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    softMaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMaxNode.inTensors = {&attentionScores};
    softMaxNode.outTensors = {&attentionProbs};

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbs, &permutedV};
    bmmVNode.outTensors = {&bmmVOut};
    bmmVNode.inTensorViewFuncs.resize(bmmVNode.inTensors.size());
    bmmVNode.inTensorViewFuncs[1] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    transposeContext1Node.opDesc = {0, "AsStridedOperation"};
    transposeContext1Node.inTensors = {&bmmVOut};
    transposeContext1Node.outTensors = {&contextTOut};
    transposeContext1Node.inTensorViewFuncs.resize(transposeContext1Node.inTensors.size());
    transposeContext1Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({0, 2, 1, 3}), transposeContext1Node);

    transposeContext2Node.opDesc = {0, "AsStridedOperation"};
    transposeContext2Node.inTensors = {&contextTOut};
    transposeContext2Node.outTensors = {&contextOut};
    transposeContext2Node.inTensorViewFuncs.resize(transposeContext2Node.inTensors.size());
    transposeContext2Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    };
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({1, 0, 2}), transposeContext2Node);
}

SelfAttentionOpsLlama7bRunner::~SelfAttentionOpsLlama7bRunner() {}

void SelfAttentionOpsLlama7bRunner::AsStrideKernelInferShapeSet(const AsdOps::SVector<int64_t> &sequence,
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
