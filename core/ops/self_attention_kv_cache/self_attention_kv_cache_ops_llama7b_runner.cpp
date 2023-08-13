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
#include "self_attention_kv_cache_ops_llama7b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 16;
static const uint64_t NODE_COUNT = 19;

namespace AclTransformer {
SelfAttentionKvCacheOpsLlama7bRunner::SelfAttentionKvCacheOpsLlama7bRunner(const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsLlama7bRunner", RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKvCacheOpsLlama7bRunner::SelfAttentionKvCacheOpsLlama7bRunner called, transKey:"
                  << param_.transKey << ", dk: " << param_.dk << ", headNum: " << param_.headNum
                  << ", layerId: " << param_.layerId;

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastKey = kernelGraph_.inTensors.at(inTensorNum++);
    AsdOps::Tensor &pastValue = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    AsdOps::Tensor &contextOut = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &permutedQ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedPastK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedPastV = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &catKeyOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &catValueOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &mulsOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScoresF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &contextTOut = kernelGraph_.internalTensors.at(internalTensorNum++);

    int64_t nodeNum = 0;
    auto &permuteQNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permuteVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permutePastKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permutePastVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &catKeyNode = kernelGraph_.nodes.at(nodeNum++);
    auto &catValueNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permutePresentKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &permutePresentVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transposePresentKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmQKNode = kernelGraph_.nodes.at(nodeNum++);
    auto &mulsNode = kernelGraph_.nodes.at(nodeNum++);
    auto &addMaskNode = kernelGraph_.nodes.at(nodeNum++);
    auto &castInNode = kernelGraph_.nodes.at(nodeNum++);
    auto &softMaxNode = kernelGraph_.nodes.at(nodeNum++);
    auto &castOutNode = kernelGraph_.nodes.at(nodeNum++);
    auto &bmmVNode = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeContext1Node = kernelGraph_.nodes.at(nodeNum++);
    auto &transposeContext2Node = kernelGraph_.nodes.at(nodeNum++);

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

    permutePastKNode.opDesc = {0, "AsStridedOperation"};
    permutePastKNode.inTensors = {&pastKey};
    permutePastKNode.outTensors = {&permutedPastK};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({1, 2, 0, 3}), permutePastKNode);

    permutePastVNode.opDesc = {0, "AsStridedOperation"};
    permutePastVNode.inTensors = {&pastValue};
    permutePastVNode.outTensors = {&permutedPastV};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({1, 2, 0, 3}), permutePastVNode);

    catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({2})};
    catKeyNode.inTensors = {&permutedPastK, &permutedK};
    catKeyNode.outTensors = {&catKeyOut};

    permutePresentKNode.opDesc = {0, "AsStridedOperation"};
    permutePresentKNode.inTensors = {&catKeyOut};
    permutePresentKNode.outTensors = {&presentKey};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({2, 0, 1, 3}), permutePresentKNode);

    catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({2})};
    catValueNode.inTensors = {&permutedPastV, &permutedV};
    catValueNode.outTensors = {&catValueOut};

    permutePresentVNode.opDesc = {0, "AsStridedOperation"};
    permutePresentVNode.inTensors = {&catValueOut};
    permutePresentVNode.outTensors = {&presentValue};
    AsStrideKernelInferShapeSet(AsdOps::SVector<int64_t>({2, 0, 1, 3}), permutePresentVNode);

    transposePresentKNode.opDesc = {0, "AsStridedOperation"};
    transposePresentKNode.inTensors = {&catKeyOut};
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

    castInNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castInNode.inTensors = {&attentionScores};
    castInNode.outTensors = {&attentionScoresF32};

    softMaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMaxNode.inTensors = {&attentionScoresF32};
    softMaxNode.outTensors = {&attentionProbsF32};

    castOutNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castOutNode.inTensors = {&attentionProbsF32};
    castOutNode.outTensors = {&attentionProbs};

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbs, &catValueOut};
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

SelfAttentionKvCacheOpsLlama7bRunner::~SelfAttentionKvCacheOpsLlama7bRunner() {}

void SelfAttentionKvCacheOpsLlama7bRunner::AsStrideKernelInferShapeSet(const AsdOps::SVector<int64_t> &sequence,
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