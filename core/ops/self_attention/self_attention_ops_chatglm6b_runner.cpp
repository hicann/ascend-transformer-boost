

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
#include "self_attention_ops_chatglm6b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionOpsChatglm6bRunner::SelfAttentionOpsChatglm6bRunner(const SelfAttentionParam &param)
    : OpsRunner("SelfAttentionOpsChatglm6bRunner", RUNNER_TYPE_SELF_ATTENTION), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionOpsChatglm6bRunner::SelfAttentionOpsChatglm6bRunner called"
                  << "transKey: " << param_.transKey << ",dk: " << param_.dk << ",headNum: " << param_.headNum
                  << ",layerId: " << param_.layerId;
    kernelGraph_.inTensors.resize(4);
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(3);

    kernelGraph_.outTensors.resize(1);
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);

    kernelGraph_.internalTensors.resize(11);
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(0);
    AsdOps::Tensor &transposedQ = kernelGraph_.internalTensors.at(1);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(2);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(3);
    AsdOps::Tensor &maskOut = kernelGraph_.internalTensors.at(4);
    AsdOps::Tensor &maskOutF32 = kernelGraph_.internalTensors.at(5);
    AsdOps::Tensor &attentionScoresF32 = kernelGraph_.internalTensors.at(6);
    AsdOps::Tensor &attentionProbsF32 = kernelGraph_.internalTensors.at(7);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(8);
    AsdOps::Tensor &transposedV = kernelGraph_.internalTensors.at(9);
    AsdOps::Tensor &bmmVout = kernelGraph_.internalTensors.at(10);

    kernelGraph_.nodes.resize(12);
    auto &mulsQNode = kernelGraph_.nodes.at(0);
    auto &permuteQNode = kernelGraph_.nodes.at(1);
    auto &permuteKNode = kernelGraph_.nodes.at(2);
    auto &bmmQkNode = kernelGraph_.nodes.at(3);
    auto &maskNode = kernelGraph_.nodes.at(4);
    auto &castInNode = kernelGraph_.nodes.at(5);
    auto &mulsMaskOutNode = kernelGraph_.nodes.at(6);
    auto &softMaxNode = kernelGraph_.nodes.at(7);
    auto &castOutNode = kernelGraph_.nodes.at(8);
    auto &permuteVNode = kernelGraph_.nodes.at(9);
    auto &bmmVNode = kernelGraph_.nodes.at(10);
    auto &permuteContextNode = kernelGraph_.nodes.at(11);

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

    permuteQNode.opDesc = {0, "AsStridedOperation"};
    permuteQNode.inTensors = {&divOut};
    permuteQNode.outTensors = {&transposedQ};
    permuteQNode.inTensorViewFuncs.resize(permuteQNode.inTensors.size());
    permuteQNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };
    permuteQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        // permute [1, 0, 2]
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        AsdOps::SVector<int64_t> inputShape = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> inputStride;
        int64_t size = inputShape.size();
        inputStride.resize(size);
        int64_t stride = 1;
        for (int64_t i = 0; i < size; i++) {
            inputStride.at(size - i - 1) = stride;
            stride *= inputShape.at(size - i - 1);
        }
        std::swap(inputShape[0], inputShape[1]);
        std::swap(inputStride[0], inputStride[1]);
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided({inputShape, inputStride, {0}})});
    };

    permuteKNode.opDesc = {0, "AsStridedOperation"};
    permuteKNode.inTensors = {&mixedKey};
    permuteKNode.outTensors = {&transposedK};
    permuteKNode.inTensorViewFuncs.resize(permuteKNode.inTensors.size());
    permuteKNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };
    permuteKNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        // permute [1, 2, 0]
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
        AsdOps::SVector<int64_t> inputShape = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> inputStride;
        int64_t size = inputShape.size();
        inputStride.resize(size);
        int64_t stride = 1;
        for (int64_t i = 0; i < size; i++) {
            inputStride.at(size - i - 1) = stride;
            stride *= inputShape.at(size - i - 1);
        }
        std::swap(inputShape[0], inputShape[1]);
        std::swap(inputShape[1], inputShape[2]);
        std::swap(inputStride[0], inputStride[1]);
        std::swap(inputStride[1], inputStride[2]);
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided({inputShape, inputStride, {0}})});
    };

    bmmQkNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmQkNode.inTensors = {&transposedQ, &transposedK};
    bmmQkNode.outTensors = {&bmmQkOut};

    float maskValue = -10000.0;
    maskNode.opDesc = {0, "BroadcastOperation",
                       AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MASKEDFILL, {maskValue}})};
    maskNode.inTensors = {&bmmQkOut, &attention_mask};
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

    permuteVNode.opDesc = {0, "AsStridedOperation"};
    permuteVNode.inTensors = {&mixedValue};
    permuteVNode.outTensors = {&transposedV};
    permuteVNode.inTensorViewFuncs.resize(permuteVNode.inTensors.size());
    permuteVNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };
    permuteVNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        // permute [1, 0, 2]
        AsdOps::SVector<int64_t> inputShape = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> inputStride;
        int64_t size = inputShape.size();
        inputStride.resize(size);
        int64_t stride = 1;
        for (int64_t i = 0; i < size; i++) {
            inputStride.at(size - i - 1) = stride;
            stride *= inputShape.at(size - i - 1);
        }
        std::swap(inputShape[0], inputShape[1]);
        std::swap(inputStride[0], inputStride[1]);
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided({inputShape, inputStride, {0}})});
    };

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbs, &transposedV};
    bmmVNode.outTensors = {&bmmVout};
    bmmVNode.inTensorViewFuncs.resize(bmmVNode.inTensors.size());
    bmmVNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };
    bmmVNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    permuteContextNode.opDesc = {0, "AsStridedOperation"};
    permuteContextNode.inTensors = {&bmmVout};
    permuteContextNode.outTensors = {&context};
    permuteContextNode.inTensorViewFuncs.resize(permuteContextNode.inTensors.size());
    permuteContextNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                  AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };
    permuteContextNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        // permute [2, 0, 1, 3]
        AsdOps::SVector<int64_t> inputShape = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> inputStride;
        int64_t size = inputShape.size();
        inputStride.resize(size);
        int64_t stride = 1;
        for (int64_t i = 0; i < size; i++) {
            inputStride.at(size - i - 1) = stride;
            stride *= inputShape.at(size - i - 1);
        }
        std::swap(inputShape[1], inputShape[2]);
        std::swap(inputShape[0], inputShape[1]);
        std::swap(inputStride[1], inputStride[2]);
        std::swap(inputStride[0], inputStride[1]);
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided({inputShape, inputStride, {0}})});
    };

}

SelfAttentionOpsChatglm6bRunner::~SelfAttentionOpsChatglm6bRunner() {}
} // namespace AclTransformer
