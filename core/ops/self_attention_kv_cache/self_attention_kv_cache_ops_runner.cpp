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
#include "self_attention_kv_cache_ops_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionKvCacheOpsRunner::SelfAttentionKvCacheOpsRunner(const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsRunner"), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKVCacheOperation::SelfAttentionKVCacheOperation called";
}

SelfAttentionKvCacheOpsRunner::~SelfAttentionKvCacheOpsRunner() {}

AsdOps::Status SelfAttentionKvCacheOpsRunner::SetupKernelGraph(const VariantPack &variantPack)
{
    ASD_LOG(INFO) << GetName() << " SetupKernelGraph start: " << "transKey: " << param_.transKey
       << ",dk: " << param_.dk << ",headNum: " << param_.headNum << ",layerId: " << param_.layerId;

    kernelGraph_.inTensors = variantPack.inTensors;
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(0);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(1);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(2);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(3);
    AsdOps::Tensor &pastKey = kernelGraph_.inTensors.at(4);
    AsdOps::Tensor &pastValue = kernelGraph_.inTensors.at(5);

    kernelGraph_.outTensors = variantPack.outTensors;
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(0);
    AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(1);
    AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(2);

    kernelGraph_.internalTensors.resize(7);
    AsdOps::Tensor &divOut = kernelGraph_.internalTensors.at(0);
    AsdOps::Tensor &transposedQ = kernelGraph_.internalTensors.at(1);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(2);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(3);
    AsdOps::Tensor &maskOut = kernelGraph_.internalTensors.at(4);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(5);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(6);

    kernelGraph_.nodes.resize(10);
    auto &mulsQNode = kernelGraph_.nodes.at(0);
    auto &permuteQNode = kernelGraph_.nodes.at(1);
    auto &catKeyNode = kernelGraph_.nodes.at(2);
    auto &permuteKNode = kernelGraph_.nodes.at(3);
    auto &bmmQkNode = kernelGraph_.nodes.at(4);
    auto &maskNode = kernelGraph_.nodes.at(5);
    auto &mulsMaskOutNode = kernelGraph_.nodes.at(6);
    auto &softMaxNode = kernelGraph_.nodes.at(7);
    auto &catValueNode = kernelGraph_.nodes.at(8);
    auto &bmmVNode = kernelGraph_.nodes.at(9);

    float varAttr = 1.0 / (sqrt(param_.dk) * (param_.layerId + 1));
    mulsQNode.opDesc = {0, "ElewiseOperation",
        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, varAttr})};
    mulsQNode.inTensors = {&mixedQuery};
    mulsQNode.outTensors = {&divOut};

    permuteQNode.opDesc = {0, "AsStridedOperation"};
    permuteQNode.inTensors = {&mixedQuery};
    permuteQNode.outTensors = {&transposedQ};
    permuteQNode.inTensorViewFuncs[0] = 
    [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
    {
        newDims= {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };
    permuteQNode.inferShapePreFunc = 
    [&](AsdOps::RunInfo &runInfo)
    {
        // permute [1, 0, 2]
        AsdOps::SVector<int64_t> inputShape = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> inputStride;
        int64_t size = inputShape.size();
        inputStride.resize(size);
        int64_t stride = 1;
        for (size_t i = 0; i < size; i++) {
             inputStride.at(size - i - 1) = stride;
             stride *= inputShape.at(size - i - 1);
        }
        std::swap(inputShape[0], inputShape[1]);
        std::swap(inputStride[0], inputStride[1]);
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided({inputShape, inputStride, 0})});
    };

    catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    catKeyNode.inTensors = {&mixedKey, &pastKey};
    catKeyNode.outTensors = {&presentKey};

    permuteKNode.opDesc = {0, "AsStridedOperation"};
    permuteKNode.inTensors = {&presentKey};
    permuteKNode.outTensors = {&transposedK};
    permuteKNode.inTensorViewFuncs[0] = 
    [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
    {
        newDims= {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };
    permuteKNode.inferShapePreFunc = 
    [](AsdOps::RunInfo &runInfo)
    {
        // permute [1, 2, 0]
        AsdOps::SVector<int64_t> inputShape = runInfo.GetInTensor(0).desc.dims;
        AsdOps::SVector<int64_t> inputStride;
        int64_t size = inputShape.size();
        inputStride.resize(size);
        int64_t stride = 1;
        for (size_t i = 0; i < size; i++) {
             inputStride.at(size - i - 1) = stride;
             stride *= inputShape.at(size - i - 1);
        }
        std::swap(inputShape[0], inputShape[1]);
        std::swap(inputShape[1], inputShape[2]);
        std::swap(inputStride[0], inputStride[1]);
        std::swap(inputStride[1], inputStride[2]);
        runInfo.SetOpDesc({0, "AsStridedOperation", AsdOps::OpParam::AsStrided({inputShape, inputStride, 0})});
    };

    bmmQkNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, param_.transKey, {/*oriShape*/}})};
    bmmQkNode.inTensors = {&transposedQ, &transposedK};
    bmmQkNode.outTensors = {&bmmQkOut};

    maskNode.opDesc = {0, "BroadcastOperation", 
        AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_MASKEDFILL, {-10000.0}})};
    maskNode.inTensors = {&bmmQkOut, &attention_mask};
    maskNode.outTensors = {&maskOut};
    maskNode.inTensorViewFuncs[0] = 
    [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
    {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(2), oldDims.at(3)};
    };

    float scale = param_.layerId + 1.0;
    mulsMaskOutNode.opDesc = {0, "ElewiseOperation", 
        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, scale})};
    mulsMaskOutNode.inTensors = {&maskOut};
    mulsMaskOutNode.outTensors = {&attentionScores};

    softMaxNode.opDesc = {0, "NormOperation", 
        AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMaxNode.inTensors = {&attentionScores};
    softMaxNode.outTensors = {&attentionProbs};

    catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({0})};
    catValueNode.inTensors = {&mixedValue, &pastValue};
    catValueNode.outTensors = {&presentValue};

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&attentionProbs, &presentValue};
    bmmVNode.outTensors = {&context};
    bmmVNode.inTensorViewFuncs[1] = 
    [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
    {
        newDims= {oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
    };

    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
