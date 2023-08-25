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
#include "self_attention_kv_cache_ops_baichuan1_7b_runner_910a.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"
#include <asdops/utils/log/log.h>

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 15;
static const uint64_t NODE_COUNT = 18;
namespace AclTransformer {
SelfAttentionKvCacheOpsBaiChuan17bRunner910a::SelfAttentionKvCacheOpsBaiChuan17bRunner910a(const SelfAttentionKvCacheParam &param)
    : OpsRunner("SelfAttentionKvCacheOpsBaiChuan17bRunner910a", RUNNER_TYPE_SELF_ATTENTION_KV_CACHE), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionKvCacheOpsBaiChuan17bRunner910a::SelfAttentionKvCacheOpsBaiChuan17bRunner910a called, transKey:"
                  << param_.transKey << ", dk: " << param_.dk << ", headNum: " << param_.headNum
                  << ", layerId: " << param_.layerId;

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    kernelGraph_.internalTensors.resize(INTERMEDIATE_TENSOR_COUNT);
    kernelGraph_.nodes.resize(NODE_COUNT);

    int64_t inTensorNum = 0;
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(inTensorNum++);  // [bs, sq, hn, hs]
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(inTensorNum++);  // [bs, sq, hn, hs]
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(inTensorNum++);   // [bs, sq, hn, hs]
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(inTensorNum++);  // [1, 1, 1, seqPast + 1]
    AsdOps::Tensor &pastKey = kernelGraph_.inTensors.at(inTensorNum++);   // [bs, seqPast, hn, hs]
    AsdOps::Tensor &pastValue = kernelGraph_.inTensors.at(inTensorNum++);   // [bs, seqPast, hn, hs]

    int64_t outTensorNum = 0;
    AsdOps::Tensor &contextOut = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentKey = kernelGraph_.outTensors.at(outTensorNum++);
    AsdOps::Tensor &presentValue = kernelGraph_.outTensors.at(outTensorNum++);

    int64_t internalTensorNum = 0;
    AsdOps::Tensor &qScaledOutND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedQND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedKND = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &permutedQTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &transposedKTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOutNZ = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmQkOutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScores = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionScoresF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsF32 = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbs = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &attentionProbsTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &catValueOutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVOut = kernelGraph_.internalTensors.at(internalTensorNum++);
    AsdOps::Tensor &bmmVoutTransResult = kernelGraph_.internalTensors.at(internalTensorNum++);



    int64_t nodeId = 0;
    auto &catKeyNode = kernelGraph_.nodes.at(nodeId++);
    auto &catValueNode = kernelGraph_.nodes.at(nodeId++);
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    auto &permuteQNode = kernelGraph_.nodes.at(nodeId++);
    auto &permuteKNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataQNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataKNode = kernelGraph_.nodes.at(nodeId++);
    auto &bmmQKNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataQKNode = kernelGraph_.nodes.at(nodeId++);
    auto &addMaskNode = kernelGraph_.nodes.at(nodeId++);
    auto &castInNode = kernelGraph_.nodes.at(nodeId++);
    auto &softMaxNode = kernelGraph_.nodes.at(nodeId++);
    auto &castOutNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataProbsNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataVNode = kernelGraph_.nodes.at(nodeId++);
    auto &bmmVNode = kernelGraph_.nodes.at(nodeId++);
    auto &transdataBmmVNode = kernelGraph_.nodes.at(nodeId++);
    auto &transposeContext1Node = kernelGraph_.nodes.at(nodeId++);



    // cat key
    // key = torch.cat(key, pastKey)
    catKeyNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({1})};
    catKeyNode.inTensors = {&pastKey, &mixedKey};
    catKeyNode.outTensors = {&presentKey};
    catKeyNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    catValueNode.opDesc = {0, "ConcatOperation", AsdOps::OpParam::Concat({1})};
    catValueNode.inTensors = {&pastValue, &mixedValue};
    catValueNode.outTensors = {&presentValue};
    catValueNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    // scaling down q
    float scalingAttr = (1.0 / (sqrt(param_.dk)));
    ASD_LOG(INFO) << "Scaling down for query with scaling factor " << scalingAttr;
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, scalingAttr})};
    mulsQNode.inTensors = {&mixedQuery};
    mulsQNode.outTensors = {&qScaledOutND};
    mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    // trans [bs, sq, hn, hs] to [bs, hn, sq, hs]
    AsdOps::OpParam::Transpose permuteSeqHnParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0 ,2, 1, 3}};
    // Q = q.permute(0, 2, 1, 3)  // [bs, hn, sq, hs]
    permuteQNode.opDesc = {0, "TransposeOperation", permuteSeqHnParam};
    permuteQNode.inTensors = {&qScaledOutND};
    permuteQNode.outTensors = {&transposedQND};

    // trans [bs, sq, hn, hs] to [bs, hn, hs, sq]
    AsdOps::OpParam::Transpose permuteSeqHnHsParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0 ,2, 3, 1}};
    permuteKNode.opDesc = {0, "TransposeOperation", permuteSeqHnHsParam};
    permuteKNode.inTensors = {&presentKey};
    permuteKNode.outTensors = {&transposedKND};

    // trans to NZ
    transdataQNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&transposedQND};
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
    transdataKNode.inTensors = {&transposedKND};
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
    bmmQKNode.outTensors = {&bmmQkOutNZ};
    bmmQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        runInfo.SetOpDesc({0, "MatMulOperation",
                           AsdOps::OpParam::MatMul({false, false,
                                                    {orgQDims_.at(1), orgQDims_.at(2), orgKDims_.at(2)}})});
    };

    transdataQKNode.opDesc = {0, "TransdataOperation",
                              AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {0, 0}})};
    transdataQKNode.inTensors = {&bmmQkOutNZ};
    transdataQKNode.outTensors = {&bmmQkOutTransResult};
    transdataQKNode.inferShapePreFunc = [=](AsdOps::RunInfo &runInfo) {
        AsdOps::SVector<int64_t> transQKTargetDims = {orgQDims_.at(1), orgKDims_.at(2)};
        runInfo.SetOpDesc(
            {0, "TransdataOperation",
             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, transQKTargetDims})});
    };

    addMaskNode.opDesc = {0, "BroadcastOperation",
                          AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    addMaskNode.inTensors = {&attention_mask, &bmmQkOutTransResult};
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
    transdataVNode.inTensors = {&presentValue};
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

    transposeContext1Node.opDesc = {0, "TransposeOperation", permuteSeqHnParam};
    transposeContext1Node.inTensors = {&bmmVoutTransResult};
    transposeContext1Node.outTensors = {&contextOut};
    transposeContext1Node.inTensorViewFuncs.resize(transposeContext1Node.inTensors.size());
    transposeContext1Node.inTensorViewFuncs[0] = [&](const AsdOps::SVector<int64_t> &oldDims,
                                                     AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };
}

SelfAttentionKvCacheOpsBaiChuan17bRunner910a::~SelfAttentionKvCacheOpsBaiChuan17bRunner910a() {}

void SelfAttentionKvCacheOpsBaiChuan17bRunner910a::AsStrideKernelInferShapeSet(const AsdOps::SVector<int64_t> &sequence,
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