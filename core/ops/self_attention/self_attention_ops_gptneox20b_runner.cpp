

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
#include "self_attention_ops_gptneox20b_runner.h"
#include <numeric>
#include <cmath>
#include <asdops/utils/log/log.h>
#include <asdops/params/params.h>
#include <asdops/utils/svector/svector.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
SelfAttentionOpsGptNeox20bRunner::SelfAttentionOpsGptNeox20bRunner(const SelfAttentionParam &param)
    : OpsRunner("SelfAttentionOpsGptNeox20bRunner", RUNNER_TYPE_SELF_ATTENTION), param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionOpsGptNeox20bRunner::SelfAttentionOpsGptNeox20bRunner called"
                  << "transKey: " << param_.transKey << ",dk: " << param_.dk << ",headNum: " << param_.headNum
                  << ",layerId: " << param_.layerId;
    kernelGraph_.inTensors.resize(4);
    int64_t inTensorId = 0;
    AsdOps::Tensor &mixedQuery = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &mixedKey = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &mixedValue = kernelGraph_.inTensors.at(inTensorId++);
    AsdOps::Tensor &attention_mask = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(1);
    int64_t outTensorId = 0;
    AsdOps::Tensor &context = kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.internalTensors.resize(10);
    int64_t internalTensorId = 0;
    AsdOps::Tensor &qScaledOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &kScaledOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &transposedK = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &bmmQkOut = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &castFP32Out = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &scalingUpFP32Out = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &maskedFP32Out = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &softmaxFP32Out = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &softmaxFP16Out = kernelGraph_.internalTensors.at(internalTensorId++);
    AsdOps::Tensor &bmmVout = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(11);
    int64_t nodeId = 0;
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    auto &mulsKNode = kernelGraph_.nodes.at(nodeId++);
    auto &permuteKNode = kernelGraph_.nodes.at(nodeId++);
    auto &bmmQkNode = kernelGraph_.nodes.at(nodeId++);

    auto &castAttnToFP32 = kernelGraph_.nodes.at(nodeId++);
    auto &scalingUpNode = kernelGraph_.nodes.at(nodeId++);
    auto &attnMaskNode = kernelGraph_.nodes.at(nodeId++);
    auto &softMaxNode = kernelGraph_.nodes.at(nodeId++);
    auto &castAttnToFp16 = kernelGraph_.nodes.at(nodeId++);

    auto &bmmVNode = kernelGraph_.nodes.at(nodeId++);
    auto &permuteContextNode = kernelGraph_.nodes.at(nodeId++);

    // scaling down q
    float scalingAttr0 = (1.0 / (sqrt(param_.dk))) * param_.scalingFactor;
    ASD_LOG(INFO) << "Scaling down for query with scaling factor " << scalingAttr0;
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, scalingAttr0})};
    mulsQNode.inTensors = {&mixedQuery};
    mulsQNode.outTensors = {&qScaledOut};
    mulsQNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    float scalingAttr1 = param_.scalingFactor;
    ASD_LOG(INFO) << "Scaling down for key with scaling factor " << scalingAttr1;
    mulsKNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, scalingAttr1})};
    mulsKNode.inTensors = {&mixedKey};
    mulsKNode.outTensors = {&kScaledOut};
    mulsKNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    // key = key.view(bs * hn, sq, hs)
    // key = key.permute(0, 2, 1) // [bs * hn, hs, sq]
    AsdOps::OpParam::Transpose permuteKNodeParam = {AsdOps::OpParam::Transpose::TransposeType::TRANSPOSE, {0 ,2, 1}};
    permuteKNode.opDesc = {0, "TransposeOperation", permuteKNodeParam};
    permuteKNode.inTensors = {&kScaledOut};
    permuteKNode.outTensors = {&transposedK};
    permuteKNode.inTensorViewFuncs.resize(permuteKNode.inTensors.size());
    permuteKNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };

    // [bs * hn, sq, sq]
    bmmQkNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmQkNode.inTensors = {&qScaledOut, &transposedK};
    bmmQkNode.outTensors = {&bmmQkOut};
    bmmQkNode.inTensorViewFuncs.resize(permuteKNode.inTensors.size());
    bmmQkNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };
    bmmQkNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    castAttnToFP32.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castAttnToFP32.inTensors = {&bmmQkOut};
    castAttnToFP32.outTensors = {&castFP32Out};
    castAttnToFP32.inTensorViewFuncs.resize(castAttnToFP32.inTensors.size());
    castAttnToFP32.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };

    float scalingAttr2 = 1.0 / (param_.scalingFactor * param_.scalingFactor);
    scalingUpNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, scalingAttr2})};
    scalingUpNode.inTensors = {&castFP32Out};
    scalingUpNode.outTensors = {&scalingUpFP32Out};
    scalingUpNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    attnMaskNode.opDesc = {0, "BroadcastOperation", AsdOps::OpParam::Broadcast({AsdOps::OpParam::Broadcast::BROADCAST_ADD})};
    attnMaskNode.inTensors = {&scalingUpFP32Out, &attention_mask};
    attnMaskNode.outTensors = {&maskedFP32Out};
    attnMaskNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(0).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    softMaxNode.opDesc = {0, "NormOperation", AsdOps::OpParam::Norm({AsdOps::OpParam::Norm::NORM_SOFTMAX, {-1}})};
    softMaxNode.inTensors = {&maskedFP32Out};
    softMaxNode.outTensors = {&softmaxFP32Out};

    castAttnToFp16.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_CAST})};
    castAttnToFp16.inTensors = {&softmaxFP32Out};
    castAttnToFp16.outTensors = {&softmaxFP16Out};

    bmmVNode.opDesc = {0, "MatMulOperation", AsdOps::OpParam::MatMul({false, false, {/*oriShape*/}})};
    bmmVNode.inTensors = {&softmaxFP16Out, &mixedValue};
    bmmVNode.outTensors = {&bmmVout};
    bmmVNode.inTensorViewFuncs.resize(bmmVNode.inTensors.size());
    bmmVNode.inTensorViewFuncs[0] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };
    bmmVNode.inTensorViewFuncs[1] = [](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)};
    };
    bmmVNode.inferShapePreFunc = [](AsdOps::RunInfo &runInfo) {
        for (size_t i = 0; i < runInfo.GetInTensorCount(); i++) {
            runInfo.GetInTensor(i).desc.format = AsdOps::TENSOR_FORMAT_ND;
        }
    };

    AsdOps::OpParam::Transpose permuteContextNodeParam = {AsdOps::OpParam::Transpose::TRANSPOSE, {0 ,2, 1, 3}};
    permuteContextNode.opDesc = {0, "TransposeOperation", permuteContextNodeParam};
    permuteContextNode.inTensors = {&bmmVout};
    permuteContextNode.outTensors = {&context};
    permuteContextNode.inTensorViewFuncs.resize(permuteContextNode.inTensors.size());
    permuteContextNode.inTensorViewFuncs[0] = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                  AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0) / param_.headNum, param_.headNum, oldDims.at(1), oldDims.at(2)};
    };
}

SelfAttentionOpsGptNeox20bRunner::~SelfAttentionOpsGptNeox20bRunner() {}
} // namespace AclTransformer
