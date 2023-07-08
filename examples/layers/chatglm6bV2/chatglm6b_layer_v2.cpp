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
#include "chatglm6b_layer_v2.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/plan_builder.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/ffn_operation.h"

static constexpr int32_t OUT_TENSOR_NUM = 3;
static constexpr int32_t OUT_TENSOR_SECOND = 2;
static constexpr int32_t IN_TENSOR_SIZE = 19;
static constexpr int32_t OUT_TENSOR_SIZE = 3;
static constexpr int32_t INTERMEDIATE_TENSOR_SIZE = 10;
namespace AclTransformer {
ChatGlm6BLayerV2::ChatGlm6BLayerV2(const nlohmann::json &paramJson) : Layer("ChatGlm6BLayerV2", paramJson)
{
    BuildGraph();
    BuildPlan();
}

ChatGlm6BLayerV2::~ChatGlm6BLayerV2() {}

AsdOps::Status ChatGlm6BLayerV2::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensors.size() != 19) {
        return AsdOps::Status::FailStatus(1, "in tensor size != 19");
    }
    const AsdOps::Tensor &keyTensor = inTensors.at(17);
    const AsdOps::Tensor &ValueTensor = inTensors.at(18);

    outTensorDescs.resize(OUT_TENSOR_NUM);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = keyTensor.desc;
    outTensorDescs.at(1).dims.at(0) += 1;
    outTensorDescs.at(OUT_TENSOR_SECOND) = ValueTensor.desc;
    outTensorDescs.at(OUT_TENSOR_SECOND).dims.at(0) += 1;
    return AsdOps::Status::OkStatus();
}

void ChatGlm6BLayerV2::BuildGraph()
{ // in
    const uint64_t hiddenStates = 0;
    const uint64_t normWeight = 1;
    const uint64_t normBias = 2;
    const uint64_t qkvMixdWeight = 3;
    const uint64_t qkvMixdBias = 4;
    const uint64_t selfOutLinearWeight = 5;
    const uint64_t selfOutLinearBias = 6;
    const uint64_t selfOutNormWeight = 7;
    const uint64_t selfOutNormBias = 8;
    const uint64_t ffnLinearWeight = 9;
    const uint64_t ffnLinearBias = 10;
    const uint64_t ffnOutLinearWeight = 11;
    const uint64_t ffnOutLinearBias = 12;
    const uint64_t positionIds = 13;
    const uint64_t cosTable = 14;
    const uint64_t sinTable = 15;
    const uint64_t attentionMask = 16;
    const uint64_t pastKey = 17;
    const uint64_t pastValue = 18;
    // out
    const uint64_t glmBlockOut = 19;
    const uint64_t presentKey = 20;
    const uint64_t presentValue = 21;
    // intermiate
    const uint64_t inputNormOut = 22;
    const uint64_t mixedLinearOutQkv = 23;
    const uint64_t positionEmbedQ = 24;
    const uint64_t positionEmbedK = 25;
    const uint64_t value = 26;
    const uint64_t selfOut = 27;
    const uint64_t selfLinearOut = 28;
    const uint64_t selfAddNormOut = 29;
    const uint64_t ffnOut = 30;
    const uint64_t ffnLinearOut = 31;

    AclTransformer::NormParam inputNormParam;
    inputNormParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    AclTransformer::LinearParam mixdQkvLinearParam;
    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = paramJson_["headNum"].get<int>();
    AclTransformer::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.transKey = paramJson_["transKey"].get<bool>();
    selfAttentionKvCacheParam.dk = paramJson_["dk"].get<int>();
    selfAttentionKvCacheParam.headNum = positionEmbeddingParam.headNum;
    selfAttentionKvCacheParam.layerId = paramJson_["layerId"].get<int>();
    AclTransformer::LinearParam selfOutLinearParam;

    AclTransformer::AddNormParam SelfAddNormParam;
    SelfAddNormParam.layerNormEps = inputNormParam.layerNormEps;
    SelfAddNormParam.zoom_scale = paramJson_["ResidualAddScale"].get<float>();
    AclTransformer::FfnParam ffnParam;
    AclTransformer::LinearParam ffnLinearParam;
    AclTransformer::AddParam ffnResidualAddParam;
    ffnResidualAddParam.scale = SelfAddNormParam.zoom_scale;

    AclTransformer::NormOperation *inputNormOp = new AclTransformer::NormOperation(inputNormParam);
    AclTransformer::LinearOperation *mixdQkvLinearOp = new AclTransformer::LinearOperation(mixdQkvLinearParam);
    AclTransformer::PositionEmbeddingOperation *positionEmbeddingOp =
        new AclTransformer::PositionEmbeddingOperation(positionEmbeddingParam);
    AclTransformer::SelfAttentionKvCacheOperation *selfAttentionKvCacheOp =
        new AclTransformer::SelfAttentionKvCacheOperation(selfAttentionKvCacheParam);
    AclTransformer::LinearOperation *selfOutLinearOp = new AclTransformer::LinearOperation(selfOutLinearParam);
    AclTransformer::AddNormOperation *SelfAddNormParamOp = new AclTransformer::AddNormOperation(SelfAddNormParam);
    AclTransformer::FfnOperation *ffnOp = new AclTransformer::FfnOperation(ffnParam);
    AclTransformer::LinearOperation *ffnLinearOp = new AclTransformer::LinearOperation(ffnLinearParam);
    AclTransformer::AddOperation *ffnResidualAddOp = new AclTransformer::AddOperation(ffnResidualAddParam);

    opGraph_.inTensorSize = IN_TENSOR_SIZE;
    opGraph_.outTensorSize = OUT_TENSOR_SIZE;
    opGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_SIZE;
    opGraph_.nodes.resize(IN_TENSOR_SIZE - INTERMEDIATE_TENSOR_SIZE);

    AclTransformer::OperationGraphNode &inputNormNode = opGraph_.nodes.at(0);
    AclTransformer::OperationGraphNode &mixdQkvLinearNode = opGraph_.nodes.at(1);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = opGraph_.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = opGraph_.nodes.at(3);
    AclTransformer::OperationGraphNode &selfOutLinearNode = opGraph_.nodes.at(4);
    AclTransformer::OperationGraphNode &selfResidualAddNormNode = opGraph_.nodes.at(5);
    AclTransformer::OperationGraphNode &ffnNode = opGraph_.nodes.at(6);
    AclTransformer::OperationGraphNode &ffnLinearNode = opGraph_.nodes.at(7);
    AclTransformer::OperationGraphNode &ffnResidualAddNode = opGraph_.nodes.at(8);

    inputNormNode.operation = inputNormOp;
    inputNormNode.inTensorIds = {hiddenStates, normWeight, normBias};
    inputNormNode.outTensorIds = {inputNormOut};

    mixdQkvLinearNode.operation = mixdQkvLinearOp;
    mixdQkvLinearNode.inTensorIds = {inputNormOut, qkvMixdWeight, qkvMixdBias};
    mixdQkvLinearNode.outTensorIds = {mixedLinearOutQkv};

    positionEmbeddingNode.operation = positionEmbeddingOp;
    positionEmbeddingNode.inTensorIds = {mixedLinearOutQkv, positionIds, cosTable, sinTable};
    positionEmbeddingNode.outTensorIds = {positionEmbedQ, positionEmbedK, value};

    selfAttentionKvCacheNode.operation = selfAttentionKvCacheOp;
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedQ, positionEmbedK, value, attentionMask, pastKey, pastValue};
    selfAttentionKvCacheNode.outTensorIds = {selfOut, presentKey, presentValue};

    selfOutLinearNode.operation = selfOutLinearOp;
    selfOutLinearNode.inTensorIds = {selfOut, selfOutLinearWeight, selfOutLinearBias};
    selfOutLinearNode.outTensorIds = {selfLinearOut};

    selfResidualAddNormNode.operation = SelfAddNormParamOp;
    selfResidualAddNormNode.inTensorIds = {selfLinearOut, inputNormOut, selfOutNormWeight, selfOutNormBias};
    selfResidualAddNormNode.outTensorIds = {selfAddNormOut};

    ffnNode.operation = ffnOp;
    ffnNode.inTensorIds = {selfAddNormOut, ffnLinearWeight, ffnLinearBias};
    ffnNode.outTensorIds = {ffnOut};

    ffnLinearNode.operation = ffnLinearOp;
    ffnLinearNode.inTensorIds = {ffnOut, ffnOutLinearWeight, ffnOutLinearBias};
    ffnLinearNode.outTensorIds = {ffnLinearOut};

    ffnResidualAddNode.operation = ffnResidualAddOp;
    ffnResidualAddNode.inTensorIds = {selfAddNormOut, ffnLinearOut};
    ffnResidualAddNode.outTensorIds = {glmBlockOut};
}
} // namespace AclTransformer
