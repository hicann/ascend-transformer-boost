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
#include "chatglm6b_fusion_layer.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/plan_builder.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
ChatGlm6BFusionLayer::ChatGlm6BFusionLayer(const nlohmann::json &paramJson) : Layer("ChatGlm6BFusionLayer", paramJson)
{
    BuildGraph();
    BuildPlan();
}

ChatGlm6BFusionLayer::~ChatGlm6BFusionLayer() {}

AsdOps::Status ChatGlm6BFusionLayer::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensors.size() != 22) {
        return AsdOps::Status::FailStatus(1, "in tensor size != 22");
    }

    outTensorDescs.resize(1);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

void ChatGlm6BFusionLayer::BuildGraph()
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
    const uint64_t cacheK = 17;
    const uint64_t cacheV = 18;
    const uint64_t seqLen = 19;
    const uint64_t tokenOffset = 20;
    const uint64_t layerId = 21;
    // out
    const uint64_t glmBlockOut = 22;
    // intermiate
    const uint64_t inputNormOut = 23;
    const uint64_t mixedLinearOutQkv = 24;
    const uint64_t positionEmbedQ = 25;
    const uint64_t positionEmbedK = 26;
    const uint64_t value = 27;
    const uint64_t selfOut = 28;
    const uint64_t selfLinearOut = 29;
    const uint64_t selfResidualAddOut = 30;
    const uint64_t selfNormOut = 31;
    const uint64_t ffnOut = 32;
    const uint64_t ffnLinearOut = 33;

    AclTransformer::NormParam inputNormParam;
    inputNormParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    AclTransformer::LinearParam mixdQkvLinearParam;
    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = paramJson_["headNum"].get<int>();
    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = paramJson_["headNum"].get<int>();
    AclTransformer::LinearParam selfOutLinearParam;
    AclTransformer::AddParam selfResidualAddParam;
    selfResidualAddParam.scale = paramJson_["ResidualAddScale"].get<float>();
    AclTransformer::NormParam selfNormParam;
    selfNormParam.layerNormEps = inputNormParam.layerNormEps;
    AclTransformer::FfnParam ffnParam;
    AclTransformer::LinearParam ffnLinearParam;
    AclTransformer::AddParam ffnResidualAddParam;
    ffnResidualAddParam.scale = selfResidualAddParam.scale;

    AclTransformer::NormOperation *inputNormOp = new AclTransformer::NormOperation(inputNormParam);
    AclTransformer::LinearOperation *mixdQkvLinearOp = new AclTransformer::LinearOperation(mixdQkvLinearParam);
    AclTransformer::PositionEmbeddingOperation *positionEmbeddingOp =
        new AclTransformer::PositionEmbeddingOperation(positionEmbeddingParam);
    AclTransformer::SelfAttentionKvCacheFusionOperation *selfAttentionKvCacheOp =
        new AclTransformer::SelfAttentionKvCacheFusionOperation(selfAttentionKvCacheParam);
    AclTransformer::LinearOperation *selfOutLinearOp = new AclTransformer::LinearOperation(selfOutLinearParam);
    AclTransformer::AddOperation *selfResidualAddOp = new AclTransformer::AddOperation(selfResidualAddParam);
    AclTransformer::NormOperation *selfNormOp = new AclTransformer::NormOperation(selfNormParam);
    AclTransformer::FfnOperation *ffnOp = new AclTransformer::FfnOperation(ffnParam);
    AclTransformer::LinearOperation *ffnLinearOp = new AclTransformer::LinearOperation(ffnLinearParam);
    AclTransformer::AddOperation *ffnResidualAddOp = new AclTransformer::AddOperation(ffnResidualAddParam);

    opGraph_.inTensorSize = 22;
    opGraph_.outTensorSize = 1;
    opGraph_.intermediateTensorSize = 11;
    opGraph_.nodes.resize(10);

    AclTransformer::OperationGraphNode &inputNormNode = opGraph_.nodes.at(0);
    AclTransformer::OperationGraphNode &mixdQkvLinearNode = opGraph_.nodes.at(1);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = opGraph_.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = opGraph_.nodes.at(3);
    AclTransformer::OperationGraphNode &selfOutLinearNode = opGraph_.nodes.at(4);
    AclTransformer::OperationGraphNode &selfResidualAddNode = opGraph_.nodes.at(5);
    AclTransformer::OperationGraphNode &selfNormNode = opGraph_.nodes.at(6);
    AclTransformer::OperationGraphNode &ffnNode = opGraph_.nodes.at(7);
    AclTransformer::OperationGraphNode &ffnLinearNode = opGraph_.nodes.at(8);
    AclTransformer::OperationGraphNode &ffnResidualAddNode = opGraph_.nodes.at(9);

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
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedK, value,  cacheK,      cacheV, positionEmbedQ,
                                            attentionMask,  seqLen, tokenOffset, layerId};
    selfAttentionKvCacheNode.outTensorIds = {selfOut};

    selfOutLinearNode.operation = selfOutLinearOp;
    selfOutLinearNode.inTensorIds = {selfOut, selfOutLinearWeight, selfOutLinearBias};
    selfOutLinearNode.outTensorIds = {selfLinearOut};

    selfResidualAddNode.operation = selfResidualAddOp;
    selfResidualAddNode.inTensorIds = {inputNormOut, selfLinearOut};
    selfResidualAddNode.outTensorIds = {selfResidualAddOut};

    selfNormNode.operation = selfNormOp;
    selfNormNode.inTensorIds = {selfResidualAddOut, selfOutNormWeight, selfOutNormBias};
    selfNormNode.outTensorIds = {selfNormOut};

    ffnNode.operation = ffnOp;
    ffnNode.inTensorIds = {selfNormOut, ffnLinearWeight, ffnLinearBias};
    ffnNode.outTensorIds = {ffnOut};

    ffnLinearNode.operation = ffnLinearOp;
    ffnLinearNode.inTensorIds = {ffnOut, ffnOutLinearWeight, ffnOutLinearBias};
    ffnLinearNode.outTensorIds = {ffnLinearOut};

    ffnResidualAddNode.operation = ffnResidualAddOp;
    ffnResidualAddNode.inTensorIds = {selfNormOut, ffnLinearOut};
    ffnResidualAddNode.outTensorIds = {glmBlockOut};
}
} // namespace AclTransformer
