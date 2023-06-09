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
#include <json/json.h>
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
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/ffn_operation.h"

void ChatGlm6BLayer(const Json::Value &paramJson, AclTransformer::VariantPack &variantPack)
{
    // in
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
    const uint64_t selfResidualAddOut = 29;
    const uint64_t selfNormOut = 30;
    const uint64_t ffnOut = 31;
    const uint64_t ffnLinearOut = 32;

    AclTransformer::NormParam inputNormParam;
    inputNormParam.layerNormEps = paramJson["layerNormEps"].asDouble();
    AclTransformer::LinearParam mixdQkvLinearParam;
    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = paramJson["headNum"].asInt();
    AclTransformer::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.transKey = paramJson["transKey"].asBool();
    selfAttentionKvCacheParam.dk = paramJson["dk"].asInt();
    selfAttentionKvCacheParam.headNum = positionEmbeddingParam.headNum;
    selfAttentionKvCacheParam.layerId = paramJson["layerId"].asInt();
    AclTransformer::LinearParam selfOutLinearParam;
    AclTransformer::AddParam selfResidualAddParam;
    selfResidualAddParam.scale = paramJson["ResidualAddScale"].asFloat();
    AclTransformer::NormParam selfNormParam;
    selfNormParam.layerNormEps = inputNormParam.layerNormEps;
    AclTransformer::FfnParam ffnParam;
    AclTransformer::LinearParam ffnLinearParam;
    AclTransformer::AddParam ffnResidualAddParam;
    ffnResidualAddParam.scale = selfResidualAddParam.scale;

    AclTransformer::NormOperation inputNormOp(inputNormParam);
    AclTransformer::LinearOperation mixdQkvLinearOp(mixdQkvLinearParam);
    AclTransformer::PositionEmbeddingOperation positionEmbeddingOp(positionEmbeddingParam);
    AclTransformer::SelfAttentionKvCacheOperation selfAttentionKvCacheOp(selfAttentionKvCacheParam);
    AclTransformer::LinearOperation selfOutLinearOp(selfOutLinearParam);
    AclTransformer::AddOperation selfResidualAddOp(selfResidualAddParam);
    AclTransformer::NormOperation selfNormOp(selfNormParam);
    AclTransformer::FfnOperation ffnOp(ffnParam);
    AclTransformer::LinearOperation ffnLinearOp(ffnLinearParam);
    AclTransformer::AddOperation ffnResidualAddOp(ffnResidualAddParam);

    static int64_t graphId = 0;
    AclTransformer::OperationGraph opGraph;
    opGraph.name = "GlmBlockGraph_" + std::to_string(graphId++);
    opGraph.inTensorSize = variantPack.inTensors.size();
    opGraph.outTensorSize = variantPack.outTensors.size();
    opGraph.intermediateTensorSize = 11;
    opGraph.nodes.resize(10);

    AclTransformer::OperationGraphNode &inputNormNode = opGraph.nodes.at(0);
    AclTransformer::OperationGraphNode &mixdQkvLinearNode = opGraph.nodes.at(1);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = opGraph.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = opGraph.nodes.at(3);
    AclTransformer::OperationGraphNode &selfOutLinearNode = opGraph.nodes.at(4);
    AclTransformer::OperationGraphNode &selfResidualAddNode = opGraph.nodes.at(5);
    AclTransformer::OperationGraphNode &selfNormNode = opGraph.nodes.at(6);
    AclTransformer::OperationGraphNode &ffnNode = opGraph.nodes.at(7);
    AclTransformer::OperationGraphNode &ffnLinearNode = opGraph.nodes.at(8);
    AclTransformer::OperationGraphNode &ffnResidualAddNode = opGraph.nodes.at(9);

    inputNormNode.operation = &inputNormOp;
    inputNormNode.inTensorIds = {hiddenStates, normWeight, normBias};
    inputNormNode.outTensorIds = {inputNormOut};

    mixdQkvLinearNode.operation = &mixdQkvLinearOp;
    mixdQkvLinearNode.inTensorIds = {inputNormOut, qkvMixdWeight, qkvMixdBias};
    mixdQkvLinearNode.outTensorIds = {mixedLinearOutQkv};

    positionEmbeddingNode.operation = &positionEmbeddingOp;
    positionEmbeddingNode.inTensorIds = {mixedLinearOutQkv, positionIds, cosTable, sinTable};
    positionEmbeddingNode.outTensorIds = {positionEmbedQ, positionEmbedK, value};

    selfAttentionKvCacheNode.operation = &selfAttentionKvCacheOp;
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedQ, positionEmbedK, value, attentionMask, pastKey, pastValue};
    selfAttentionKvCacheNode.outTensorIds = {selfOut, presentKey, presentValue};

    selfOutLinearNode.operation = &selfOutLinearOp;
    selfOutLinearNode.inTensorIds = {selfOut, selfOutLinearWeight, selfOutLinearBias};
    selfOutLinearNode.outTensorIds = {selfLinearOut};

    selfResidualAddNode.operation = &selfResidualAddOp;
    selfResidualAddNode.inTensorIds = {inputNormOut, selfLinearOut};
    selfResidualAddNode.outTensorIds = {selfResidualAddOut};

    selfNormNode.operation = &selfNormOp;
    selfNormNode.inTensorIds = {selfResidualAddOut, selfOutNormWeight, selfOutNormBias};
    selfNormNode.outTensorIds = {selfNormOut};

    ffnNode.operation = &ffnOp;
    ffnNode.inTensorIds = {selfResidualAddOut, ffnLinearWeight, ffnLinearBias};
    ffnNode.outTensorIds = {ffnOut};

    ffnLinearNode.operation = &ffnLinearOp;
    ffnLinearNode.inTensorIds = {ffnOut, ffnOutLinearWeight, ffnOutLinearBias};
    ffnLinearNode.outTensorIds = {ffnLinearOut};

    ffnResidualAddNode.operation = &ffnResidualAddOp;
    ffnResidualAddNode.inTensorIds = {selfNormOut, ffnLinearOut};
    ffnResidualAddNode.outTensorIds = {glmBlockOut};

    ExampleUtil::ExecuteOperationGraph(opGraph, variantPack);
}
