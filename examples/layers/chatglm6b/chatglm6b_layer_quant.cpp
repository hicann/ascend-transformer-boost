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
#include <iostream>
#include <string>
#include <regex>

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

#include "acltransformer/ops/quant_operation.h"
#include "acltransformer/ops/norm_quant_operation.h"
#include "acltransformer/ops/add_norm_quant_operation.h"
#include "acltransformer/ops/linear_quant_operation.h"
#include "acltransformer/ops/ffn_quant_operation.h"

#include "chatglm6b_layer_quant.h"


namespace AclTransformer {
ChatGlm6BLayerQuant::ChatGlm6BLayerQuant(const nlohmann::json &paramJson) : Layer("ChatGlm6BLayerQuant", paramJson)
{
    BuildGraph();
    BuildPlan();
}

ChatGlm6BLayerQuant::~ChatGlm6BLayerQuant() {}

AsdOps::Status ChatGlm6BLayerQuant::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                               AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    int layerId = paramJson_["layerId"].get<int>();

    const AsdOps::Tensor &keyTensor = inTensors.at(21);
    const AsdOps::Tensor &ValueTensor = inTensors.at(22);

    if (layerId == 27) {
        outTensorDescs.resize(3);
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(1) = keyTensor.desc;
        outTensorDescs.at(1).dims.at(0) += 1;
        outTensorDescs.at(2) = ValueTensor.desc;
        outTensorDescs.at(2).dims.at(0) += 1;
    } else {
        outTensorDescs.resize(4);
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(1) = keyTensor.desc;
        outTensorDescs.at(1).dims.at(0) += 1;
        outTensorDescs.at(2) = ValueTensor.desc;
        outTensorDescs.at(2).dims.at(0) += 1;
        outTensorDescs.at(3) = inTensors.at(0).desc;
    }

    return AsdOps::Status::OkStatus();
}

void ChatGlm6BLayerQuant::BuildGraph()
{
    int layerId = paramJson_["layerId"].get<int>();
    if (layerId == 0) {
        // Use middle graph
        BuildMidGraph();
    } else if (layerId == 27) {
        BuildLastGraph();
    } else {
        BuildMidGraph();
    }
}

void ChatGlm6BLayerQuant::BuildMidGraph()
{
    // in
    uint64_t tensorId = 0;
    const uint64_t hiddenStates = tensorId++;
    const uint64_t qkvMixdQuantWeight = tensorId++;
    const uint64_t qkvMixdQuantDeqScale = tensorId++;
    const uint64_t qkvMixdQuantBias = tensorId++;

    const uint64_t denseQuantWeight = tensorId++;
    const uint64_t denseQuantDeqScale = tensorId++;
    const uint64_t denseQuantBias = tensorId++;

    const uint64_t hto4hQuantWeight = tensorId++;
    const uint64_t hto4hQuantDeqScale = tensorId++;
    const uint64_t hto4hQuantBias = tensorId++;

    const uint64_t fhtohQuantWeight = tensorId++;
    const uint64_t fhtohQuantDeqScale = tensorId++;
    const uint64_t fhtohQuantBias = tensorId++;

    const uint64_t inputLayerNormWeight = tensorId++;
    const uint64_t inputLayerNormBias = tensorId++;

    const uint64_t postLayerNormWeight = tensorId++;
    const uint64_t postLayerNormBias = tensorId++;

    const uint64_t positionIds = tensorId++;
    const uint64_t cosTable = tensorId++;
    const uint64_t sinTable = tensorId++;
    const uint64_t attentionMask = tensorId++;
    const uint64_t pastKey = tensorId++;
    const uint64_t pastValue = tensorId++;

    const uint64_t resIn = tensorId++;

    // out
    const uint64_t glmBlockOut = tensorId++;
    const uint64_t presentKey = tensorId++;
    const uint64_t presentValue = tensorId++;
    const uint64_t resOut = tensorId++;

    // intermiate
    const uint64_t inputNormOut = tensorId++;
    const uint64_t inputNormResOut = tensorId++;

    const uint64_t mixedLinearQuantOutQkv = tensorId++;

    const uint64_t positionEmbedQ = tensorId++;
    const uint64_t positionEmbedK = tensorId++;
    const uint64_t value = tensorId++;
    const uint64_t selfOut = tensorId++;

    const uint64_t denseQuantOut = tensorId++;
    const uint64_t denseLinearQuantOut = tensorId++;

    const uint64_t selfLayernormQuantOut = tensorId++;
    const uint64_t ffnLinearQuantOut = tensorId++;
    const uint64_t ffnOutQuantOut = tensorId++;

    AclTransformer::AddNormQuantParam inputLayernormQuantParam;
    inputLayernormQuantParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    inputLayernormQuantParam.inputScale = paramJson_["QkvInputScale"].get<float>();
    inputLayernormQuantParam.inputOffset = paramJson_["QkvInputOffset"].get<int>();
    inputLayernormQuantParam.inputAlpha = paramJson_["ResidualAddScale"].get<float>();

    AclTransformer::LinearQuantParam minxedQKVLinearQuantParam;
    minxedQKVLinearQuantParam.transposeA = false;
    minxedQKVLinearQuantParam.transposeB = false;

    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = paramJson_["headNum"].get<int>();

    AclTransformer::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.transKey = paramJson_["transKey"].get<bool>();
    selfAttentionKvCacheParam.dk = paramJson_["dk"].get<int>();
    selfAttentionKvCacheParam.headNum = positionEmbeddingParam.headNum;
    selfAttentionKvCacheParam.layerId = paramJson_["layerId"].get<int>();

    AclTransformer::QuantParam denseQuantParam;
    denseQuantParam.inputScale = paramJson_["DenseInputScale"].get<float>();
    denseQuantParam.inputOffset = paramJson_["DenseInputOffset"].get<int>();

    AclTransformer::LinearQuantParam denseLinearQuantParam;
    denseLinearQuantParam.transposeA = false;
    denseLinearQuantParam.transposeB = false;

    AclTransformer::AddNormQuantParam selfLayernormQuantParam;
    selfLayernormQuantParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    selfLayernormQuantParam.inputScale = paramJson_["SelfLnInputScale"].get<float>();
    selfLayernormQuantParam.inputOffset = paramJson_["SelfLnInputOffset"].get<int>();
    selfLayernormQuantParam.inputAlpha = paramJson_["ResidualAddScale"].get<float>();

    AclTransformer::FfnQuantParam ffnLinearQuantParam;
    ffnLinearQuantParam.transposeA = false;
    ffnLinearQuantParam.transposeB = false;

    AclTransformer::QuantParam ffnOutQuantParam;
    ffnOutQuantParam.inputScale = paramJson_["FfnOutInputScale"].get<float>();
    ffnOutQuantParam.inputOffset = paramJson_["FfnOutInputOffset"].get<int>();

    AclTransformer::LinearQuantParam ffnOutLinearQuantParam;
    ffnOutLinearQuantParam.transposeA = false;
    ffnOutLinearQuantParam.transposeB = false;

    AclTransformer::AddNormQuantOperation *inputLayernormQuantOp =
        new AclTransformer::AddNormQuantOperation(inputLayernormQuantParam);
    AclTransformer::LinearQuantOperation *mixdQkvLinearQuantOp =
        new AclTransformer::LinearQuantOperation(minxedQKVLinearQuantParam);

    AclTransformer::PositionEmbeddingOperation *positionEmbeddingOp =
        new AclTransformer::PositionEmbeddingOperation(positionEmbeddingParam);
    AclTransformer::SelfAttentionKvCacheOperation *selfAttentionKvCacheOp =
        new AclTransformer::SelfAttentionKvCacheOperation(selfAttentionKvCacheParam);

    AclTransformer::QuantOperation *denseQuantOp = new AclTransformer::QuantOperation(denseQuantParam);
    AclTransformer::LinearQuantOperation *denseLinearQuantOp =
        new AclTransformer::LinearQuantOperation(denseLinearQuantParam);

    AclTransformer::AddNormQuantOperation *selfLayernormQuantOp =
        new AclTransformer::AddNormQuantOperation(selfLayernormQuantParam);
    AclTransformer::FfnQuantOperation *ffnLinearQuantOp = new AclTransformer::FfnQuantOperation(ffnLinearQuantParam);

    AclTransformer::QuantOperation *ffnOutQuantOp = new AclTransformer::QuantOperation(ffnOutQuantParam);
    AclTransformer::LinearQuantOperation *ffnOutLinearQuantOp =
        new AclTransformer::LinearQuantOperation(ffnOutLinearQuantParam);

    opGraph_.inTensorSize = 24;
    opGraph_.outTensorSize = 4;
    opGraph_.intermediateTensorSize = 12;
    opGraph_.nodes.resize(10);

    AclTransformer::OperationGraphNode &inputQuantNode = opGraph_.nodes.at(0);
    AclTransformer::OperationGraphNode &mixdQkvLinearQuantNode = opGraph_.nodes.at(1);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = opGraph_.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = opGraph_.nodes.at(3);
    AclTransformer::OperationGraphNode &denseQuantNode = opGraph_.nodes.at(4);
    AclTransformer::OperationGraphNode &denseLinearQuantNode = opGraph_.nodes.at(5);
    AclTransformer::OperationGraphNode &selfLayernormQuantNode = opGraph_.nodes.at(6);
    AclTransformer::OperationGraphNode &ffnLinearQuantNode = opGraph_.nodes.at(7);
    AclTransformer::OperationGraphNode &ffnOutQuantNode = opGraph_.nodes.at(8);
    AclTransformer::OperationGraphNode &ffnOutLinearQuantNode = opGraph_.nodes.at(9);

    inputQuantNode.operation = inputLayernormQuantOp;
    inputQuantNode.inTensorIds = {hiddenStates, inputLayerNormWeight, inputLayerNormBias, resIn};
    inputQuantNode.outTensorIds = {inputNormOut, inputNormResOut};

    mixdQkvLinearQuantNode.operation = mixdQkvLinearQuantOp;
    mixdQkvLinearQuantNode.inTensorIds = {inputNormOut, qkvMixdQuantWeight, qkvMixdQuantBias, qkvMixdQuantDeqScale};
    mixdQkvLinearQuantNode.outTensorIds = {mixedLinearQuantOutQkv};

    positionEmbeddingNode.operation = positionEmbeddingOp;
    positionEmbeddingNode.inTensorIds = {mixedLinearQuantOutQkv, positionIds, cosTable, sinTable};
    positionEmbeddingNode.outTensorIds = {positionEmbedQ, positionEmbedK, value};

    selfAttentionKvCacheNode.operation = selfAttentionKvCacheOp;
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedQ, positionEmbedK, value, attentionMask, pastKey, pastValue};
    selfAttentionKvCacheNode.outTensorIds = {selfOut, presentKey, presentValue};

    denseQuantNode.operation = denseQuantOp;
    denseQuantNode.inTensorIds = {selfOut};
    denseQuantNode.outTensorIds = {denseQuantOut};

    denseLinearQuantNode.operation = denseLinearQuantOp;
    denseLinearQuantNode.inTensorIds = {denseQuantOut, denseQuantWeight, denseQuantBias, denseQuantDeqScale};
    denseLinearQuantNode.outTensorIds = {denseLinearQuantOut};

    selfLayernormQuantNode.operation = selfLayernormQuantOp;
    selfLayernormQuantNode.inTensorIds = {denseLinearQuantOut, postLayerNormWeight, postLayerNormBias, inputNormResOut};
    selfLayernormQuantNode.outTensorIds = {selfLayernormQuantOut, resOut};

    ffnLinearQuantNode.operation = ffnLinearQuantOp;
    ffnLinearQuantNode.inTensorIds = {selfLayernormQuantOut, hto4hQuantWeight, hto4hQuantBias, hto4hQuantDeqScale};
    ffnLinearQuantNode.outTensorIds = {ffnLinearQuantOut};

    ffnOutQuantNode.operation = ffnOutQuantOp;
    ffnOutQuantNode.inTensorIds = {ffnLinearQuantOut};
    ffnOutQuantNode.outTensorIds = {ffnOutQuantOut};

    ffnOutLinearQuantNode.operation = ffnOutLinearQuantOp;
    ffnOutLinearQuantNode.inTensorIds = {ffnOutQuantOut, fhtohQuantWeight, fhtohQuantBias, fhtohQuantDeqScale};
    ffnOutLinearQuantNode.outTensorIds = {glmBlockOut};
}

void ChatGlm6BLayerQuant::BuildFirstGraph()
{
    // in
    uint64_t tensorId = 0;
    const uint64_t hiddenStates = tensorId++;
    const uint64_t qkvMixdQuantWeight = tensorId++;
    const uint64_t qkvMixdQuantDeqScale = tensorId++;
    const uint64_t qkvMixdQuantBias = tensorId++;

    const uint64_t denseQuantWeight = tensorId++;
    const uint64_t denseQuantDeqScale = tensorId++;
    const uint64_t denseQuantBias = tensorId++;

    const uint64_t hto4hQuantWeight = tensorId++;
    const uint64_t hto4hQuantDeqScale = tensorId++;
    const uint64_t hto4hQuantBias = tensorId++;

    const uint64_t fhtohQuantWeight = tensorId++;
    const uint64_t fhtohQuantDeqScale = tensorId++;
    const uint64_t fhtohQuantBias = tensorId++;

    const uint64_t inputLayerNormWeight = tensorId++;
    const uint64_t inputLayerNormBias = tensorId++;

    const uint64_t postLayerNormWeight = tensorId++;
    const uint64_t postLayerNormBias = tensorId++;

    const uint64_t positionIds = tensorId++;
    const uint64_t cosTable = tensorId++;
    const uint64_t sinTable = tensorId++;
    const uint64_t attentionMask = tensorId++;
    const uint64_t pastKey = tensorId++;
    const uint64_t pastValue = tensorId++;

    // out
    const uint64_t glmBlockOut = tensorId++;
    const uint64_t presentKey = tensorId++;
    const uint64_t presentValue = tensorId++;
    const uint64_t resOut = tensorId++;

    // intermiate
    const uint64_t inputNormOut = tensorId++;
    const uint64_t inputNormResOut = tensorId++;

    const uint64_t mixedLinearQuantOutQkv = tensorId++;

    const uint64_t positionEmbedQ = tensorId++;
    const uint64_t positionEmbedK = tensorId++;
    const uint64_t value = tensorId++;
    const uint64_t selfOut = tensorId++;

    const uint64_t denseQuantOut = tensorId++;
    const uint64_t denseLinearQuantOut = tensorId++;

    const uint64_t selfLayernormQuantOut = tensorId++;
    const uint64_t ffnLinearQuantOut = tensorId++;
    const uint64_t ffnOutQuantOut = tensorId++;

    AclTransformer::NormQuantParam inputLayernormQuantParam;
    inputLayernormQuantParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    inputLayernormQuantParam.inputScale = paramJson_["QkvInputScale"].get<float>();
    inputLayernormQuantParam.inputOffset = paramJson_["QkvInputOffset"].get<int>();
    inputLayernormQuantParam.inputAlpha = paramJson_["ResidualAddScale"].get<float>();

    AclTransformer::LinearQuantParam minxedQKVLinearQuantParam;
    minxedQKVLinearQuantParam.transposeA = false;
    minxedQKVLinearQuantParam.transposeB = false;

    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = paramJson_["headNum"].get<int>();

    AclTransformer::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.transKey = paramJson_["transKey"].get<bool>();
    selfAttentionKvCacheParam.dk = paramJson_["dk"].get<int>();
    selfAttentionKvCacheParam.headNum = positionEmbeddingParam.headNum;
    selfAttentionKvCacheParam.layerId = paramJson_["layerId"].get<int>();

    AclTransformer::QuantParam denseQuantParam;
    denseQuantParam.inputScale = paramJson_["DenseInputScale"].get<float>();
    denseQuantParam.inputOffset = paramJson_["DenseInputOffset"].get<int>();

    AclTransformer::LinearQuantParam denseLinearQuantParam;
    denseLinearQuantParam.transposeA = false;
    denseLinearQuantParam.transposeB = false;

    AclTransformer::AddNormQuantParam selfLayernormQuantParam;
    selfLayernormQuantParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    selfLayernormQuantParam.inputScale = paramJson_["SelfLnInputScale"].get<float>();
    selfLayernormQuantParam.inputOffset = paramJson_["selfLnInputOffset"].get<int>();
    selfLayernormQuantParam.inputAlpha = paramJson_["ResidualAddScale"].get<float>();

    AclTransformer::FfnQuantParam ffnLinearQuantParam;
    ffnLinearQuantParam.transposeA = false;
    ffnLinearQuantParam.transposeB = false;

    AclTransformer::QuantParam ffnOutQuantParam;
    ffnOutQuantParam.inputScale = paramJson_["FfnOutInputScale"].get<float>();
    ffnOutQuantParam.inputOffset = paramJson_["FfnOutInputOffset"].get<int>();

    AclTransformer::LinearQuantParam ffnOutLinearQuantParam;
    ffnOutLinearQuantParam.transposeA = false;
    ffnOutLinearQuantParam.transposeB = false;

    AclTransformer::NormQuantOperation *inputLayernormQuantOp =
        new AclTransformer::NormQuantOperation(inputLayernormQuantParam);
    AclTransformer::LinearQuantOperation *mixdQkvLinearQuantOp =
        new AclTransformer::LinearQuantOperation(minxedQKVLinearQuantParam);

    AclTransformer::PositionEmbeddingOperation *positionEmbeddingOp =
        new AclTransformer::PositionEmbeddingOperation(positionEmbeddingParam);
    AclTransformer::SelfAttentionKvCacheOperation *selfAttentionKvCacheOp =
        new AclTransformer::SelfAttentionKvCacheOperation(selfAttentionKvCacheParam);

    AclTransformer::QuantOperation *denseQuantOp = new AclTransformer::QuantOperation(denseQuantParam);
    AclTransformer::LinearQuantOperation *denseLinearQuantOp =
        new AclTransformer::LinearQuantOperation(denseLinearQuantParam);

    AclTransformer::AddNormQuantOperation *selfLayernormQuantOp =
        new AclTransformer::AddNormQuantOperation(selfLayernormQuantParam);
    AclTransformer::FfnQuantOperation *ffnLinearQuantOp = new AclTransformer::FfnQuantOperation(ffnLinearQuantParam);

    AclTransformer::QuantOperation *ffnOutQuantOp = new AclTransformer::QuantOperation(ffnOutQuantParam);
    AclTransformer::LinearQuantOperation *ffnOutLinearQuantOp =
        new AclTransformer::LinearQuantOperation(ffnOutLinearQuantParam);

    opGraph_.inTensorSize = 23;
    opGraph_.outTensorSize = 4;
    opGraph_.intermediateTensorSize = 12;
    opGraph_.nodes.resize(10);

    AclTransformer::OperationGraphNode &inputQuantNode = opGraph_.nodes.at(0);
    AclTransformer::OperationGraphNode &mixdQkvLinearQuantNode = opGraph_.nodes.at(1);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = opGraph_.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = opGraph_.nodes.at(3);
    AclTransformer::OperationGraphNode &denseQuantNode = opGraph_.nodes.at(4);
    AclTransformer::OperationGraphNode &denseLinearQuantNode = opGraph_.nodes.at(5);
    AclTransformer::OperationGraphNode &selfLayernormQuantNode = opGraph_.nodes.at(6);
    AclTransformer::OperationGraphNode &ffnLinearQuantNode = opGraph_.nodes.at(7);
    AclTransformer::OperationGraphNode &ffnOutQuantNode = opGraph_.nodes.at(8);
    AclTransformer::OperationGraphNode &ffnOutLinearQuantNode = opGraph_.nodes.at(9);

    inputQuantNode.operation = inputLayernormQuantOp;
    inputQuantNode.inTensorIds = {hiddenStates, inputLayerNormWeight, inputLayerNormBias};
    inputQuantNode.outTensorIds = {inputNormOut, inputNormResOut};

    mixdQkvLinearQuantNode.operation = mixdQkvLinearQuantOp;
    mixdQkvLinearQuantNode.inTensorIds = {inputNormOut, qkvMixdQuantWeight, qkvMixdQuantBias, qkvMixdQuantDeqScale};
    mixdQkvLinearQuantNode.outTensorIds = {mixedLinearQuantOutQkv};

    positionEmbeddingNode.operation = positionEmbeddingOp;
    positionEmbeddingNode.inTensorIds = {mixedLinearQuantOutQkv, positionIds, cosTable, sinTable};
    positionEmbeddingNode.outTensorIds = {positionEmbedQ, positionEmbedK, value};

    selfAttentionKvCacheNode.operation = selfAttentionKvCacheOp;
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedQ, positionEmbedK, value, attentionMask, pastKey, pastValue};
    selfAttentionKvCacheNode.outTensorIds = {selfOut, presentKey, presentValue};

    denseQuantNode.operation = denseQuantOp;
    denseQuantNode.inTensorIds = {selfOut};
    denseQuantNode.outTensorIds = {denseQuantOut};

    denseLinearQuantNode.operation = denseLinearQuantOp;
    denseLinearQuantNode.inTensorIds = {denseQuantOut, denseQuantWeight, denseQuantBias, denseQuantDeqScale};
    denseLinearQuantNode.outTensorIds = {denseLinearQuantOut};

    selfLayernormQuantNode.operation = selfLayernormQuantOp;
    selfLayernormQuantNode.inTensorIds = {denseLinearQuantOut, postLayerNormWeight, postLayerNormBias, inputNormResOut};
    selfLayernormQuantNode.outTensorIds = {selfLayernormQuantOut, resOut};

    ffnLinearQuantNode.operation = ffnLinearQuantOp;
    ffnLinearQuantNode.inTensorIds = {selfLayernormQuantOut, hto4hQuantWeight, hto4hQuantBias, hto4hQuantDeqScale};
    ffnLinearQuantNode.outTensorIds = {ffnLinearQuantOut};

    ffnOutQuantNode.operation = ffnOutQuantOp;
    ffnOutQuantNode.inTensorIds = {ffnLinearQuantOut};
    ffnOutQuantNode.outTensorIds = {ffnOutQuantOut};

    ffnOutLinearQuantNode.operation = ffnOutLinearQuantOp;
    ffnOutLinearQuantNode.inTensorIds = {ffnOutQuantOut, fhtohQuantWeight, fhtohQuantBias, fhtohQuantDeqScale};
    ffnOutLinearQuantNode.outTensorIds = {glmBlockOut};
}
void ChatGlm6BLayerQuant::BuildLastGraph()
{
    // in
    uint64_t tensorId = 0;
    const uint64_t hiddenStates = tensorId++;
    const uint64_t qkvMixdQuantWeight = tensorId++;
    const uint64_t qkvMixdQuantDeqScale = tensorId++;
    const uint64_t qkvMixdQuantBias = tensorId++;

    const uint64_t denseQuantWeight = tensorId++;
    const uint64_t denseQuantDeqScale = tensorId++;
    const uint64_t denseQuantBias = tensorId++;

    const uint64_t hto4hQuantWeight = tensorId++;
    const uint64_t hto4hQuantDeqScale = tensorId++;
    const uint64_t hto4hQuantBias = tensorId++;

    const uint64_t fhtohQuantWeight = tensorId++;
    const uint64_t fhtohQuantDeqScale = tensorId++;
    const uint64_t fhtohQuantBias = tensorId++;

    const uint64_t inputLayerNormWeight = tensorId++;
    const uint64_t inputLayerNormBias = tensorId++;

    const uint64_t postLayerNormWeight = tensorId++;
    const uint64_t postLayerNormBias = tensorId++;

    const uint64_t positionIds = tensorId++;
    const uint64_t cosTable = tensorId++;
    const uint64_t sinTable = tensorId++;
    const uint64_t attentionMask = tensorId++;
    const uint64_t pastKey = tensorId++;
    const uint64_t pastValue = tensorId++;

    const uint64_t resIn = tensorId++;

    // out
    const uint64_t glmBlockOut = tensorId++;
    const uint64_t presentKey = tensorId++;
    const uint64_t presentValue = tensorId++;

    // intermiate
    const uint64_t inputNormOut = tensorId++;
    const uint64_t inputNormResOut = tensorId++;

    const uint64_t mixedLinearQuantOutQkv = tensorId++;

    const uint64_t positionEmbedQ = tensorId++;
    const uint64_t positionEmbedK = tensorId++;
    const uint64_t value = tensorId++;
    const uint64_t selfOut = tensorId++;

    const uint64_t denseQuantOut = tensorId++;
    const uint64_t denseLinearQuantOut = tensorId++;

    const uint64_t selfLayernormQuantOut = tensorId++;
    const uint64_t ffnLinearQuantOut = tensorId++;
    const uint64_t ffnOutQuantOut = tensorId++;
    const uint64_t ffnOutLinearQuantOut = tensorId++;

    const uint64_t resOut = tensorId++;

    AclTransformer::AddNormQuantParam inputLayernormQuantParam;
    inputLayernormQuantParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    inputLayernormQuantParam.inputScale = paramJson_["QkvInputScale"].get<float>();
    inputLayernormQuantParam.inputOffset = paramJson_["QkvInputOffset"].get<int>();
    inputLayernormQuantParam.inputAlpha = paramJson_["ResidualAddScale"].get<float>();

    AclTransformer::LinearQuantParam minxedQKVLinearQuantParam;
    minxedQKVLinearQuantParam.transposeA = false;
    minxedQKVLinearQuantParam.transposeB = false;

    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = paramJson_["headNum"].get<int>();

    AclTransformer::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.transKey = paramJson_["transKey"].get<bool>();
    selfAttentionKvCacheParam.dk = paramJson_["dk"].get<int>();
    selfAttentionKvCacheParam.headNum = positionEmbeddingParam.headNum;
    selfAttentionKvCacheParam.layerId = paramJson_["layerId"].get<int>();

    AclTransformer::QuantParam denseQuantParam;
    denseQuantParam.inputScale = paramJson_["DenseInputScale"].get<float>();
    denseQuantParam.inputOffset = paramJson_["DenseInputOffset"].get<int>();

    AclTransformer::LinearQuantParam denseLinearQuantParam;
    denseLinearQuantParam.transposeA = false;
    denseLinearQuantParam.transposeB = false;

    AclTransformer::AddNormQuantParam selfLayernormQuantParam;
    selfLayernormQuantParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    selfLayernormQuantParam.inputScale = paramJson_["SelfLnInputScale"].get<float>();
    selfLayernormQuantParam.inputOffset = paramJson_["SelfLnInputOffset"].get<int>();
    selfLayernormQuantParam.inputAlpha = paramJson_["ResidualAddScale"].get<float>();

    AclTransformer::FfnQuantParam ffnLinearQuantParam;
    ffnLinearQuantParam.transposeA = false;
    ffnLinearQuantParam.transposeB = false;

    AclTransformer::QuantParam ffnOutQuantParam;
    ffnOutQuantParam.inputScale = paramJson_["FfnOutInputScale"].get<float>();
    ffnOutQuantParam.inputOffset = paramJson_["FfnOutInputOffset"].get<int>();

    AclTransformer::LinearQuantParam ffnOutLinearQuantParam;
    ffnOutLinearQuantParam.transposeA = false;
    ffnOutLinearQuantParam.transposeB = false;

    AclTransformer::AddParam ffnResidualAddParam;
    ffnResidualAddParam.scale = paramJson_["ResidualAddScale"].get<float>();

    AclTransformer::AddNormQuantOperation *inputLayernormQuantOp =
        new AclTransformer::AddNormQuantOperation(inputLayernormQuantParam);
    AclTransformer::LinearQuantOperation *mixdQkvLinearQuantOp =
        new AclTransformer::LinearQuantOperation(minxedQKVLinearQuantParam);

    AclTransformer::PositionEmbeddingOperation *positionEmbeddingOp =
        new AclTransformer::PositionEmbeddingOperation(positionEmbeddingParam);
    AclTransformer::SelfAttentionKvCacheOperation *selfAttentionKvCacheOp =
        new AclTransformer::SelfAttentionKvCacheOperation(selfAttentionKvCacheParam);

    AclTransformer::QuantOperation *denseQuantOp = new AclTransformer::QuantOperation(denseQuantParam);
    AclTransformer::LinearQuantOperation *denseLinearQuantOp =
        new AclTransformer::LinearQuantOperation(denseLinearQuantParam);

    AclTransformer::AddNormQuantOperation *selfLayernormQuantOp =
        new AclTransformer::AddNormQuantOperation(selfLayernormQuantParam);
    AclTransformer::FfnQuantOperation *ffnLinearQuantOp = new AclTransformer::FfnQuantOperation(ffnLinearQuantParam);

    AclTransformer::QuantOperation *ffnOutQuantOp = new AclTransformer::QuantOperation(ffnOutQuantParam);
    AclTransformer::LinearQuantOperation *ffnOutLinearQuantOp =
        new AclTransformer::LinearQuantOperation(ffnOutLinearQuantParam);
    AclTransformer::AddOperation *ffnResidualAddOp = new AclTransformer::AddOperation(ffnResidualAddParam);

    opGraph_.inTensorSize = 24;
    opGraph_.outTensorSize = 3;
    opGraph_.intermediateTensorSize = 14;
    opGraph_.nodes.resize(11);

    AclTransformer::OperationGraphNode &inputQuantNode = opGraph_.nodes.at(0);
    AclTransformer::OperationGraphNode &mixdQkvLinearQuantNode = opGraph_.nodes.at(1);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = opGraph_.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = opGraph_.nodes.at(3);
    AclTransformer::OperationGraphNode &denseQuantNode = opGraph_.nodes.at(4);
    AclTransformer::OperationGraphNode &denseLinearQuantNode = opGraph_.nodes.at(5);
    AclTransformer::OperationGraphNode &selfLayernormQuantNode = opGraph_.nodes.at(6);
    AclTransformer::OperationGraphNode &ffnLinearQuantNode = opGraph_.nodes.at(7);
    AclTransformer::OperationGraphNode &ffnOutQuantNode = opGraph_.nodes.at(8);
    AclTransformer::OperationGraphNode &ffnOutLinearQuantNode = opGraph_.nodes.at(9);
    AclTransformer::OperationGraphNode &ffnResidualAddNode = opGraph_.nodes.at(10);

    inputQuantNode.operation = inputLayernormQuantOp;
    inputQuantNode.inTensorIds = {hiddenStates, inputLayerNormWeight, inputLayerNormBias, resIn};
    inputQuantNode.outTensorIds = {inputNormOut, inputNormResOut};

    mixdQkvLinearQuantNode.operation = mixdQkvLinearQuantOp;
    mixdQkvLinearQuantNode.inTensorIds = {inputNormOut, qkvMixdQuantWeight, qkvMixdQuantBias, qkvMixdQuantDeqScale};
    mixdQkvLinearQuantNode.outTensorIds = {mixedLinearQuantOutQkv};

    positionEmbeddingNode.operation = positionEmbeddingOp;
    positionEmbeddingNode.inTensorIds = {mixedLinearQuantOutQkv, positionIds, cosTable, sinTable};
    positionEmbeddingNode.outTensorIds = {positionEmbedQ, positionEmbedK, value};

    selfAttentionKvCacheNode.operation = selfAttentionKvCacheOp;
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedQ, positionEmbedK, value, attentionMask, pastKey, pastValue};
    selfAttentionKvCacheNode.outTensorIds = {selfOut, presentKey, presentValue};

    denseQuantNode.operation = denseQuantOp;
    denseQuantNode.inTensorIds = {selfOut};
    denseQuantNode.outTensorIds = {denseQuantOut};

    denseLinearQuantNode.operation = denseLinearQuantOp;
    denseLinearQuantNode.inTensorIds = {denseQuantOut, denseQuantWeight, denseQuantBias, denseQuantDeqScale};
    denseLinearQuantNode.outTensorIds = {denseLinearQuantOut};

    selfLayernormQuantNode.operation = selfLayernormQuantOp;
    selfLayernormQuantNode.inTensorIds = {denseLinearQuantOut, postLayerNormWeight, postLayerNormBias, inputNormResOut};
    selfLayernormQuantNode.outTensorIds = {selfLayernormQuantOut, resOut};

    ffnLinearQuantNode.operation = ffnLinearQuantOp;
    ffnLinearQuantNode.inTensorIds = {selfLayernormQuantOut, hto4hQuantWeight, hto4hQuantBias, hto4hQuantDeqScale};
    ffnLinearQuantNode.outTensorIds = {ffnLinearQuantOut};

    ffnOutQuantNode.operation = ffnOutQuantOp;
    ffnOutQuantNode.inTensorIds = {ffnLinearQuantOut};
    ffnOutQuantNode.outTensorIds = {ffnOutQuantOut};

    ffnOutLinearQuantNode.operation = ffnOutLinearQuantOp;
    ffnOutLinearQuantNode.inTensorIds = {ffnOutQuantOut, fhtohQuantWeight, fhtohQuantBias, fhtohQuantDeqScale};
    ffnOutLinearQuantNode.outTensorIds = {glmBlockOut};

    ffnOutLinearQuantNode.operation = ffnOutLinearQuantOp;
    ffnOutLinearQuantNode.inTensorIds = {ffnOutQuantOut, fhtohQuantWeight, fhtohQuantBias, fhtohQuantDeqScale};
    ffnOutLinearQuantNode.outTensorIds = {ffnOutLinearQuantOut};

    ffnResidualAddNode.operation = ffnResidualAddOp;
    ffnResidualAddNode.inTensorIds = {resOut, ffnOutLinearQuantOut};
    ffnResidualAddNode.outTensorIds = {glmBlockOut};
}
} // namespace AclTransformer
