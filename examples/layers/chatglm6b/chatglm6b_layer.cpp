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
#include "chatglm6b_layer.h"
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
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
ChatGlm6BLayer::ChatGlm6BLayer(const nlohmann::json &paramJson) : Layer("ChatGlm6BLayer", paramJson)
{
    BuildGraph();
    BuildPlan();
}

ChatGlm6BLayer::~ChatGlm6BLayer() {}

AsdOps::Status ChatGlm6BLayer::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                          AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    std::string phase = "decoder";
    if (paramJson_.contains("phase")) {
        phase = paramJson_["phase"].get<std::string>();
        ASD_LOG(INFO) << "phase" << phase;
    }
    if (phase == "decoder") {
        const AsdOps::Tensor &keyTensor = inTensors.at(17);
        const AsdOps::Tensor &ValueTensor = inTensors.at(18);

        outTensorDescs.resize(3);
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(1) = keyTensor.desc;
        outTensorDescs.at(1).dims.at(0) += 1;
        outTensorDescs.at(2) = ValueTensor.desc;
        outTensorDescs.at(2).dims.at(0) += 1;
        return AsdOps::Status::OkStatus();
    } else if (phase == "encoder") {
        int64_t headNum = paramJson_["headNum"].get<int>();
        int64_t headSize = paramJson_["dk"].get<int>();
        outTensorDescs.resize(3);
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(1) = inTensors.at(0).desc;
        outTensorDescs.at(1).dims.at(2) = headNum;
        outTensorDescs.at(1).dims.push_back(headSize);
        outTensorDescs.at(2) = outTensorDescs.at(1);
        return AsdOps::Status::OkStatus();
    }

    return AsdOps::Status::FailStatus(1, "in tensor size != 17 or 19");
}

void ChatGlm6BLayer::BuildGraph()
{
    std::string phase = "decoder";
    if (paramJson_.contains("phase")) {
        phase = paramJson_["phase"].get<std::string>();
    }
    if (phase == "encoder") {
        BuildEncoderGraph();
    } else if (phase == "decoder") {
        BuildDecoderGraph();
    }
}

void ChatGlm6BLayer::BuildEncoderGraph()
{
    ASD_LOG(INFO) << "start BuildEncoderGraph";
    uint64_t tensorId = 0;
    // in
    const uint64_t hiddenStates = tensorId++;
    const uint64_t normWeight = tensorId++;
    const uint64_t normBias = tensorId++;
    const uint64_t qkvMixdWeight = tensorId++;
    const uint64_t qkvMixdBias = tensorId++;
    const uint64_t selfOutLinearWeight = tensorId++;
    const uint64_t selfOutLinearBias = tensorId++;
    const uint64_t selfOutNormWeight = tensorId++;
    const uint64_t selfOutNormBias = tensorId++;
    const uint64_t ffnLinearWeight = tensorId++;
    const uint64_t ffnLinearBias = tensorId++;
    const uint64_t ffnOutLinearWeight = tensorId++;
    const uint64_t ffnOutLinearBias = tensorId++;
    const uint64_t positionIds = tensorId++;
    const uint64_t cosTable = tensorId++;
    const uint64_t sinTable = tensorId++;
    const uint64_t attentionMask = tensorId++;
    // out
    const uint64_t glmBlockOut = tensorId++;
    const uint64_t presentKey = tensorId++;
    const uint64_t presentValue = tensorId++;
    // intermiate
    const uint64_t inputNormOut = tensorId++;
    const uint64_t mixedLinearOutQkv = tensorId++;
    const uint64_t positionEmbedQ = tensorId++;
    const uint64_t selfOut = tensorId++;
    const uint64_t selfLinearOut = tensorId++;
    const uint64_t selfResidualAddOut = tensorId++;
    const uint64_t selfNormOut = tensorId++;
    const uint64_t ffnOut = tensorId++;
    const uint64_t ffnLinearOut = tensorId++;

    AclTransformer::NormParam inputNormParam;
    inputNormParam.layerNormEps = paramJson_["layerNormEps"].get<double>();
    AclTransformer::LinearParam mixdQkvLinearParam;
    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = paramJson_["headNum"].get<int>();
    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.transKey = paramJson_["transKey"].get<bool>();
    selfAttentionParam.dk = paramJson_["dk"].get<int>();
    selfAttentionParam.headNum = positionEmbeddingParam.headNum;
    selfAttentionParam.layerId = paramJson_["layerId"].get<int>();
    selfAttentionParam.model = "chatglm6b";
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
    AclTransformer::SelfAttentionOperation *selfAttentionOp =
        new AclTransformer::SelfAttentionOperation(selfAttentionParam);
    AclTransformer::LinearOperation *selfOutLinearOp = new AclTransformer::LinearOperation(selfOutLinearParam);
    AclTransformer::AddOperation *selfResidualAddOp = new AclTransformer::AddOperation(selfResidualAddParam);
    AclTransformer::NormOperation *selfNormOp = new AclTransformer::NormOperation(selfNormParam);
    AclTransformer::FfnOperation *ffnOp = new AclTransformer::FfnOperation(ffnParam);
    AclTransformer::LinearOperation *ffnLinearOp = new AclTransformer::LinearOperation(ffnLinearParam);
    AclTransformer::AddOperation *ffnResidualAddOp = new AclTransformer::AddOperation(ffnResidualAddParam);

    opGraph_.inTensorSize = 17;
    opGraph_.outTensorSize = 3;
    opGraph_.intermediateTensorSize = 9;
    opGraph_.nodes.resize(10);

    AclTransformer::OperationGraphNode &inputNormNode = opGraph_.nodes.at(0);
    AclTransformer::OperationGraphNode &mixdQkvLinearNode = opGraph_.nodes.at(1);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = opGraph_.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionNode = opGraph_.nodes.at(3);
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
    positionEmbeddingNode.outTensorIds = {positionEmbedQ, presentKey, presentValue};

    selfAttentionNode.operation = selfAttentionOp;
    selfAttentionNode.inTensorIds = {positionEmbedQ, presentKey, presentValue, attentionMask};
    selfAttentionNode.outTensorIds = {selfOut};

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

void ChatGlm6BLayer::BuildDecoderGraph()
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
    AclTransformer::SelfAttentionKvCacheOperation *selfAttentionKvCacheOp =
        new AclTransformer::SelfAttentionKvCacheOperation(selfAttentionKvCacheParam);
    AclTransformer::LinearOperation *selfOutLinearOp = new AclTransformer::LinearOperation(selfOutLinearParam);
    AclTransformer::AddOperation *selfResidualAddOp = new AclTransformer::AddOperation(selfResidualAddParam);
    AclTransformer::NormOperation *selfNormOp = new AclTransformer::NormOperation(selfNormParam);
    AclTransformer::FfnOperation *ffnOp = new AclTransformer::FfnOperation(ffnParam);
    AclTransformer::LinearOperation *ffnLinearOp = new AclTransformer::LinearOperation(ffnLinearParam);
    AclTransformer::AddOperation *ffnResidualAddOp = new AclTransformer::AddOperation(ffnResidualAddParam);

    opGraph_.inTensorSize = 19;
    opGraph_.outTensorSize = 3;
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
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedQ, positionEmbedK, value, attentionMask, pastKey, pastValue};
    selfAttentionKvCacheNode.outTensorIds = {selfOut, presentKey, presentValue};

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
