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
#include "chatglm6b_fusion_operation.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
enum Chatglm6BFusionTensorId {
    IN_HIDDENSTATES_ID = 0,
    normWeight = 1,
    normBias = 2,
    qkvMixdWeight = 3,
    qkvMixdBias = 4,
    selfOutLinearWeight = 5,
    selfOutLinearBias = 6,
    selfOutNormWeight = 7,
    selfOutNormBias = 8,
    ffnLinearWeight = 9,
    ffnLinearBias = 10,
    ffnOutLinearWeight = 11,
    ffnOutLinearBias = 12,
    positionIds = 13,
    cosTable = 14,
    sinTable = 15,
    attentionMask = 16,
    cacheK = 17,
    cacheV = 18,
    seqLen = 19,
    tokenOffset = 20,
    layerId = 21,
    OUT_GLMBLOCKOUT_ID = 22,
    inputNormOut = 23,
    mixedLinearOutQkv = 24,
    positionEmbedQ = 25,
    positionEmbedK = 26,
    value = 27,
    selfOut = 28,
    selfLinearOut = 29,
    selfResidualAddOut = 30,
    selfNormOut = 31,
    ffnOut = 32,
    ffnLinearOut = 33,
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 10;

ChatGlm6BFusionOperation::ChatGlm6BFusionOperation(const ChatGlm6BFusionParam &param)
    : GraphOperation("ChatGlm6BFusionOperation"), param_(param)
{
    BuildGraph();
}

ChatGlm6BFusionOperation::~ChatGlm6BFusionOperation() {}

uint64_t ChatGlm6BFusionOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t ChatGlm6BFusionOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status ChatGlm6BFusionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                        AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

void ChatGlm6BFusionOperation::BuildGraph()
{
    operationGraph_.inTensorSize = IN_TENSOR_COUNT;
    operationGraph_.outTensorSize = OUT_TENSOR_COUNT;
    operationGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    operationGraph_.nodes.resize(NODE_COUNT);

    AclTransformer::NormParam inputNormParam;
    inputNormParam.layerNormEps = param_.layerNormEps;
    AclTransformer::LinearParam mixdQkvLinearParam;
    AclTransformer::PositionEmbeddingParam positionEmbeddingParam;
    positionEmbeddingParam.headNum = param_.headNum;
    AclTransformer::SelfAttentionKvCacheFusionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headNum = param_.headNum;
    AclTransformer::LinearParam selfOutLinearParam;
    AclTransformer::AddParam selfResidualAddParam;
    selfResidualAddParam.scale = param_.residualAddScale;
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

    size_t nodeId = 0;
    AclTransformer::OperationGraphNode &inputNormNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &mixdQkvLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &positionEmbeddingNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfOutLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfResidualAddNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfNormNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &ffnNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &ffnLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &ffnResidualAddNode = operationGraph_.nodes.at(nodeId++);

    inputNormNode.operation = inputNormOp;
    inputNormNode.inTensorIds = {IN_HIDDENSTATES_ID, normWeight, normBias};
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
    ffnResidualAddNode.outTensorIds = {OUT_GLMBLOCKOUT_ID};
}
} // namespace AclTransformer
