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
#include "llama7b_layer.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/plan_builder.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/transpose_operation.h"

namespace AclTransformer {
Llama7BLayer::Llama7BLayer() : Layer("Llama7BLayer") {}

Llama7BLayer::~Llama7BLayer() {}

AsdOps::Status Llama7BLayer::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                          AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensors.size() != 20) {
        return AsdOps::Status::FailStatus(1, "in tensor size != 20");
    }
    const AsdOps::Tensor &keyTensor = inTensors.at(18);
    const AsdOps::Tensor &ValueTensor = inTensors.at(19);

    outTensorDescs.resize(3);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = keyTensor.desc;
    outTensorDescs.at(1).dims.at(0) += 1;
    outTensorDescs.at(2) = ValueTensor.desc;
    outTensorDescs.at(2).dims.at(0) += 1;
    return AsdOps::Status::OkStatus();
}

AsdOps::Status Llama7BLayer::Execute(Handle &handle, VariantPack &variantPack)
{ // in
    const uint64_t hiddenStates = 0;
    const uint64_t normWeight = 1;
    const uint64_t qMixdWeight = 2;
    const uint64_t qMixdBias = 3;
    const uint64_t kMixdWeight = 4;
    const uint64_t kMixdBias = 5;
    const uint64_t vMixdWeight = 6;
    const uint64_t vMixdBias = 7;
    const uint64_t selfOutLinearWeight = 8;
    const uint64_t selfOutLinearBias = 9;
    const uint64_t selfOutNormWeight = 10;
    const uint64_t mlpGateWeight = 11;
    const uint64_t mlpDownWeight = 12;
    const uint64_t mlpUpWeight = 13;
    const uint64_t positionIds = 14;
    const uint64_t cosTable = 15;
    const uint64_t sinTable = 16;
    const uint64_t attentionMask = 17;
    const uint64_t pastKey = 18;
    const uint64_t pastValue = 19;
    // out
    const uint64_t llama7bLayerOut = 20;
    const uint64_t presentKey = 21;
    const uint64_t presentValue = 22;
    // intermiate
    const uint64_t inputNormOut = 23;
    const uint64_t mixedQ = 24;
    const uint64_t mixedK = 25;
    const uint64_t mixedV = 26;
    const uint64_t positionEmbedQ = 27;
    const uint64_t positionEmbedK = 28;
    const uint64_t transposeVout = 29;
    const uint64_t selfOut = 30;
    const uint64_t selfLinearOut = 31;
    const uint64_t selfResidualAddOut = 32;
    const uint64_t selfNormOut = 33;
    const uint64_t mlpOut = 34;

    AclTransformer::RmsNormParam inputNormParam;
<<<<<<< HEAD
    inputNormParam.rmsNormEps = paramJson_["rmsNormEps"].get<double>();
    AclTransformer::LinearParam mixdQLinearParam;
    AclTransformer::LinearParam mixdKLinearParam;
    AclTransformer::LinearParam mixdVLinearParam;
    AclTransformer::PositionEmbedding1dSplitParam qPositionEmbeddingParam;
    qPositionEmbeddingParam.headNum = paramJson_["headNum"].get<int>();
    AclTransformer::PositionEmbedding1dSplitParam kPositionEmbeddingParam;
=======
    inputNormParam.layerNormEps = paramJson_["rmsNormEps"].get<double>();
    AclTransformer::LinearParam mixdQLinearParam;
    AclTransformer::LinearParam mixdKLinearParam;
    AclTransformer::LinearParam mixdVLinearParam;
    AclTransformer::PositionEmbeddingParam qPositionEmbeddingParam;
    qPositionEmbeddingParam.headNum = paramJson_["headNum"].get<int>();
    AclTransformer::PositionEmbeddingParam kPositionEmbeddingParam;
>>>>>>> fix llama layer
    kPositionEmbeddingParam.headNum = qPositionEmbeddingParam.headNum;
    AclTransformer::TransposeParam vTransposeParam = {0, 1};
    AclTransformer::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.dk = paramJson_["dk"].get<int>();
<<<<<<< HEAD
    selfAttentionKvCacheParam.headNum = kPositionEmbeddingParam.headNum;
=======
    selfAttentionKvCacheParam.headNum = positionEmbeddingParam.headNum;
>>>>>>> fix llama layer
    selfAttentionKvCacheParam.model = paramJson_["model"].get<std::string>();
    AclTransformer::LinearParam selfOutLinearParam;
    AclTransformer::AddParam selfResidualAddParam;
    AclTransformer::RmsNormParam selfNormParam;
<<<<<<< HEAD
    selfNormParam.rmsNormEps = inputNormParam.rmsNormEps;
=======
    selfNormParam.layerNormEps = inputNormParam.layerNormEps;
>>>>>>> fix llama layer
    AclTransformer::MlpParam mlpParam;
    AclTransformer::AddParam mlpResidualAddParam;

    AclTransformer::RmsNormOperation inputNormOp(inputNormParam);
    AclTransformer::LinearOperation mixdQLinearOp(mixdQLinearParam);
    AclTransformer::LinearOperation mixdKLinearOp(mixdKLinearParam);
    AclTransformer::LinearOperation mixdVLinearOp(mixdVLinearParam);
<<<<<<< HEAD
    AclTransformer::PositionEmbedding1dSplitOperation qPositionEmbeddingOp(qPositionEmbeddingParam);
    AclTransformer::PositionEmbedding1dSplitOperation kPositionEmbeddingOp(kPositionEmbeddingParam);
=======
    AclTransformer::PositionEmbeddingOperation qPositionEmbeddingOp(qPositionEmbeddingParam);
    AclTransformer::PositionEmbeddingOperation kPositionEmbeddingOp(kPositionEmbeddingParam);
>>>>>>> fix llama layer
    AclTransformer::TransposeOperation vTransposeOp(vTransposeParam);
    AclTransformer::SelfAttentionKvCacheOperation selfAttentionKvCacheOp(selfAttentionKvCacheParam);
    AclTransformer::LinearOperation selfOutLinearOp(selfOutLinearParam);
    AclTransformer::AddOperation selfResidualAddOp(selfResidualAddParam);
    AclTransformer::RmsNormOperation selfNormOp(selfNormParam);
    AclTransformer::MlpOperation mlpOp(mlpParam);
    AclTransformer::AddOperation mlpResidualAddOp(mlpResidualAddParam);

    static int64_t graphId = 0;
    AclTransformer::OperationGraph opGraph;
    opGraph.name = "GlmBlockGraph_" + std::to_string(graphId++);
    opGraph.inTensorSize = variantPack.inTensors.size();
    opGraph.outTensorSize = variantPack.outTensors.size();
    opGraph.intermediateTensorSize = 12;
    opGraph.nodes.resize(13);

    AclTransformer::OperationGraphNode &inputNormNode = opGraph.nodes.at(0);
    AclTransformer::OperationGraphNode &mixdQLinearNode = opGraph.nodes.at(1);
    AclTransformer::OperationGraphNode &mixdKLinearNode = opGraph.nodes.at(2);
    AclTransformer::OperationGraphNode &mixdVLinearNode = opGraph.nodes.at(3);
    AclTransformer::OperationGraphNode &qPositionEmbeddingNode = opGraph.nodes.at(4);
    AclTransformer::OperationGraphNode &kPositionEmbeddingNode = opGraph.nodes.at(5);
    AclTransformer::OperationGraphNode &vTransposeNode = opGraph.nodes.at(6);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = opGraph.nodes.at(7);
    AclTransformer::OperationGraphNode &selfOutLinearNode = opGraph.nodes.at(8);
    AclTransformer::OperationGraphNode &selfResidualAddNode = opGraph.nodes.at(9);
    AclTransformer::OperationGraphNode &selfNormNode = opGraph.nodes.at(10);
    AclTransformer::OperationGraphNode &mlpNode = opGraph.nodes.at(11);
    AclTransformer::OperationGraphNode &mlpResidualAddNode = opGraph.nodes.at(12);

    inputNormNode.operation = &inputNormOp;
    inputNormNode.inTensorIds = {hiddenStates, normWeight};
    inputNormNode.outTensorIds = {inputNormOut};

    mixdQLinearNode.operation = &mixdQLinearOp;
    mixdQLinearNode.inTensorIds = {inputNormOut, qMixdWeight, qMixdBias};
    mixdQLinearNode.outTensorIds = {mixedQ};

    mixdKLinearNode.operation = &mixdKLinearOp;
    mixdKLinearNode.inTensorIds = {inputNormOut, kMixdWeight, kMixdBias};
    mixdKLinearNode.outTensorIds = {mixedK};

    mixdVLinearNode.operation = &mixdVLinearOp;
    mixdVLinearNode.inTensorIds = {inputNormOut, vMixdWeight, vMixdBias};
    mixdVLinearNode.outTensorIds = {mixedV};

    qPositionEmbeddingNode.operation = &qPositionEmbeddingOp;
    qPositionEmbeddingNode.inTensorIds = {mixedQ, positionIds, cosTable, sinTable};
    qPositionEmbeddingNode.outTensorIds = {positionEmbedQ};

    kPositionEmbeddingNode.operation = &kPositionEmbeddingOp;
    kPositionEmbeddingNode.inTensorIds = {mixedK, positionIds, cosTable, sinTable};
    kPositionEmbeddingNode.outTensorIds = {positionEmbedK};

    vTransposeNode.operation = &vTransposeOp;
    vTransposeNode.inTensorIds = {mixedV};
    vTransposeNode.outTensorIds = {transposeVout};

    selfAttentionKvCacheNode.operation = &selfAttentionKvCacheOp;
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedQ, positionEmbedK, transposeVout, attentionMask, pastKey, pastValue};
    selfAttentionKvCacheNode.outTensorIds = {selfOut, presentKey, presentValue};
<<<<<<< HEAD
    selfAttentionKvCacheNode.inTensorViewFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorViewFuncs.at(2) = 
    [=](const AsdOps::SVector<int64_t> &oldDims, AsdOps::SVector<int64_t> &newDims)
    {
        newDims = {oldDims.at(0), oldDims.at(1), qPositionEmbeddingParam.headNum, oldDims.at(2) / qPositionEmbeddingParam.headNum};
    };
=======
>>>>>>> fix llama layer

    selfOutLinearNode.operation = &selfOutLinearOp;
    selfOutLinearNode.inTensorIds = {selfOut, selfOutLinearWeight, selfOutLinearBias};
    selfOutLinearNode.outTensorIds = {selfLinearOut};

    selfResidualAddNode.operation = &selfResidualAddOp;
    selfResidualAddNode.inTensorIds = {hiddenStates, selfLinearOut};
    selfResidualAddNode.outTensorIds = {selfResidualAddOut};

    selfNormNode.operation = &selfNormOp;
    selfNormNode.inTensorIds = {selfResidualAddOut, selfOutNormWeight};
    selfNormNode.outTensorIds = {selfNormOut};

    mlpNode.operation = &mlpOp;
    mlpNode.inTensorIds = {selfNormOut, mlpGateWeight, mlpDownWeight, mlpUpWeight};
    mlpNode.outTensorIds = {mlpOut};

    mlpResidualAddNode.operation = &mlpResidualAddOp;
    mlpResidualAddNode.inTensorIds = {selfResidualAddOut, mlpOut};
    mlpResidualAddNode.outTensorIds = {llama7bLayerOut};

    ExampleUtil::ExecuteOperationGraph(opGraph, variantPack);
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer
