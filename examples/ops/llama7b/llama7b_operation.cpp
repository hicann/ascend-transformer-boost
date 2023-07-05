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
#include "llama7b_operation.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/ops/add_operation.h"
#include "acltransformer/ops/rms_norm_operation.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/ops/mlp_operation.h"
#include "acltransformer/ops/transpose_operation.h"

namespace AclTransformer {
enum LlamaTensorId {
    hiddenStates = 0,
    normWeight = 1,
    qMixdWeight = 2,
    qMixdBias = 3,
    kMixdWeight = 4,
    kMixdBias = 5,
    vMixdWeight = 6,
    vMixdBias = 7,
    selfOutLinearWeight = 8,
    selfOutLinearBias = 9,
    selfOutNormWeight = 10,
    mlpGateWeight = 11,
    mlpDownWeight = 12,
    mlpUpWeight = 13,
    positionIds = 14,
    cosTable = 15,
    sinTable = 16,
    attentionMask = 17,
    pastKey = 18,
    pastValue = 19,
    // out
    Llam7BOperationOut = 20,
    presentKey = 21,
    presentValue = 22,
    // intermiate
    inputNormOut = 23,
    mixedQ = 24,
    mixedK = 25,
    mixedV = 26,
    positionEmbedQ = 27,
    positionEmbedK = 28,
    transposeVout = 29,
    selfOut = 30,
    selfLinearOut = 31,
    selfResidualAddOut = 32,
    selfNormOut = 33,
    mlpOut = 34,
};

static const uint64_t IN_TENSOR_COUNT = 20;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 13;

Llam7BOperation::Llam7BOperation(const Llam7BParam &param) : GraphOperation("ChatGlm6B28Operation"), param_(param)
{
    BuildGraph();
}

Llam7BOperation::~Llam7BOperation() {}

uint64_t Llam7BOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t Llam7BOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status Llam7BOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                               AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    const AsdOps::Tensor &keyTensor = inTensors.at(pastKey);
    const AsdOps::Tensor &ValueTensor = inTensors.at(pastValue);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = keyTensor.desc;
    outTensorDescs.at(1).dims.at(0) += 1;
    outTensorDescs.at(2) = ValueTensor.desc;
    outTensorDescs.at(2).dims.at(0) += 1;
    return AsdOps::Status::OkStatus();
}

void Llam7BOperation::BuildGraph()
{
    operationGraph_.inTensorSize = IN_TENSOR_COUNT;
    operationGraph_.outTensorSize = OUT_TENSOR_COUNT;
    operationGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    operationGraph_.nodes.resize(NODE_COUNT);

    AclTransformer::RmsNormParam inputNormParam;
    inputNormParam.rmsNormEps = param_.rmsNormEps;
    AclTransformer::LinearParam mixdQLinearParam;
    AclTransformer::LinearParam mixdKLinearParam;
    AclTransformer::LinearParam mixdVLinearParam;
    AclTransformer::PositionEmbedding1dSplitParam qPositionEmbeddingParam;
    qPositionEmbeddingParam.headNum = param_.headNum;
    AclTransformer::PositionEmbedding1dSplitParam kPositionEmbeddingParam;
    kPositionEmbeddingParam.headNum = qPositionEmbeddingParam.headNum;
    AclTransformer::TransposeParam vTransposeParam = {0, 1};
    AclTransformer::SelfAttentionKvCacheParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.dk = param_.dk;
    selfAttentionKvCacheParam.headNum = kPositionEmbeddingParam.headNum;
    selfAttentionKvCacheParam.model = param_.model;
    AclTransformer::LinearParam selfOutLinearParam;
    AclTransformer::AddParam selfResidualAddParam;
    AclTransformer::RmsNormParam selfNormParam;
    selfNormParam.rmsNormEps = inputNormParam.rmsNormEps;
    AclTransformer::MlpParam mlpParam;
    AclTransformer::AddParam mlpResidualAddParam;

    AclTransformer::RmsNormOperation *inputNormOp = new AclTransformer::RmsNormOperation(inputNormParam);
    AclTransformer::LinearOperation *mixdQLinearOp = new AclTransformer::LinearOperation(mixdQLinearParam);
    AclTransformer::LinearOperation *mixdKLinearOp = new AclTransformer::LinearOperation(mixdKLinearParam);
    AclTransformer::LinearOperation *mixdVLinearOp = new AclTransformer::LinearOperation(mixdVLinearParam);
    AclTransformer::PositionEmbedding1dSplitOperation *qPositionEmbeddingOp =
        new AclTransformer::PositionEmbedding1dSplitOperation(qPositionEmbeddingParam);
    AclTransformer::PositionEmbedding1dSplitOperation *kPositionEmbeddingOp =
        new AclTransformer::PositionEmbedding1dSplitOperation(kPositionEmbeddingParam);
    AclTransformer::TransposeOperation *vTransposeOp = new AclTransformer::TransposeOperation(vTransposeParam);
    AclTransformer::SelfAttentionKvCacheOperation *selfAttentionKvCacheOp =
        new AclTransformer::SelfAttentionKvCacheOperation(selfAttentionKvCacheParam);
    AclTransformer::LinearOperation *selfOutLinearOp = new AclTransformer::LinearOperation(selfOutLinearParam);
    AclTransformer::AddOperation *selfResidualAddOp = new AclTransformer::AddOperation(selfResidualAddParam);
    AclTransformer::RmsNormOperation *selfNormOp = new AclTransformer::RmsNormOperation(selfNormParam);
    AclTransformer::MlpOperation *mlpOp = new AclTransformer::MlpOperation(mlpParam);
    AclTransformer::AddOperation *mlpResidualAddOp = new AclTransformer::AddOperation(mlpResidualAddParam);

    size_t nodeId = 0;
    AclTransformer::OperationGraphNode &inputNormNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &mixdQLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &mixdKLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &mixdVLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &qPositionEmbeddingNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &kPositionEmbeddingNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &vTransposeNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfAttentionKvCacheNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfOutLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfResidualAddNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfNormNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &mlpNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &mlpResidualAddNode = operationGraph_.nodes.at(nodeId++);

    inputNormNode.operation = inputNormOp;
    inputNormNode.inTensorIds = {hiddenStates, normWeight};
    inputNormNode.outTensorIds = {inputNormOut};

    mixdQLinearNode.operation = mixdQLinearOp;
    mixdQLinearNode.inTensorIds = {inputNormOut, qMixdWeight, qMixdBias};
    mixdQLinearNode.outTensorIds = {mixedQ};

    mixdKLinearNode.operation = mixdKLinearOp;
    mixdKLinearNode.inTensorIds = {inputNormOut, kMixdWeight, kMixdBias};
    mixdKLinearNode.outTensorIds = {mixedK};

    mixdVLinearNode.operation = mixdVLinearOp;
    mixdVLinearNode.inTensorIds = {inputNormOut, vMixdWeight, vMixdBias};
    mixdVLinearNode.outTensorIds = {mixedV};

    qPositionEmbeddingNode.operation = qPositionEmbeddingOp;
    qPositionEmbeddingNode.inTensorIds = {mixedQ, positionIds, cosTable, sinTable};
    qPositionEmbeddingNode.outTensorIds = {positionEmbedQ};

    kPositionEmbeddingNode.operation = kPositionEmbeddingOp;
    kPositionEmbeddingNode.inTensorIds = {mixedK, positionIds, cosTable, sinTable};
    kPositionEmbeddingNode.outTensorIds = {positionEmbedK};

    vTransposeNode.operation = vTransposeOp;
    vTransposeNode.inTensorIds = {mixedV};
    vTransposeNode.outTensorIds = {transposeVout};

    selfAttentionKvCacheNode.operation = selfAttentionKvCacheOp;
    selfAttentionKvCacheNode.inTensorIds = {positionEmbedQ, positionEmbedK, transposeVout,
                                            attentionMask,  pastKey,        pastValue};
    selfAttentionKvCacheNode.outTensorIds = {selfOut, presentKey, presentValue};
    selfAttentionKvCacheNode.inTensorViewFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorViewFuncs.at(2) = [=](const AsdOps::SVector<int64_t> &oldDims,
                                                           AsdOps::SVector<int64_t> &newDims) {
        newDims = {oldDims.at(0), oldDims.at(1), qPositionEmbeddingParam.headNum,
                   oldDims.at(2) / qPositionEmbeddingParam.headNum};
    };

    selfOutLinearNode.operation = selfOutLinearOp;
    selfOutLinearNode.inTensorIds = {selfOut, selfOutLinearWeight, selfOutLinearBias};
    selfOutLinearNode.outTensorIds = {selfLinearOut};

    selfResidualAddNode.operation = selfResidualAddOp;
    selfResidualAddNode.inTensorIds = {hiddenStates, selfLinearOut};
    selfResidualAddNode.outTensorIds = {selfResidualAddOut};

    selfNormNode.operation = selfNormOp;
    selfNormNode.inTensorIds = {selfResidualAddOut, selfOutNormWeight};
    selfNormNode.outTensorIds = {selfNormOut};

    mlpNode.operation = mlpOp;
    mlpNode.inTensorIds = {selfNormOut, mlpGateWeight, mlpDownWeight, mlpUpWeight};
    mlpNode.outTensorIds = {mlpOut};

    mlpResidualAddNode.operation = mlpResidualAddOp;
    mlpResidualAddNode.inTensorIds = {selfResidualAddOut, mlpOut};
    mlpResidualAddNode.outTensorIds = {Llam7BOperationOut};
}
} // namespace AclTransformer
