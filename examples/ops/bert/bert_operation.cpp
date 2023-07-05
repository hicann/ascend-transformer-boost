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
#include "bert_operation.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"

namespace AclTransformer {
enum BERT_TENSOR_ID {
    IN_HIDDENSTATESID = 0,
    qLinearWeightId,
    qLinearBiasId,
    kLinearWeightId,
    kLinearBiasId,
    vLinearWeightId,
    vLinearBiasId,
    selfOutLinearWeightId,
    selfOutLinearBiasId,
    selfOutNormWeightId,
    selfOutNormBiasId,
    ffnLinearWeightId,
    ffnLinearBiasId,
    bertOutLinearWeightId,
    bertOutLinearBiasId,
    bertOutNormWeightId,
    bertOutNormBiasId,
    attentionMaskId,
    OUT_BERTOPERATIONOUTID,
    mixedQueryId,
    mixedKeyId,
    mixedValueId,
    selfAttentionOutId,
    selfLinearOutId,
    selfAddNormOutId,
    ffnOutId,
    bertOutLinearOutId,
};

static const uint64_t IN_TENSOR_COUNT = 18;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t NODE_COUNT = 9;

BertOperation::BertOperation(const BertParam &param) : GraphOperation("BertOperation"), param_(param)
{
    BuildGraph();
}

BertOperation::~BertOperation() {}

uint64_t BertOperation::GetInTensorCount() const { return IN_TENSOR_COUNT; }

uint64_t BertOperation::GetOutTensorCount() const { return OUT_TENSOR_COUNT; }

AsdOps::Status BertOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                             AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    return AsdOps::Status::OkStatus();
}

void BertOperation::BuildGraph()
{
    operationGraph_.inTensorSize = IN_TENSOR_COUNT;
    operationGraph_.outTensorSize = OUT_TENSOR_COUNT;
    operationGraph_.intermediateTensorSize = INTERMEDIATE_TENSOR_COUNT;
    operationGraph_.nodes.resize(NODE_COUNT);

    AclTransformer::LinearParam qLinearParam;
    AclTransformer::LinearParam kLinearParam;
    AclTransformer::LinearParam vLinearParam;
    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.transKey = param_.transKey;
    selfAttentionParam.dk = param_.dk;
    selfAttentionParam.headNum = param_.headNum;
    AclTransformer::LinearParam selfOutLinearParam;
    AclTransformer::AddNormParam selfOutAddNormParam;
    AclTransformer::FfnParam ffnParam;
    AclTransformer::LinearParam bertOutLinearParam;
    AclTransformer::AddNormParam bertOutAddNormParam;

    AclTransformer::LinearOperation *qLinearOp = new AclTransformer::LinearOperation(qLinearParam);
    AclTransformer::LinearOperation *kLinearOp = new AclTransformer::LinearOperation(kLinearParam);
    AclTransformer::LinearOperation *vLinearOp = new AclTransformer::LinearOperation(vLinearParam);
    AclTransformer::SelfAttentionOperation *selfAttentionOp =
        new AclTransformer::SelfAttentionOperation(selfAttentionParam);
    AclTransformer::LinearOperation *selfOutLinearOp = new AclTransformer::LinearOperation(selfOutLinearParam);
    AclTransformer::AddNormOperation *selfOutAddNormOp = new AclTransformer::AddNormOperation(selfOutAddNormParam);
    AclTransformer::FfnOperation *ffnOp = new AclTransformer::FfnOperation(ffnParam);
    AclTransformer::LinearOperation *bertOutLinearOp = new AclTransformer::LinearOperation(bertOutLinearParam);
    AclTransformer::AddNormOperation *bertOutAddNormOp = new AclTransformer::AddNormOperation(bertOutAddNormParam);

    size_t nodeId = 0;
    AclTransformer::OperationGraphNode &qLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &kLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &vLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfAttentionNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfOutLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &selfOutAddNormNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &ffnNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &bertOutLinearNode = operationGraph_.nodes.at(nodeId++);
    AclTransformer::OperationGraphNode &bertOutAddNormNode = operationGraph_.nodes.at(nodeId++);

    qLinearNode.operation = qLinearOp;
    qLinearNode.inTensorIds = {IN_HIDDENSTATESID, qLinearWeightId, qLinearBiasId};
    qLinearNode.outTensorIds = {mixedQueryId};

    kLinearNode.operation = kLinearOp;
    kLinearNode.inTensorIds = {IN_HIDDENSTATESID, kLinearWeightId, kLinearBiasId};
    kLinearNode.outTensorIds = {mixedKeyId};

    vLinearNode.operation = vLinearOp;
    vLinearNode.inTensorIds = {IN_HIDDENSTATESID, vLinearWeightId, vLinearBiasId};
    vLinearNode.outTensorIds = {mixedValueId};

    selfAttentionNode.operation = selfAttentionOp;
    selfAttentionNode.inTensorIds = {mixedQueryId, mixedKeyId, mixedValueId, attentionMaskId};
    selfAttentionNode.outTensorIds = {selfAttentionOutId};

    selfOutLinearNode.operation = selfOutLinearOp;
    selfOutLinearNode.inTensorIds = {selfAttentionOutId, selfOutLinearWeightId, selfOutLinearBiasId};
    selfOutLinearNode.outTensorIds = {selfLinearOutId};

    selfOutAddNormNode.operation = selfOutAddNormOp;
    selfOutAddNormNode.inTensorIds = {selfLinearOutId, IN_HIDDENSTATESID, selfOutNormWeightId, selfOutNormBiasId};
    selfOutAddNormNode.outTensorIds = {selfAddNormOutId};

    ffnNode.operation = ffnOp;
    ffnNode.inTensorIds = {selfAddNormOutId, ffnLinearWeightId, ffnLinearBiasId};
    ffnNode.outTensorIds = {ffnOutId};

    bertOutLinearNode.operation = bertOutLinearOp;
    bertOutLinearNode.inTensorIds = {ffnOutId, bertOutLinearWeightId, bertOutLinearBiasId};
    bertOutLinearNode.outTensorIds = {bertOutLinearOutId};

    bertOutAddNormNode.operation = bertOutAddNormOp;
    bertOutAddNormNode.inTensorIds = {bertOutLinearOutId, selfAddNormOutId, bertOutNormWeightId, bertOutNormBiasId};
    bertOutAddNormNode.outTensorIds = {OUT_BERTOPERATIONOUTID};
}
} // namespace AclTransformer
