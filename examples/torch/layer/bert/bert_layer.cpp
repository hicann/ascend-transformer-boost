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
#include "examples/utils/example_utils.h"
#include "acltransformer/plan_builder.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/ops/ffn_operation.h"


void BertLayer(const Json::Value &paramJson, AclTransformer::VariantPack &variantPack)
{
    const uint64_t hiddenStatesId = 0;
    const uint64_t qLinearWeightId = 1;
    const uint64_t qLinearBiasId = 2;
    const uint64_t kLinearWeightId = 3;
    const uint64_t kLinearBiasId = 4;
    const uint64_t vLinearWeightId = 5;
    const uint64_t vLinearBiasId = 6;
    const uint64_t selfOutLinearWeightId = 7;
    const uint64_t selfOutLinearBiasId = 8;
    const uint64_t selfOutNormWeightId = 9;
    const uint64_t selfOutNormBiasId = 10;
    const uint64_t ffnLinearWeightId = 11;
    const uint64_t ffnLinearBiasId = 12;
    const uint64_t bertOutLinearWeightId = 13;
    const uint64_t bertOutLinearBiasId = 14;
    const uint64_t bertOutNormWeightId = 15;
    const uint64_t bertOutNormBiasId = 16;
    const uint64_t attentionMaskId = 17;
    // out
    const uint64_t bertLayerOutId = 18;
    // intermiate
    const uint64_t mixedQueryId = 19;
    const uint64_t mixedKeyId = 20;
    const uint64_t mixedValueId = 21;
    const uint64_t selfAttentionOutId = 22;
    const uint64_t selfLinearOutId = 23;
    const uint64_t selfAddNormOutId = 24;
    const uint64_t ffnOutId = 25;
    const uint64_t bertOutLinearOutId = 26;

    AclTransformer::LinearParam qLinearParam;
    AclTransformer::LinearParam kLinearParam;
    AclTransformer::LinearParam vLinearParam;
    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.transKey = paramJson["transKey"].asBool();
    selfAttentionParam.dk = paramJson["dk"].asInt();
    selfAttentionParam.headNum = paramJson["headNum"].asInt();
    AclTransformer::LinearParam selfOutLinearParam;
    AclTransformer::AddNormParam selfOutAddNormParam;
    AclTransformer::FfnParam ffnParam;
    AclTransformer::LinearParam bertOutLinearParam;
    AclTransformer::AddNormParam bertOutAddNormParam;

    AclTransformer::LinearOperation qLinearOp(qLinearParam);
    AclTransformer::LinearOperation kLinearOp(kLinearParam);
    AclTransformer::LinearOperation vLinearOp(vLinearParam);
    AclTransformer::SelfAttentionOperation selfAttentionOp(selfAttentionParam);
    AclTransformer::LinearOperation selfOutLinearOp(selfOutLinearParam);
    AclTransformer::AddNormOperation selfOutAddNormOp(selfOutAddNormParam);
    AclTransformer::FfnOperation ffnOp(ffnParam);
    AclTransformer::LinearOperation bertOutLinearOp(bertOutLinearParam);
    AclTransformer::AddNormOperation bertOutAddNormOp(bertOutAddNormParam);

    static int64_t graphId = 0;
    AclTransformer::OperationGraph opGraph;
    opGraph.name = "BertLayerGraph_" + std::to_string(graphId++);
    opGraph.inTensorSize = variantPack.inTensors.size();
    opGraph.outTensorSize = variantPack.outTensors.size();
    opGraph.intermediateTensorSize = 8;
    opGraph.nodes.resize(9);

    AclTransformer::OperationGraphNode &qLinearNode = opGraph.nodes.at(0);
    AclTransformer::OperationGraphNode &kLinearNode = opGraph.nodes.at(1);
    AclTransformer::OperationGraphNode &vLinearNode = opGraph.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionNode = opGraph.nodes.at(3);
    AclTransformer::OperationGraphNode &selfOutLinearNode = opGraph.nodes.at(4);
    AclTransformer::OperationGraphNode &selfOutAddNormNode = opGraph.nodes.at(5);
    AclTransformer::OperationGraphNode &ffnNode = opGraph.nodes.at(6);
    AclTransformer::OperationGraphNode &bertOutLinearNode = opGraph.nodes.at(7);
    AclTransformer::OperationGraphNode &bertOutAddNormNode = opGraph.nodes.at(8);

    qLinearNode.operation = &qLinearOp;
    qLinearNode.inTensorIds = {hiddenStatesId, qLinearWeightId, qLinearBiasId};
    qLinearNode.outTensorIds = {mixedQueryId};

    kLinearNode.operation = &kLinearOp;
    kLinearNode.inTensorIds = {hiddenStatesId, kLinearWeightId, kLinearBiasId};
    kLinearNode.outTensorIds = {mixedKeyId};

    vLinearNode.operation = &vLinearOp;
    vLinearNode.inTensorIds = {hiddenStatesId, vLinearWeightId, vLinearBiasId};
    vLinearNode.outTensorIds = {mixedValueId};

    selfAttentionNode.operation = &selfAttentionOp;
    selfAttentionNode.inTensorIds = {mixedQueryId, mixedKeyId, mixedValueId, attentionMaskId};
    selfAttentionNode.outTensorIds = {selfAttentionOutId};

    selfOutLinearNode.operation = &selfOutLinearOp;
    selfOutLinearNode.inTensorIds = {selfAttentionOutId, selfOutLinearWeightId, selfOutLinearBiasId};
    selfOutLinearNode.outTensorIds = {selfLinearOutId};

    selfOutAddNormNode.operation = &selfOutAddNormOp;
    selfOutAddNormNode.inTensorIds = {selfLinearOutId, hiddenStatesId, selfOutNormWeightId, selfOutNormBiasId};
    selfOutAddNormNode.outTensorIds = {selfAddNormOutId};

    ffnNode.operation = &ffnOp;
    ffnNode.inTensorIds = {selfAddNormOutId, ffnLinearWeightId, ffnLinearBiasId};
    ffnNode.outTensorIds = {ffnOutId};

    bertOutLinearNode.operation = &bertOutLinearOp;
    bertOutLinearNode.inTensorIds = {ffnOutId, bertOutLinearWeightId, bertOutLinearBiasId};
    bertOutLinearNode.outTensorIds = {bertOutLinearOutId};

    bertOutAddNormNode.operation = &bertOutAddNormOp;
    bertOutAddNormNode.inTensorIds = {bertOutLinearOutId, selfAddNormOutId, bertOutNormWeightId, bertOutNormBiasId};
    bertOutAddNormNode.outTensorIds = {bertLayerOutId};

    ExecuteOperationGraph(opGraph, variantPack);
}
