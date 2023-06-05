/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

void BertOutputAttentionLayer(const Json::Value &paramJson, AclTransformer::VariantPack &variantPack)
{
    const uint64_t inputId = 0;
    const uint64_t linearWeightId = 1;
    const uint64_t linearBiasId = 2;
    const uint64_t residualAddInId = 3;
    const uint64_t normWeightId = 4;
    const uint64_t normBiasId = 5;
    // out
    const uint64_t bertOutId = 6;
    // intermiate
    const uint64_t linearOutId = 7;

    AclTransformer::LinearParam linearParam;
    AclTransformer::AddNormParam addNormParam;
    AclTransformer::LinearOperation linearOp(linearParam);
    AclTransformer::AddNormOperation addNormOp(addNormParam);

    AclTransformer::OperationGraph opGraph;
    opGraph.inTensorSize = variantPack.inTensors.size();
    opGraph.outTensorSize = variantPack.outTensors.size();
    opGraph.intermediateTensorSize = 1;
    opGraph.nodes.resize(2);
    AclTransformer::OperationGraphNode &linearNode = opGraph.nodes.at(0);
    AclTransformer::OperationGraphNode &addNormNode = opGraph.nodes.at(1);
    linearNode.operation = &linearOp;
    linearNode.inTensorIds = {inputId, linearWeightId, linearBiasId};
    linearNode.outTensorIds = {linearOutId};

    addNormNode.operation = &addNormOp;
    addNormNode.inTensorIds = {linearOutId, residualAddInId, normWeightId, normBiasId};
    addNormNode.outTensorIds = {bertOutId};

    ExecuteOperationGraph(opGraph, variantPack);
}