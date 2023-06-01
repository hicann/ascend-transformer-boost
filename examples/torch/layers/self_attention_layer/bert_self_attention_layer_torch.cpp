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
#include "bert_self_attention_layer_torch.h"
#include <thread>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_utils.h"
#include "acltransformer/plan_builder.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/utils/tensor_util.h"

SelfAttentionLayerTorch::SelfAttentionLayerTorch()
{
    ASD_LOG(INFO) << "SelfAttentionLayerTorch::SelfAttentionLayerTorch";
}

SelfAttentionLayerTorch::~SelfAttentionLayerTorch() {}

void SelfAttentionLayerTorch::Test() { ASD_LOG(INFO) << "SelfAttentionLayerTorch::Test called"; }

void SelfAttentionLayerTorch::Execute(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensors)
{
    ASD_LOG(INFO) << "SelfAttentionLayerTorch::Execute start, executeCount:" << executeCount_++;
    torch::Tensor resultTensor = at::zeros(inTensors.at(0).sizes(), inTensors.at(0).options());

    for (size_t i = 0; i < inTensors.size(); ++i) {
        inTensors.at(i) = inTensors.at(i).contiguous();
        ASD_LOG(INFO) << "inTensors[" << i << "].options:" << inTensors.at(i).options()
                      << ", data:" << inTensors.at(i).data_ptr();
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        outTensors.at(i) = outTensors.at(i).contiguous();
        ASD_LOG(INFO) << "outTensors[" << i << "].options:" << outTensors.at(i).options()
                      << ", data:" << outTensors.at(i).data_ptr();
    }
    const uint64_t inputId = 0;
    const uint64_t queryLinearWeightId = 1;
    const uint64_t queryLinearBiasId = 2;
    const uint64_t keyLinearWeightId = 3;
    const uint64_t keyLinearBiasId = 4;
    const uint64_t valueLinearWeightId = 5;
    const uint64_t valueLinearBiasId = 6;
    const uint64_t attentionMaskId = 7;
    // out
    const uint64_t contextId = 8;
    // intermiate
    const uint64_t queryId = 9;
    const uint64_t keyId = 10;
    const uint64_t valueId = 11;

    AclTransformer::VariantPack variantPack;
    BuildVariantPack(inTensors, outTensors, variantPack);

    AclTransformer::LinearParam queryLinearParam;
    AclTransformer::LinearParam keyLinearParam;
    AclTransformer::LinearParam valueLinearParam;
    AclTransformer::SelfAttentionParam selfAttentionParam = {false, 64, 16};
    AclTransformer::LinearOperation queryLinearOp(queryLinearParam);
    AclTransformer::LinearOperation keyLinearOp(keyLinearParam);
    AclTransformer::LinearOperation valueLinearOp(valueLinearParam);
    AclTransformer::SelfAttentionOperation selfAttentionOp(selfAttentionParam);

    AclTransformer::OperationGraph opGraph;
    static int64_t graphId = 0;
    opGraph.name = "BertSelfAttentionGraph_" + std::to_string(graphId++);
    opGraph.inTensorSize = variantPack.inTensors.size();
    opGraph.outTensorSize = variantPack.outTensors.size();
    opGraph.intermediateTensorSize = 3;
    opGraph.nodes.resize(4);
    AclTransformer::OperationGraphNode &queryLinearNode = opGraph.nodes.at(0);
    AclTransformer::OperationGraphNode &keyLinearNode = opGraph.nodes.at(1);
    AclTransformer::OperationGraphNode &valueLinearNode = opGraph.nodes.at(2);
    AclTransformer::OperationGraphNode &selfAttentionNode = opGraph.nodes.at(3);

    queryLinearNode.operation = &queryLinearOp;
    queryLinearNode.inTensorIds = {inputId, queryLinearWeightId, queryLinearBiasId};
    queryLinearNode.outTensorIds = {queryId};

    keyLinearNode.operation = &keyLinearOp;
    keyLinearNode.inTensorIds = {inputId, keyLinearWeightId, keyLinearBiasId};
    keyLinearNode.outTensorIds = {keyId};

    valueLinearNode.operation = &valueLinearOp;
    valueLinearNode.inTensorIds = {inputId, valueLinearWeightId, valueLinearBiasId};
    valueLinearNode.outTensorIds = {valueId};

    selfAttentionNode.operation = &selfAttentionOp;
    selfAttentionNode.inTensorIds = {queryId, keyId, valueId, attentionMaskId};
    selfAttentionNode.outTensorIds = {contextId};

    ExecuteOperationGraph(opGraph, variantPack);
    ASD_LOG(INFO) << "SelfAttentionLayerTorch::Execute success";
}

TORCH_LIBRARY(SelfAttentionLayerTorch, m)
{
    m.class_<SelfAttentionLayerTorch>("SelfAttentionLayerTorch")
        .def(torch::init<>())
        .def("test", &SelfAttentionLayerTorch::Test)
        .def("execute", &SelfAttentionLayerTorch::Execute);
}