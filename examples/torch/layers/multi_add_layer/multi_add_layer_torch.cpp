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
#include "multi_add_layer_torch.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "acltransformer/ops/add_operation.h"
#include "examples/utils/example_utils.h"

const uint64_t aTensorId = 0;
const uint64_t bTensorId = 1;
const uint64_t cTensorId = 2;
const uint64_t dTensorId = 3;
const uint64_t resultTensorId = 4;
const uint64_t add0NodeOutTensorId = 5;
const uint64_t add1NodeOutTensorId = 6;

MultiAddLayerTorch::MultiAddLayerTorch() { ASD_LOG(INFO) << "MultiAddLayerTorch::MultiAddLayerTorch"; }

MultiAddLayerTorch::~MultiAddLayerTorch() {}

void MultiAddLayerTorch::Test() { ASD_LOG(INFO) << "MultiAddLayerTorch::Test called"; }

void MultiAddLayerTorch::Execute(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensors)
{
    AclTransformer::VariantPack variantPack;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2AsdTensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2AsdTensor(outTensors.at(i)));
    }

    AclTransformer::AddParam addParam;
    AclTransformer::AddOperation add0Op(addParam);
    AclTransformer::AddOperation add1Op(addParam);
    AclTransformer::AddOperation add2Op(addParam);

    AclTransformer::OperationGraph opGraph;
    opGraph.inTensorSize = variantPack.inTensors.size();
    opGraph.outTensorSize = variantPack.outTensors.size();
    opGraph.intermediateTensorSize = 2;
    opGraph.nodes.resize(3);
    AclTransformer::OperationGraphNode &add0Node = opGraph.nodes.at(0);
    AclTransformer::OperationGraphNode &add1Node = opGraph.nodes.at(1);
    AclTransformer::OperationGraphNode &add2Node = opGraph.nodes.at(2);
    add0Node.operation = &add0Op;
    add0Node.inTensorIds = {aTensorId, bTensorId};
    add0Node.outTensorIds = {add0NodeOutTensorId};

    add1Node.operation = &add1Op;
    add1Node.inTensorIds = {cTensorId, dTensorId};
    add1Node.outTensorIds = {add1NodeOutTensorId};

    add2Node.operation = &add2Op;
    add2Node.inTensorIds = {add0NodeOutTensorId, add1NodeOutTensorId};
    add2Node.outTensorIds = {resultTensorId};

    ExecuteOperationGraph(opGraph, variantPack);
}

TORCH_LIBRARY(MultiAddLayerTorch, m)
{
    m.class_<MultiAddLayerTorch>("MultiAddLayerTorch")
        .def(torch::init<>())
        .def("test", &MultiAddLayerTorch::Test)
        .def("execute", &MultiAddLayerTorch::Execute);
}