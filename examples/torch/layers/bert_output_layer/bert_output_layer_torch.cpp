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
#include "bert_output_layer_torch.h"
#include <sys/stat.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/filesystem/filesystem.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_utils.h"
#include "acltransformer/plan_builder.h"
#include "acltransformer/ops/linear_operation.h"
#include "acltransformer/ops/add_norm_operation.h"
#include "examples/utils/example_utils.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"

BertOutputLayerTorch::BertOutputLayerTorch() { ASD_LOG(INFO) << "BertOutputLayerTorch::BertOutputLayerTorch"; }

BertOutputLayerTorch::~BertOutputLayerTorch() {}

void BertOutputLayerTorch::Test() { ASD_LOG(INFO) << "BertOutputLayerTorch::Test called"; }

void BertOutputLayerTorch::Execute(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensors)
{
    AclTransformer::OperationGraph opGraph;
    static int64_t graphId = 0;
    opGraph.name = "BertOutputLayerGraph_" + std::to_string(graphId++);
    ASD_LOG(INFO) << "BertOutputLayerTorch::Execute start, executeCount:" << executeCount_++;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        inTensors.at(i) = inTensors.at(i).contiguous();
        ASD_LOG(INFO) << "inTensors[" << i << "].options:" << inTensors.at(i).options()
                      << ", data:" << inTensors.at(i).data_ptr();
        if (AclTransformer::Config::IsSaveTensor()) {
            AsdOps::FileSystem::Makedirs("savetensor/" + opGraph.name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            std::string filePath = "savetensor/" + opGraph.name + "/inTensor" + std::to_string(i) + ".pth";
            torch::save(inTensors.at(i), filePath);
        }
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        outTensors.at(i) = outTensors.at(i).contiguous();
        ASD_LOG(INFO) << "outTensors[" << i << "].options:" << outTensors.at(i).options()
                      << ", data:" << outTensors.at(i).data_ptr();
        if (AclTransformer::Config::IsSaveTensor()) {
            std::string filePath = "savetensor/" + opGraph.name + "/outTensor" + std::to_string(i) + ".pth";
            torch::save(outTensors.at(i), filePath);
        }
    }

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

    AclTransformer::VariantPack variantPack;

    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2AsdTensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2AsdTensor(outTensors.at(i)));
    }

    AclTransformer::LinearParam linearParam;
    AclTransformer::AddNormParam addNormParam;
    AclTransformer::LinearOperation linearOp(linearParam);
    AclTransformer::AddNormOperation addNormOp(addNormParam);

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
    ASD_LOG(INFO) << "BertOutputLayerTorch::Execute success";
}

TORCH_LIBRARY(BertOutputLayerTorch, m)
{
    m.class_<BertOutputLayerTorch>("BertOutputLayerTorch")
        .def(torch::init<>())
        .def("test", &BertOutputLayerTorch::Test)
        .def("execute", &BertOutputLayerTorch::Execute);
}