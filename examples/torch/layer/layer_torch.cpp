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
#include "layer_torch.h"
#include <nlohmann/json.hpp>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include <asdops/utils/singleton/singleton.h>
#include <asdops/ops.h>
#include "acltransformer/statistic.h"
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "acltransformer/plan_builder.h"
#include "acltransformer/utils/tensor_util.h"
#include "acltransformer/config.h"
#include "examples/utils/example_util.h"
#include "examples/layers/bert/bert_layer.h"
#include "examples/layers/chatglm6b/chatglm6b_layer.h"
#include "examples/layers/chatglm6b/chatglm6b_fusion_layer.h"
#include "examples/layers/llama7b_layer/llama7b_layer.h"

uint64_t LayerTorch::totalExecuteCount_ = 0;

LayerTorch::LayerTorch(std::string layerName, std::string param) : layerName_(layerName), param_(param)
{
    ASD_LOG(INFO) << "LayerTorch::LayerTorch called, layerName:" << layerName;
    std::vector<AsdOps::Operation *> ops;
    AsdOps::Ops::Instance().GetAllOperations(ops);
    nlohmann::json paramJson = nlohmann::json::parse(param);
    layerId_ = paramJson["layerId"].get<int>();
    if (layerName == "BertLayer") {
        layer_ = new AclTransformer::BertLayer(paramJson);
    } else if (layerName == "ChatGlm6BLayer") {
        layer_ = new AclTransformer::ChatGlm6BLayer(paramJson);
    } else if (layerName == "ChatGlm6BFusionLayer") {
        layer_ = new AclTransformer::ChatGlm6BFusionLayer(paramJson);
    } else if (layerName == "Llama7BLayer") {
        layer_ = new AclTransformer::Llama7BLayer(paramJson);
    }
    ASD_LOG(INFO) << "LayerTorch::LayerTorch end";
}

LayerTorch::~LayerTorch()
{
    if (layer_) {
        delete layer_;
        layer_ = nullptr;
    }
}

std::vector<torch::Tensor> LayerTorch::Execute(std::vector<torch::Tensor> atInTensors)
{
    if (!layer_) {
        ASD_LOG(FATAL) << "LayerTorch::Execute fail, layer is null";
    }

    ExampleUtil::ContiguousAtTensor(atInTensors);
    std::vector<torch::Tensor> atOutTensors;
    CreateAtOutTensors(atInTensors, atOutTensors);
    ExampleUtil::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors);
    return atOutTensors;
}

void LayerTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors)
{
    if (!layer_) {
        ASD_LOG(FATAL) << "LayerTorch::Execute fail, layer is null";
    }

    ExampleUtil::ContiguousAtTensor(atInTensors);
    ExampleUtil::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors);
}

void LayerTorch::ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors)
{
    ASD_LOG(INFO) << GetLogPrefix() << " ExecuteOut execCount:" << executeCount_;
    AsdOps::Timer timer;

    AclTransformer::Handle handle = {ExampleUtil::GetCurrentStream()};

    AclTransformer::VariantPack variantPack;
    BuildVariantPack(atInTensors, atOutTensors, variantPack);

    layer_->Execute(handle, variantPack);

    AsdOps::GetSingleton<AclTransformer::Statistic>().totalTime += timer.ElapsedMicroSecond();
    ASD_LOG(FATAL) << GetLogPrefix() << " totalExecuteCount:" << totalExecuteCount_
                   << ", executeCount:" << executeCount_ << ", statistic:["
                   << AsdOps::GetSingleton<AclTransformer::Statistic>().ToString() << "]";
    AsdOps::GetSingleton<AclTransformer::Statistic>().Reset();
    executeCount_++;
    totalExecuteCount_++;
}

void LayerTorch::CreateAtOutTensors(const std::vector<torch::Tensor> &atInTensors,
                                    std::vector<torch::Tensor> &atOutTensors)
{
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;

    AsdOps::SVector<AsdOps::Tensor> inTensors;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        auto &atInTensor = atInTensors.at(i);
        AsdOps::Tensor inTensor = ExampleUtil::AtTensor2AsdTensor(atInTensor);
        inTensors.push_back(inTensor);
        ASD_LOG(INFO) << "infer shape inTensors[" << i
                      << "]:" << AclTransformer::TensorUtil::AsdOpsTensorToString(inTensors.at(i));
    }
    layer_->InferShape(inTensors, outTensorDescs);

    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ASD_LOG(INFO) << "infer shape outTensorDescs[" << i
                      << "]:" << AclTransformer::TensorUtil::AsdOpsTensorDescToString(outTensorDescs.at(i));
        at::Tensor newTensor = ExampleUtil::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(i));
        atOutTensors.at(i) = newTensor;
    }
}

void LayerTorch::BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                                  AclTransformer::VariantPack &variantPack)
{
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ASD_LOG(INFO) << "inTensors[" << i << "].options:" << atInTensors.at(i).options()
                      << ", data:" << atInTensors.at(i).data_ptr()
                      << ", storage_offset:" << atInTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atInTensors.at(i));
        variantPack.inTensors.push_back(ExampleUtil::AtTensor2AsdTensor(atInTensors.at(i)));
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + layerName_ + "/intensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atInTensors.at(i), filePath);
            ASD_LOG(INFO) << layer_->GetName() << " save tensor:" << filePath;
        }
    }

    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ASD_LOG(INFO) << "atOutTensors[" << i << "].options:" << atOutTensors.at(i).options()
                      << ", data:" << atOutTensors.at(i).data_ptr()
                      << ", storage_offset:" << atOutTensors.at(i).storage_offset()
                      << ", format:" << ExampleUtil::GetTensorNpuFormat(atOutTensors.at(i));
        variantPack.outTensors.push_back(ExampleUtil::AtTensor2AsdTensor(atOutTensors.at(i)));
        if (AsdOps::GetSingleton<AclTransformer::Config>().IsSaveTensor()) {
            std::string filePath = AclTransformer::Config::GetSaveTensorDir() + "/" + std::to_string(executeCount_) +
                                   "_" + layerName_ + "/outtensor" + std::to_string(i) + ".pth";
            ExampleUtil::SaveTensor(atOutTensors.at(i), filePath);
            ASD_LOG(INFO) << layer_->GetName() << " save tensor:" << filePath;
        }
    }
}

std::string LayerTorch::GetLogPrefix() { return layerName_ + "_" + std::to_string(layerId_); }

TORCH_LIBRARY(LayerTorch, m)
{
    m.class_<LayerTorch>("LayerTorch")
        .def(torch::init<std::string, std::string>())
        .def("execute", &LayerTorch::Execute)
        .def("execute_out", &LayerTorch::ExecuteOut);
}