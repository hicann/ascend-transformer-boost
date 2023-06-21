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
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/plan_builder.h"
#include "examples/layers/bert/bert_layer.h"
#include "examples/layers/chatglm6b/chatglm6b_layer.h"
#include "examples/layers/llama7b_layer/llama7b_layer.h"

LayerTorch::LayerTorch(std::string layerName) : layerName_(layerName)
{
    ASD_LOG(INFO) << "LayerTorch::LayerTorch called, layerName:" << layerName;
    if (layerName == "BertLayer") {
        layer_ = new AclTransformer::BertLayer();
    } else if (layerName == "ChatGlm6BLayer") {
        layer_ = new AclTransformer::ChatGlm6BLayer();
    } else if (layerName == "Llama7BLayer") {
        layer_ = new AclTransformer::Llama7BLayer();
    }
}

LayerTorch::~LayerTorch()
{
    if (layer_) {
        delete layer_;
        layer_ = nullptr;
    }
}

void LayerTorch::SetParam(std::string param)
{
    if (!layer_) {
        ASD_LOG(ERROR) << "layer is null";
        return;
    }

    nlohmann::json paramJson = nlohmann::json::parse(param);
    layer_->SetParam(paramJson);
}

std::vector<torch::Tensor> LayerTorch::Execute(std::vector<torch::Tensor> inTensors)
{
    std::vector<torch::Tensor> outTensors;
    if (!layer_) {
        ASD_LOG(ERROR) << "LayerTorch::Execute fail, layer is null";
        return outTensors;
    }
    AsdOps::Timer timer;

    AclTransformer::VariantPack variantPack;
    ASD_LOG(INFO) << "LayerTorch::Execute start";
    for (size_t i = 0; i < inTensors.size(); ++i) {
        inTensors.at(i) = inTensors.at(i).contiguous();
        // inTensors.at(i) = ExampleUtil::NpuFormatCast(inTensors.at(i));
        ASD_LOG(INFO) << "inTensors[" << i << "].options:" << inTensors.at(i).options()
                      << ", data:" << inTensors.at(i).data_ptr();
        variantPack.inTensors.push_back(ExampleUtil::AtTensor2AsdTensor(inTensors.at(i)));
    }

    ASD_LOG(INFO) << "LayerTorch::Start infer outTensors";
    CreateAtOutTensors(variantPack.inTensors, outTensors);

    for (size_t i = 0; i < outTensors.size(); ++i) {
        outTensors.at(i) = outTensors.at(i).contiguous();
        ASD_LOG(INFO) << "outTensors[" << i << "].options:" << outTensors.at(i).options()
                      << ", data:" << outTensors.at(i).data_ptr();
        variantPack.outTensors.push_back(ExampleUtil::AtTensor2AsdTensor(outTensors.at(i)));
    }

    AclTransformer::Handle handle = {ExampleUtil::GetCurrentStream()};
    if (handle.stream != nullptr) {
        ASD_LOG(INFO) << "LayerTorch::Get Handle success!";
    }
    layer_->Execute(handle, variantPack);

    ASD_LOG(WARN) << "LayerTorch::Execute end, use time:" << timer.ElapsedMicroSecond() << " microsecond";
    return outTensors;
}

void LayerTorch::ExecuteOut(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensors)
{
    ASD_LOG(ERROR) << "not implement";
}

void LayerTorch::CreateAtOutTensors(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                    std::vector<torch::Tensor> &atOutTensors)
{
    AsdOps::SVector<AsdOps::TensorDesc> outTensorDescs;
    layer_->InferShape(inTensors, outTensorDescs);

    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        at::Tensor newTensor = ExampleUtil::CreateAtTensorFromAsdOpsTensorDesc(outTensorDescs.at(i));
        atOutTensors.at(i) = newTensor;
    }
}

TORCH_LIBRARY(LayerTorch, m)
{
    m.class_<LayerTorch>("LayerTorch")
        .def(torch::init<std::string>())
        .def("execute", &LayerTorch::Execute)
        .def("execute_out", &LayerTorch::ExecuteOut)
        .def("set_param", &LayerTorch::SetParam);
}