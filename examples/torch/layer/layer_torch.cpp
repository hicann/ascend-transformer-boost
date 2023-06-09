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
#include <json/json.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/operation_graph.h"
#include "examples/utils/example_util.h"
#include "acltransformer/plan_builder.h"
#include "layer.h"

LayerTorch::LayerTorch(std::string layerName) : layerName_(layerName)
{
    ASD_LOG(INFO) << "LayerTorch::LayerTorch called, layerName:" << layerName;
}

LayerTorch::~LayerTorch() {}

void LayerTorch::SetParam(std::string param) { param_ = param; }

void LayerTorch::Execute(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensors)
{
    AsdOps::Timer timer;
    ASD_LOG(INFO) << "LayerTorch::Execute start";
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

    AclTransformer::VariantPack variantPack;
    ExampleUtil::BuildVariantPack(inTensors, outTensors, variantPack);

    ExecuteLayer(layerName_, param_, variantPack);

    ASD_LOG(WARN) << "LayerTorch::Execute end, use time:" << timer.ElapsedMicroSecond() << " microsecond";
}

TORCH_LIBRARY(LayerTorch, m)
{
    m.class_<LayerTorch>("LayerTorch")
        .def(torch::init<std::string>())
        .def("execute", &LayerTorch::Execute)
        .def("set_param", &LayerTorch::SetParam);
}