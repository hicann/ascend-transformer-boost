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
#include "add_norm_operation_torch.h"
#include <json/json.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/add_norm_operation.h"
#include "examples/utils/example_utils.h"

AddNormOperationTorch::AddNormOperationTorch(std::string param) : param_(param)
{
    ASD_LOG(INFO) << "AddNormOperationTorch::AddNormOperationTorch";
    Json::Reader paramReader;
    Json::Value paramJson;
    if (!paramReader.parse(param, paramJson)) {
        ASD_LOG(ERROR) << "json parse error";
    }
    AclTransformer::AddNormParam addNormParam;
    addNormParam.layerNormEps = paramJson["layerNormEps"].asDouble();
    for (int i = 0; i < paramJson["dims"].size(); i++) {
        addNormParam.dims.push_back(paramJson["dims"][i].asInt());
    }
    operation_ = new AclTransformer::AddNormOperation(addNormParam);
}

AddNormOperationTorch::~AddNormOperationTorch()
{
    if (operation_) {
        delete operation_;
        operation_ = nullptr;
    }
}

void AddNormOperationTorch::Test() { ASD_LOG(INFO) << "AddNormOperationTorch::Test called"; }

torch::Tensor AddNormOperationTorch::Execute(torch::Tensor a, torch::Tensor b, torch::Tensor normWeight,
                                             torch::Tensor normBias)
{
    a = a.contiguous();
    b = b.contiguous();
    normWeight = normWeight.contiguous();
    normBias = normBias.contiguous();
    ASD_LOG(INFO) << "AddNormOperationTorch::Execute start, a.device.type:" << a.device().type();
    torch::Tensor resultTensor = at::zeros(a.sizes(), a.options()).contiguous();
    ExecuteOperation(operation_, {a, b, normWeight, normBias}, {resultTensor});
    ASD_LOG(INFO) << "AddNormOperationTorch::Execute end";
    return resultTensor;
    // at::Tensor addResultTensor = at::add(a, b);
    // return at::layer_norm(addResultTensor, normWeight.sizes(), normWeight, normBias, 1e-12);
}

TORCH_LIBRARY(AddNormOperationTorch, m)
{
    m.class_<AddNormOperationTorch>("AddNormOperationTorch")
        .def(torch::init<std::string>())
        .def("test", &AddNormOperationTorch::Test)
        .def("execute", &AddNormOperationTorch::Execute);
}