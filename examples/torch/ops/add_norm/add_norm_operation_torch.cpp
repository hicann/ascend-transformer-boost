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
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/add_norm_operation.h"
#include "examples/utils/example_utils.h"

AddNormOperationTorch::AddNormOperationTorch()
{
    ASD_LOG(INFO) << "AddNormOperationTorch::AddNormOperationTorch";
    AclTransformer::AddNormParam param;
    operation_ = new AclTransformer::AddNormOperation(param);
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
        .def(torch::init<>())
        .def("test", &AddNormOperationTorch::Test)
        .def("execute", &AddNormOperationTorch::Execute);
}