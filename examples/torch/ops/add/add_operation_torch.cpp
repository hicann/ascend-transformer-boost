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
#include "add_operation_torch.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/add_operation.h"
#include "examples/utils/example_utils.h"

AddOperationTorch::AddOperationTorch()
{
    ASD_LOG(INFO) << "AddOperationTorch::AddOperationTorch";
    AclTransformer::AddParam param;
    operation_ = new AclTransformer::AddOperation(param);
}

AddOperationTorch::~AddOperationTorch()
{
    if (operation_) {
        delete operation_;
        operation_ = nullptr;
    }
}

void AddOperationTorch::Test() { ASD_LOG(INFO) << "AddOperationTorch::Test called"; }

torch::Tensor AddOperationTorch::Execute(torch::Tensor a, torch::Tensor b)
{
    ASD_LOG(INFO) << "AddOperationTorch::Execute start";
    torch::Tensor resultTensor = at::zeros(a.sizes(), a.options());
    ExecuteOperation(operation_, {&a, &b}, {&resultTensor});
    ASD_LOG(INFO) << "AddOperationTorch::Execute end";
    return resultTensor;
}

TORCH_LIBRARY(AddOperationTorch, m)
{
    m.class_<AddOperationTorch>("AddOperationTorch")
        .def(torch::init<>())
        .def("test", &AddOperationTorch::Test)
        .def("execute", &AddOperationTorch::Execute);
}