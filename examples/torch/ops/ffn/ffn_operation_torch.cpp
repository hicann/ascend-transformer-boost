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
#include "ffn_operation_torch.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/ffn_operation.h"
#include "examples/utils/example_utils.h"

FfnOperationTorch::FfnOperationTorch()
{
    AclTransformer::FfnParam param;
    operation_ = new AclTransformer::FfnOperation(param);
    ASD_LOG(INFO) << "FfnOperationTorch::FfnOperationTorch";
}

FfnOperationTorch::~FfnOperationTorch()
{
    if (operation_) {
        delete operation_;
        operation_ = nullptr;
    }
}

void FfnOperationTorch::Test() { ASD_LOG(INFO) << "FfnOperationTorch::Test called"; }

torch::Tensor FfnOperationTorch::Execute(torch::Tensor a, torch::Tensor b, torch::Tensor c)
{
    a = a.contiguous();
    b = b.contiguous();
    c = c.contiguous();
    torch::Tensor resultTensor;
    if (a.sizes().size() == 3) {
        resultTensor = at::empty({a.sizes()[0], a.sizes()[1], b.sizes()[0]}, a.options()); // to do  shape
    } else {
        resultTensor = at::empty({a.sizes()[0], b.sizes()[0]}, a.options());
    }
    resultTensor = resultTensor.contiguous();

    // at::Tensor outputTensor = at::linear(a, b, c);
    // d = at::gelu(outputTensor);

    ExecuteOperation(operation_, {&a, &b, &c}, {&resultTensor});
    ASD_LOG(INFO) << "FfnOperationTorch::Execute end";
    return resultTensor;
}

TORCH_LIBRARY(FfnOperationTorch, m)
{
    m.class_<FfnOperationTorch>("FfnOperationTorch")
        .def(torch::init<>())
        .def("test", &FfnOperationTorch::Test)
        .def("execute", &FfnOperationTorch::Execute);
}