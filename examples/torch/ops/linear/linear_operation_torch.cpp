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
#include "linear_operation_torch.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/linear_operation.h"
#include "examples/utils/example_utils.h"
#include <json/json.h>

LinearOperationTorch::LinearOperationTorch(std::string param) : param_(param)
{
    ASD_LOG(INFO) << "LinearOperationTorch::LinearOperationTorch";
    Json::Reader paramReader;
    Json::Value paramJson;
    if (!paramReader.parse(param, paramJson)) {
        ASD_LOG(ERROR) << "json parse error";
    }
    AclTransformer::LinearParam linearParam;
    linearParam.transposeA = paramJson["transposeA"].asBool();
    linearParam.transposeB = paramJson["transposeB"].asBool();
    operation_ = new AclTransformer::LinearOperation(linearParam);
}

LinearOperationTorch::~LinearOperationTorch()
{
    if (operation_) {
        delete operation_;
        operation_ = nullptr;
    }
}

void LinearOperationTorch::Test() { ASD_LOG(INFO) << "LinearOperationTorch::Test called"; }

torch::Tensor LinearOperationTorch::Execute(torch::Tensor a, torch::Tensor b, torch::Tensor c)
{
    a = a.contiguous();
    b = b.contiguous();
    c = c.contiguous();
    ASD_LOG(INFO) << "LinearOperationTorch::Execute start";
    ASD_LOG(INFO) << "inTensors[a].options:" << a.options() << ", data:" << a.data_ptr();
    ASD_LOG(INFO) << "inTensors[b].options:" << b.options() << ", data:" << b.data_ptr();
    ASD_LOG(INFO) << "inTensors[c].options:" << c.options() << ", data:" << c.data_ptr();

    torch::Tensor resultTensor;
    if (a.sizes().size() == 3) {
        resultTensor = at::zeros({a.sizes()[0], a.sizes()[1], b.sizes()[0]}, a.options()); // to do shape
    } else {
        resultTensor = at::zeros({a.sizes()[0], b.sizes()[0]}, a.options());
    }
    resultTensor.contiguous();
    ExecuteOperation(operation_, {a, b, c}, {resultTensor});
    ASD_LOG(INFO) << "LinearOperationTorch::Execute end";
    return resultTensor;

    // return at::linear(a, b, c);
}

TORCH_LIBRARY(LinearOperationTorch, m)
{
    m.class_<LinearOperationTorch>("LinearOperationTorch")
        .def(torch::init<std::string>())
        .def("test", &LinearOperationTorch::Test)
        .def("execute", &LinearOperationTorch::Execute);
}