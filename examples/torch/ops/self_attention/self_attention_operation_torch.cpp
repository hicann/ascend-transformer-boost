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
#include "self_attention_operation_torch.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/utils/tensor_util.h"
#include "examples/utils/example_utils.h"
#include <json/json.h>

SelfAttentionOperationTorch::SelfAttentionOperationTorch(std::string param) : param_(param)
{
    ASD_LOG(INFO) << "SelfAttentionOperationTorch::SelfAttentionOperationTorch";
    Json::Reader paramReader;
    Json::Value paramJson;
    if (!paramReader.parse(param, paramJson)) {
        ASD_LOG(ERROR) << "json parse error";
    }
    AclTransformer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.transKey = paramJson["transKey"].asBool();
    selfAttentionParam.dk = paramJson["dk"].asInt();
    selfAttentionParam.headNum = paramJson["headNum"].asInt();
    this->selfAttentionParam_ = selfAttentionParam;
    operation_ = new AclTransformer::SelfAttentionOperation(selfAttentionParam);
}

SelfAttentionOperationTorch::~SelfAttentionOperationTorch()
{
    if (operation_) {
        delete operation_;
        operation_ = nullptr;
    }
}

void SelfAttentionOperationTorch::Test() { ASD_LOG(INFO) << "SelfAttentionOperationTorch::Test called"; }

torch::Tensor SelfAttentionOperationTorch::Execute(torch::Tensor query, torch::Tensor key, torch::Tensor value,
                                                   torch::Tensor attentionMask)
{
    query = query.contiguous();
    key = key.contiguous();
    value = value.contiguous();
    attentionMask = attentionMask.contiguous();
    torch::Tensor resultTensor = torch::zeros(query.sizes(), query.options()).contiguous();
    ExecuteOperation(operation_, {&query, &key, &value, &attentionMask}, {&resultTensor});
    ASD_LOG(INFO) << "SelfAttentionOperationTorch::Execute end";
    return resultTensor;
}

TORCH_LIBRARY(SelfAttentionOperationTorch, m)
{
    m.class_<SelfAttentionOperationTorch>("SelfAttentionOperationTorch")
        .def(torch::init<std::string>())
        .def("test", &SelfAttentionOperationTorch::Test)
        .def("execute", &SelfAttentionOperationTorch::Execute);
}