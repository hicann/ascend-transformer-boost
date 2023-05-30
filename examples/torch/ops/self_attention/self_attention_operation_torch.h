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
#ifndef SELF_ATTENTION_OPERATION_TORCH_H
#define SELF_ATTENTION_OPERATION_TORCH_H
#include <torch/script.h>
#include <torch/custom_class.h>
#include "acltransformer/ops/self_attention_operation.h"

namespace AclTransformer {
class SelfAttentionOperation;
}

class SelfAttentionOperationTorch : public torch::CustomClassHolder {
public:
    SelfAttentionOperationTorch(std::string param);
    ~SelfAttentionOperationTorch();
    void Test();
    torch::Tensor Execute(torch::Tensor aquery, torch::Tensor key, torch::Tensor value, torch::Tensor attentionMask);
    c10::intrusive_ptr<SelfAttentionOperationTorch> clone() const { return c10::make_intrusive<SelfAttentionOperationTorch>(param_); }

private:
    AclTransformer::SelfAttentionOperation *operation_ = nullptr;
    std::string param_;
    AclTransformer::SelfAttentionParam selfAttentionParam_;
};

#endif