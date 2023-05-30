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
#ifndef FFN_OPERATION_TORCH_H
#define FFN_OPERATION_TORCH_H
#include <torch/script.h>
#include <torch/custom_class.h>
#include <acltransformer/ops/ffn_operation.h>

class FfnOperationTorch : public torch::CustomClassHolder {
public:
    FfnOperationTorch();
    ~FfnOperationTorch();
    void Test();
    torch::Tensor Execute(torch::Tensor a, torch::Tensor b, torch::Tensor c);
    c10::intrusive_ptr<FfnOperationTorch> clone() const { return c10::make_intrusive<FfnOperationTorch>(); }

private:
    AclTransformer::FfnOperation *operation_ = nullptr;
};

#endif