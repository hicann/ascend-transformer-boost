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
#ifndef OPERATION_TORCH_H
#define OPERATION_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "acltransformer/operation.h"
#include "acltransformer/plan.h"

class OperationTorch : public torch::CustomClassHolder {
public:
    OperationTorch(std::string opName);
    ~OperationTorch();
    void SetParam(std::string param);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> inTensors);
    void ExecuteOut(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensor);
    c10::intrusive_ptr<OperationTorch> clone() const { return c10::make_intrusive<OperationTorch>(opName_); }

private:
    void CreateAtOutTensors(const std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    void ExecuteOutImpl(std::vector<torch::Tensor> &inTensors, std::vector<torch::Tensor> &outTensor);
    void BuildVariantPack(std::vector<torch::Tensor> &inTensors, std::vector<torch::Tensor> &outTensor,
                          AclTransformer::VariantPack &variantPack);

private:
    std::string opName_;
    std::string param_;
    std::unique_ptr<AclTransformer::Operation> operation_;
    AclTransformer::Plan plan_;
    uint64_t executeCount_ = 0;
};

#endif