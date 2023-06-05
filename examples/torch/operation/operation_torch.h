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
#ifndef OPERATION_TORCH_H
#define OPERATION_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "acltransformer/operation.h"

class OperationTorch : public torch::CustomClassHolder {
public:
    OperationTorch(std::string opName);
    ~OperationTorch();
    void SetParam(std::string param);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> inTensors);
    c10::intrusive_ptr<OperationTorch> clone() const { return c10::make_intrusive<OperationTorch>(opName_); }

private:
    void ExecuteOperation(AclTransformer::Operation *operation, std::vector<torch::Tensor> &atInTensors,
                          std::vector<torch::Tensor> &atOutTensors);
    void CreateAtOutTensors(AclTransformer::Operation *operation, const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                            std::vector<torch::Tensor> &atOutTensors);

private:
    std::string opName_;
    std::string param_;
};

#endif