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
#ifndef GLM130BMODEL_DECODER_TORCH_H
#define GLM130BMODEL_DECODER_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <asdops/utils/time/timer.h>
#include "glm130bmodel_param.h"
#include "acltransformer/operation.h"
#include "acltransformer/plan.h"

class Glm130BModelDecoderTorch : public torch::CustomClassHolder {
public:
    Glm130BModelDecoderTorch();
    ~Glm130BModelDecoderTorch();
    void SetParam(std::string param);

    // 每个layer 12个权重
    void SetWeight(std::vector<torch::Tensor> weightTensors);

    // outTensor, 70 presentKeyTensors, 70 presentValueTensors
    std::vector<torch::Tensor> Execute(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor,
                                       torch::Tensor cosTableTensor, torch::Tensor sinTableTensor,
                                       torch::Tensor attentionMaskTensor, std::vector<torch::Tensor> pastKeyTensors,
                                       std::vector<torch::Tensor> pastValueTensors);
    void ExecuteOut(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor, torch::Tensor cosTableTensor,
                    torch::Tensor sinTableTensor, torch::Tensor attentionMaskTensor,
                    std::vector<torch::Tensor> pastKeyTensors, std::vector<torch::Tensor> pastValueTensors,
                    torch::Tensor outTensor, std::vector<torch::Tensor> presentKeyTensors,
                    std::vector<torch::Tensor> presentValueTensors);
    c10::intrusive_ptr<Glm130BModelDecoderTorch> clone() const
    {
        return c10::make_intrusive<Glm130BModelDecoderTorch>();
    }

private:
    void BuildVariantPack(int layerId, std::vector<torch::Tensor> &atInTensors, torch::Tensor &outTensor,
                          torch::Tensor &presentKeyTensor, torch::Tensor &presentValueTensor, bool newOut,
                          AclTransformer::VariantPack &variantPack);
    void ExecuteOutImpl(torch::Tensor &hiddenStateTensor, torch::Tensor &positionIdTensor,
                        torch::Tensor &cosTableTensor, torch::Tensor &sinTableTensor,
                        torch::Tensor &attentionMaskTensor, std::vector<torch::Tensor> &pastKeyTensors,
                        std::vector<torch::Tensor> &pastValueTensors, torch::Tensor &outTensor,
                        std::vector<torch::Tensor> &presentKeyTensors, std::vector<torch::Tensor> &presentValueTensors,
                        bool newOut);
    // IN:hiddenStateTensor+12个权重+positionIdTensor+cosTable+sinTable+attentionMaskTensor+pastKeyTensor+pastValueTensor
    // OUT:outTensor + presentKey + presentValue
    void ExecuteSingleOperation(int layerId, std::vector<torch::Tensor> &opAtInTensors, torch::Tensor &outTensor,
                                torch::Tensor &presendKeyTensor, torch::Tensor &presentValueTensor, bool newOut);
    std::string GetSaveTensorDir();

private:
    Glm130BModelParam modelParam_;
    std::vector<std::shared_ptr<AclTransformer::Operation>> operations_;
    std::vector<std::shared_ptr<AclTransformer::Plan>> plans_;
    std::vector<torch::Tensor> weightTensors_;
    uint64_t executeCount_ = 0;
    AclTransformer::Handle handle_;
    AsdOps::Timer timer_;
};

#endif