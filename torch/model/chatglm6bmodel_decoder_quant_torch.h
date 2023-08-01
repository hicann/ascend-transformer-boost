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
#ifndef CHATGLM6BMODEL_DECODER_QUANT_TORCH_H
#define CHATGLM6BMODEL_DECODER_QUANT_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <asdops/utils/time/timer.h>
#include "chatglm6bmodel_quant_param.h"
#include "acltransformer/operation.h"
#include "acltransformer/plan.h"

class ChatGlm6BModelDecoderQuantTorch : public torch::CustomClassHolder {
public:
    ChatGlm6BModelDecoderQuantTorch();
    ~ChatGlm6BModelDecoderQuantTorch();
    void SetParam(std::string param);

    void SetWeight(std::vector<torch::Tensor> weightTensors);

    // outTensor, 28 presendKeyTensors, 28 presentValueTensors
    std::vector<torch::Tensor> Execute(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor,
                                       torch::Tensor cosTableTensor, torch::Tensor sinTableTensor,
                                       torch::Tensor attentionMaskTensor, std::vector<torch::Tensor> pastKeyTensors,
                                       std::vector<torch::Tensor> pastValueTensors, torch::Tensor seqLen);
    void ExecuteOut(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor, torch::Tensor cosTableTensor,
                    torch::Tensor sinTableTensor, torch::Tensor attentionMaskTensor,
                    std::vector<torch::Tensor> pastKeyTensors, std::vector<torch::Tensor> pastValueTensors,
                    torch::Tensor seqLen, torch::Tensor outTensor, std::vector<torch::Tensor> presendKeyTensors,
                    std::vector<torch::Tensor> presentValueTensors);
    c10::intrusive_ptr<ChatGlm6BModelDecoderQuantTorch> clone() const
    {
        return c10::make_intrusive<ChatGlm6BModelDecoderQuantTorch>();
    }

private:
    void BuildVariantPack(int layerId, std::vector<torch::Tensor> &atInTensors, torch::Tensor &outTensor,
                          torch::Tensor &presendKeyTensor, torch::Tensor &presentValueTensor, torch::Tensor &resIn,
                          bool newOut, AclTransformer::VariantPack &variantPack);
    void BuildVariantPackLast(int layerId, std::vector<torch::Tensor> &atInTensors, torch::Tensor &outTensor,
                              torch::Tensor &presendKeyTensor, torch::Tensor &presentValueTensor, bool newOut,
                              AclTransformer::VariantPack &variantPack);
    void ExecuteOutImpl(torch::Tensor &hiddenStateTensor, torch::Tensor &positionIdTensor,
                        torch::Tensor &cosTableTensor, torch::Tensor &sinTableTensor,
                        torch::Tensor &attentionMaskTensor, std::vector<torch::Tensor> &pastKeyTensors,
                        std::vector<torch::Tensor> &pastValueTensors, torch::Tensor seqLen, torch::Tensor &outTensor,
                        std::vector<torch::Tensor> &presendKeyTensors, std::vector<torch::Tensor> &presentValueTensors,
                        bool newOut);
    void ExecuteSingleOperation(int layerId, std::vector<torch::Tensor> &opAtInTensors, torch::Tensor &outTensor,
                                torch::Tensor &presendKeyTensor, torch::Tensor &presentValueTensor,
                                torch::Tensor &resIn, bool newOut);
    void ExecuteLastSingleOperation(int layerId, std::vector<torch::Tensor> &opAtInTensors, torch::Tensor &outTensor,
                                    torch::Tensor &presendKeyTensor, torch::Tensor &presentValueTensor, bool newOut);
    std::string GetSaveTensorDir();

private:
    ChatGlm6BModelQuantParam modelParam_;
    std::vector<std::shared_ptr<AclTransformer::Operation>> operations_;
    std::vector<std::shared_ptr<AclTransformer::Plan>> plans_;
    std::vector<torch::Tensor> weightTensors_;
    uint64_t executeCount_ = 0;
    AclTransformer::Handle handle_;
    AsdOps::Timer timer_;
};

#endif