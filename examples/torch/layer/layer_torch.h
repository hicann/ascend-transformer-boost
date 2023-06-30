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
#ifndef LAYER_TORCH_H
#define LAYER_TORCH_H
#include <string>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation_graph.h"
#include "examples/layers/layer.h"

class LayerTorch : public torch::CustomClassHolder {
public:
    LayerTorch(std::string layerName, std::string param);
    ~LayerTorch();
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> inTensors);
    void ExecuteOut(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensors);
    void SetChatGlmWeights(std::vector<torch::Tensor> weightTensors);
    void ExecuteOutChatGlm(torch::Tensor hiddenStateTensor, torch::Tensor positionIdTensor,
                           torch::Tensor attentionMaskTensor, torch::Tensor pastKeyTensor,
                           torch::Tensor pastValueTensor, torch::Tensor blockOutTensor, torch::Tensor presentKey,
                           torch::Tensor presentValue);
    c10::intrusive_ptr<LayerTorch> clone() const { return c10::make_intrusive<LayerTorch>(layerName_, param_); }

private:
    void CreateAtOutTensors(const std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    void ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    void BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                          AclTransformer::VariantPack &variantPack);

    std::string GetLogPrefix();

private:
    std::string layerName_;
    std::string param_;
    AclTransformer::Layer *layer_ = nullptr;
    uint64_t executeCount_ = 0;
    uint64_t layerId_ = 0;
    static uint64_t totalExecuteCount_;
    AsdOps::Timer timer_;
    std::vector<torch::Tensor> chatGlmInTensors_;
    std::vector<torch::Tensor> chatGlmOutTensors_;
};

#endif