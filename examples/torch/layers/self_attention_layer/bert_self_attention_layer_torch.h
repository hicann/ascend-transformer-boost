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
#ifndef BERT_OUTPUT_LAYER_TORCH_H
#define BERT_OUTPUT_LAYER_TORCH_H
#include <torch/script.h>
#include <torch/custom_class.h>

class SelfAttentionLayerTorch : public torch::CustomClassHolder {
public:
    SelfAttentionLayerTorch();
    ~SelfAttentionLayerTorch();
    void Test();
    void Execute(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensors);
    c10::intrusive_ptr<SelfAttentionLayerTorch> clone() const { return c10::make_intrusive<SelfAttentionLayerTorch>(); }

private:
    int64_t executeCount_ = 0;
};

#endif