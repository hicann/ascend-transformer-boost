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
#ifndef MULTI_ADD_LAYER_TORCH_H
#define MULTI_ADD_LAYER_TORCH_H
#include <torch/script.h>
#include <torch/custom_class.h>

// a + b + c + d
class MultiAddLayerTorch : public torch::CustomClassHolder {
public:
    MultiAddLayerTorch();
    ~MultiAddLayerTorch();
    void Test();
    void Execute(std::vector<torch::Tensor> inTensors, std::vector<torch::Tensor> outTensors);
    c10::intrusive_ptr<MultiAddLayerTorch> clone() const { return c10::make_intrusive<MultiAddLayerTorch>(); }

private:
};

#endif