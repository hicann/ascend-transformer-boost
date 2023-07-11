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
#ifndef ChatGlm6BLayerQuant_H
#define ChatGlm6BLayerQuant_H
#include "examples/layers/layer.h"

namespace AclTransformer {
class ChatGlm6BLayerQuant : public Layer {
public:
    ChatGlm6BLayerQuant(const nlohmann::json &paramJson);
    virtual ~ChatGlm6BLayerQuant();
    AsdOps::Status InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph();
    void BuildFirstGraph();
    void BuildMidGraph();
    void BuildLastGraph();
    void GetParams(std::string param, std::vector<std::string> &data);
};
} // namespace AclTransformer
#endif