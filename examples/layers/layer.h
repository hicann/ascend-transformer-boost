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
#ifndef LAYER_H
#define LAYER_H
#include <string>
#include <nlohmann/json.hpp>
#include "acltransformer/operation_graph.h"
#include "acltransformer/variant_pack.h"
#include "acltransformer/plan.h"

namespace AclTransformer {
class Layer {
public:
    Layer(const std::string &layerName, const nlohmann::json &paramJson);
    virtual ~Layer();
    std::string GetName() const;
    virtual AsdOps::Status InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                      AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) = 0;

    void Execute(Handle &handle, AclTransformer::VariantPack &variantPack);

protected:
    void BuildPlan();

protected:
    std::string layerName_;
    uint64_t layerId_ = 0;
    nlohmann::json paramJson_;
    AclTransformer::OperationGraph opGraph_;
    void *lastStream_ = nullptr;
    AclTransformer::Plan plan_;
};
} // namespace AclTransformer
#endif