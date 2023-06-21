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
#ifndef LLAMA7BBLAYER_H
#define LLAMA7BBLAYER_H
#include "examples/layers/layer.h"

namespace AclTransformer {
class Llama7BLayer : public Layer {
public:
    Llama7BLayer();
    virtual ~Llama7BLayer();
    AsdOps::Status InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) override;
    AsdOps::Status Execute(Handle &handle, VariantPack &variantPack) override;
};
} // namespace AclTransformer
#endif