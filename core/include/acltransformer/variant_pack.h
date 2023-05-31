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
#ifndef ACLTRANSFORMER_VARIANT_PACK_H
#define ACLTRANSFORMER_VARIANT_PACK_H
#include <vector>
#include <string>
#include <asdops/tensor.h>

namespace AclTransformer {
struct VariantPack {
    std::vector<AsdOps::Tensor> inTensors;
    std::vector<AsdOps::Tensor> outTensors;
    void *workspace = nullptr;
    uint64_t workspaceSize = 0;
    std::string ToString() const;
};
} // namespace AclTransformer
#endif