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
#ifndef ACLTRANSFORMER_RUNNER_VARIANT_PACK_H
#define ACLTRANSFORMER_RUNNER_VARIANT_PACK_H
#include <string>
#include <asdops/tensor.h>
#include <asdops/utils/svector/svector.h>

namespace AclTransformer {
struct VariantPack {
    AsdOps::SVector<AsdOps::Tensor> inTensors;
    AsdOps::SVector<AsdOps::Tensor> outTensors;
    void *tilingBuffer = nullptr;
    uint64_t tilingBufferSize = 0;
    void *workspaceBuffer = nullptr;
    uint64_t workspaceBufferSize = 0;
    void *intermediateBuffer = nullptr;
    uint64_t intermediateBufferSize = 0;
    std::string ToString() const;
};
} // namespace AclTransformer
#endif