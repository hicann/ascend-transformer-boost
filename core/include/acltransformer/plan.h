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
#ifndef ACLTRANSFORMER_PLAN_H
#define ACLTRANSFORMER_PLAN_H

#include <memory>
#include <asdops/utils/status/status.h>
#include "acltransformer/handle.h"
#include "acltransformer/variant_pack.h"

namespace AclTransformer {
class Runner;
class Plan {
public:
    Plan();
    ~Plan();
    AsdOps::Status Setup(const VariantPack &variantPack);
    uint64_t GetTilingBufferSize();
    void FillHostTilingBufferSize(void *hostTilingBuffer, uint64_t tilingBufferSize);
    uint64_t GetWorkspaceBufferSize();
    uint64_t GetIntermediateBufferSize();
    AsdOps::Status Execute(Handle &handle, VariantPack &variantPack);

protected:
    std::unique_ptr<Runner> runner_;
    friend class Operation;
    friend class GraphOperation;
};
} // namespace AclTransformer
#endif