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
#include "acltransformer/plan.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/status/status.h>
#include "acltransformer/runner/runner.h"

namespace AclTransformer {

Plan::Plan() {}

Plan::~Plan() {}

AsdOps::Status Plan::Setup(const VariantPack &variantPack) { return runner_->Setup(variantPack); }

uint64_t Plan::GetTilingBufferSize() { return runner_->GetTilingBufferSize(); }

void Plan::FillHostTilingBufferSize(void *hostTilingBuffer, uint64_t tilingBufferSize)
{
    runner_->FillHostTilingBufferSize(hostTilingBuffer, tilingBufferSize);
}

uint64_t Plan::GetWorkspaceBufferSize() { return runner_->GetWorkspaceBufferSize(); }

uint64_t Plan::GetIntermediateBufferSize() { return runner_->GetIntermediateBufferSize(); }

AsdOps::Status Plan::Execute(Handle &handle, VariantPack &variantPack) { return runner_->Execute(handle, variantPack); }
} // namespace AclTransformer