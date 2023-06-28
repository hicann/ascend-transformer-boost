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
#include "acltransformer/runner.h"

namespace AclTransformer {
Runner::Runner(const std::string &name) : name_(name) {}

Runner::~Runner() {}

std::string Runner::GetName() const { return name_; }

AsdOps::Status Runner::Setup(const RunnerVariantPack &runnerVariantPack)
{
    AsdOps::Status st = IsConsistent(runnerVariantPack);
    if (!st.Ok()) {
        return st;
    }
    return SetupImpl(runnerVariantPack);
}

uint64_t Runner::GetIntermediateBufferSize() { return GetIntermediateBufferSizeImpl(); }

uint64_t Runner::GetTilingBufferSize() { return GetTilingBufferSizeImpl(); }

void Runner::FillHostTilingBufferSize(void *hostTilingBuffer, uint64_t tilingBufferSize)
{
    FillHostTilingBufferSizeImpl(hostTilingBuffer, tilingBufferSize);
}

uint64_t Runner::GetWorkspaceBufferSize() { return GetWorkspaceBufferSizeImpl(); }

AsdOps::Status Runner::Execute(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    AsdOps::Status st = IsConsistent(runnerVariantPack);
    if (!st.Ok()) {
        return st;
    }
    return ExecuteImpl(handle, runnerVariantPack);
}

AsdOps::Status Runner::IsConsistent(const RunnerVariantPack &runnerVariantPack) { return AsdOps::Status::OkStatus(); }

AsdOps::Status Runner::SetupImpl(const RunnerVariantPack &runnerVariantPack) { return AsdOps::Status::OkStatus(); }

uint64_t Runner::GetIntermediateBufferSizeImpl() { return 0; }

uint64_t Runner::GetTilingBufferSizeImpl() { return 0; }

void Runner::FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize) {}

uint64_t Runner::GetWorkspaceBufferSizeImpl() { return 0; }

AsdOps::Status Runner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    return AsdOps::Status::OkStatus();
}
} // namespace AclTransformer