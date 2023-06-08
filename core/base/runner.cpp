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

AsdOps::Status Runner::Setup(const VariantPack &variantPack)
{
    AsdOps::Status st = IsConsistent(variantPack);
    if (!st.Ok()) {
        return st;
    }
    return SetupImpl(variantPack);
}

uint64_t Runner::GetWorkspaceSize() { return GetWorkspaceSizeImpl(); }

AsdOps::Status Runner::Execute(Handle &handle, VariantPack &variantPack)
{
    AsdOps::Status st = IsConsistent(variantPack);
    if (!st.Ok()) {
        return st;
    }
    return ExecuteImpl(handle, variantPack);
}

AsdOps::Status Runner::IsConsistent(const VariantPack &variantPack) { return AsdOps::Status::OkStatus(); }

AsdOps::Status Runner::SetupImpl(const VariantPack &variantPack) { return AsdOps::Status::OkStatus(); }

uint64_t Runner::GetWorkspaceSizeImpl() { return 0; }

AsdOps::Status Runner::ExecuteImpl(Handle &handle, VariantPack &variantPack) { return AsdOps::Status::OkStatus(); }
} // namespace AclTransformer