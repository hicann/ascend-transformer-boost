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
#ifndef ACLTRANSFORMER_RUNNER_H
#define ACLTRANSFORMER_RUNNER_H
#include <string>
#include "acltransformer/handle.h"
#include "acltransformer/variant_pack.h"
#include "asdops/utils/status/status.h"

namespace AclTransformer {
class Runner {
public:
    Runner(const std::string &name);
    virtual ~Runner() = default;
    virtual AsdOps::Status Init() = 0;
    virtual AsdOps::Status Setup(Handle &handle, VariantPack &runInfo) = 0;
    virtual uint64_t GetWorkspaceSize() = 0;
    virtual AsdOps::Status Execute(Handle &handle, VariantPack &runInfo) = 0;
    std::string GetName() const;

private:
    std::string name_;
};
} // namespace AclTransformer
#endif