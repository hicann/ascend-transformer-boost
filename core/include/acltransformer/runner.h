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
#ifndef ACLTRANSFORMER_RUNNER_H
#define ACLTRANSFORMER_RUNNER_H
#include <string>
#include <asdops/utils/status/status.h>
#include "acltransformer/handle.h"
#include "acltransformer/runner_variant_pack.h"

namespace AclTransformer {
class Runner {
public:
    Runner(const std::string &name);
    virtual ~Runner();
    std::string GetName() const;
    AsdOps::Status Setup(const RunnerVariantPack &runnerVariantPack);
    uint64_t GetTilingBufferSize();
    void FillHostTilingBufferSize(void *hostTilingBuffer, uint64_t tilingBufferSize);
    uint64_t GetWorkspaceBufferSize();
    uint64_t GetIntermediateBufferSize();
    AsdOps::Status Execute(Handle &handle, RunnerVariantPack &runnerVariantPack);

private:
    virtual AsdOps::Status IsConsistent(const RunnerVariantPack &runnerVariantPack);
    virtual AsdOps::Status SetupImpl(const RunnerVariantPack &runnerVariantPack);
    virtual uint64_t GetTilingBufferSizeImpl();
    virtual void FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize);
    virtual uint64_t GetWorkspaceBufferSizeImpl();
    virtual uint64_t GetIntermediateBufferSizeImpl();
    virtual AsdOps::Status ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack);

private:
    std::string name_;
};
} // namespace AclTransformer
#endif