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
#ifndef ACLTRANSFORMER_LCCLRUNNER_H
#define ACLTRANSFORMER_LCCLRUNNER_H

#include "acltransformer/runner.h"
#include "acltransformer/runner_type.h"
#ifdef USE_LCCL_RUNNER
#include <lccl.h>
#endif
#include <asdops/types.h>


namespace AclTransformer {
class LcclRunner : public Runner{
public:
    LcclRunner(const std::string &name, RunnerType runnerType = RUNNER_TYPE_UNDEFINED, 
        int rank = 0, int rankSize = 0);
    virtual ~LcclRunner();

protected:
    AsdOps::Status SetupImpl(const RunnerVariantPack &runnerVariantPack) override;
    uint64_t GetTilingBufferSizeImpl() override;
    void FillHostTilingBufferSizeImpl(void *hostTilingBuffer, uint64_t tilingBufferSize) override;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    uint64_t GetIntermediateBufferSizeImpl() override;
    AsdOps::Status ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack) override;

private:
    uint64_t GetLcclDtype(AsdOps::TensorDType dtype);

protected:
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
    int rank_ = 0;
    int rankSize = 0;
};
} // namespace AclTransformer
#endif