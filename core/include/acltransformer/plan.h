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
#include <vector>
#include <asdops/utils/status/status.h>
#include "acltransformer/handle.h"
#include "acltransformer/variant_pack.h"
#include "acltransformer/runner_variant_pack.h"

namespace AclTransformer {
class Runner;

#ifdef USE_PROFILING
#define MAX_PROFILING_FUNC_NAME 2
enum ProfilingFuncName {
    PLAN_SETUP = 0,
    PLAN_EXECUTE
};
#endif

class Plan {
public:
    Plan();
    ~Plan();
    AsdOps::Status Setup(Handle handle, const VariantPack &variantPack);
    uint64_t GetWorkspaceSize();
    AsdOps::Status Execute(Handle handle, VariantPack &variantPack);

protected:
    std::unique_ptr<Runner> runner_;
    std::string name_;
    friend class Operation;

private:
    void Reset();
    AsdOps::Status CopyHostTilingToDevice(Handle handle);
#ifdef USE_PROFILING
    void ReportApiInfo(const uint64_t beginTime, ProfilingFuncName type);
#endif

private:
    std::vector<char> hostTilingBuffer_;
    RunnerVariantPack runnerVariantPack_;

#ifdef USE_PROFILING
    std::vector<uint64_t> hashIdArray_;
#endif
};
} // namespace AclTransformer
#endif