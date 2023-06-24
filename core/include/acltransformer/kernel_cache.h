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
#ifndef ACLTRANSFORMER_KERNEL_CACHE_H
#define ACLTRANSFORMER_KERNEL_CACHE_H
#include <vector>
#include <asdops/run_info.h>
#include <asdops/kernel.h>
#include "acltransformer/runner_type.h"

namespace AclTransformer {
class KernelCache {
public:
    KernelCache();
    ~KernelCache();
    void Init(RunnerType runnerType, uint64_t kernelCount);
    void Add(RunnerType runnerType, int64_t kernelIndex, const AsdOps::RunInfo &runInfo, AsdOps::Kernel *kernel);
    AsdOps::Kernel *Get(RunnerType runnerType, int64_t kernelIndex, AsdOps::RunInfo &runInfo);

private:
    bool IsRunInfoEqual(const AsdOps::RunInfo &runInfo1, const AsdOps::RunInfo &runInfo2);
    bool IsTensorDescEqual(const AsdOps::TensorDesc &tensorDesc1, const AsdOps::TensorDesc &tensorDesc2);

private:
    std::vector<std::vector<std::pair<AsdOps::RunInfo, AsdOps::Kernel *>>> cachedKernels_;
};
} // namespace AclTransformer
#endif