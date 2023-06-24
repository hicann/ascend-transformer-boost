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
#include "acltransformer/kernel_cache.h"
namespace AclTransformer {
KernelCache::KernelCache() { cachedKernels_.resize(RUNNER_TYPE_MAX); }

KernelCache::~KernelCache() {}

void KernelCache::Init(RunnerType runnerType, uint64_t kernelCount)
{
    if (cachedKernels_.at(runnerType).empty()) {
        cachedKernels_.at(runnerType).resize(kernelCount);
    }
}

void KernelCache::Add(RunnerType runnerType, int64_t kernelIndex, const AsdOps::RunInfo &runInfo,
                      AsdOps::Kernel *kernel)
{
    if (runnerType >= 0 && runnerType < RUNNER_TYPE_MAX) {
        if (kernelIndex >= 0 && (uint64_t)kernelIndex < cachedKernels_.at(runnerType).size()) {
            cachedKernels_.at(runnerType).at(kernelIndex).first = runInfo;
            cachedKernels_.at(runnerType).at(kernelIndex).second = kernel;
        }
    }
}

AsdOps::Kernel *KernelCache::Get(RunnerType runnerType, int64_t kernelIndex, AsdOps::RunInfo &runInfo)
{
    if (runnerType >= 0 && runnerType < RUNNER_TYPE_MAX) {
        if (kernelIndex >= 0 && (uint64_t)kernelIndex < cachedKernels_.at(runnerType).size()) {
            if (IsRunInfoEqual(cachedKernels_.at(runnerType).at(kernelIndex).first, runInfo)) {
                runInfo = cachedKernels_.at(runnerType).at(kernelIndex).first;
                return cachedKernels_.at(runnerType).at(kernelIndex).second;
            }
        }
    }
    return nullptr;
}

bool KernelCache::IsRunInfoEqual(const AsdOps::RunInfo &runInfo1, const AsdOps::RunInfo &runInfo2)
{
    if (runInfo1.GetInTensorCount() != runInfo2.GetInTensorCount()) {
        return false;
    }

    for (uint64_t i = 0; i < runInfo1.GetInTensorCount(); ++i) {
        if (!IsTensorDescEqual(runInfo1.GetInTensor(i).desc, runInfo2.GetInTensor(i).desc)) {
            return false;
        }
    }

    return true;
}

bool KernelCache::IsTensorDescEqual(const AsdOps::TensorDesc &tensorDesc1, const AsdOps::TensorDesc &tensorDesc2)
{
    return tensorDesc1.dtype == tensorDesc2.dtype && tensorDesc1.format == tensorDesc2.format &&
           tensorDesc1.dims == tensorDesc2.dims;
}
} // namespace AclTransformer
