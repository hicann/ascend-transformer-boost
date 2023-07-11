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
#include <functional>
#include <asdops/params/params.h>
#include "asdops/utils/log/log.h"
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
using ParamCompareFunc = std::function<bool(const AsdOps::Any &, const AsdOps::Any &)>;

template <typename T> bool ParamCompareFuncImpl(const AsdOps::Any &any1, const AsdOps::Any &any2)
{
    const auto &content1 = AsdOps::AnyCast<T>(any1);
    const auto &content2 = AsdOps::AnyCast<T>(any2);
    return content1 == content2;
}

static std::map<std::size_t, ParamCompareFunc> ParamCompareMap_ = {
    {typeid(AsdOps::OpParam::AsStrided).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::AsStrided>},
    {typeid(AsdOps::OpParam::Attention).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Attention>},
    {typeid(AsdOps::OpParam::Broadcast).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Broadcast>},
    {typeid(AsdOps::OpParam::Concat).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Concat>},
    {typeid(AsdOps::OpParam::Elewise).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Elewise>},
    {typeid(AsdOps::OpParam::Gather).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Gather>},
    {typeid(AsdOps::OpParam::GlobalInfo).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::GlobalInfo>},
    {typeid(AsdOps::OpParam::MatMul).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::MatMul>},
    {typeid(AsdOps::OpParam::Norm).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Norm>},
    {typeid(AsdOps::OpParam::Split).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Split>},
    {typeid(AsdOps::OpParam::Transdata).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Transdata>},
    {typeid(AsdOps::OpParam::Transpose).hash_code(), ParamCompareFuncImpl<AsdOps::OpParam::Transpose>}};

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
        if (!TensorUtil::AsdOpsTensorDescEqual(runInfo1.GetInTensor(i).desc, runInfo2.GetInTensor(i).desc)) {
            return false;
        }
    }

    const AsdOps::OpDesc &opDesc1 = runInfo1.GetOpDesc();
    const AsdOps::OpDesc &opDesc2 = runInfo2.GetOpDesc();
    if (opDesc1.opName != opDesc2.opName) {
        return false;
    }
    auto it = ParamCompareMap_.find(opDesc1.specificParam.Type().hash_code());
    if (it != ParamCompareMap_.end()) {
        return it->second(opDesc1.specificParam, opDesc2.specificParam);
    } else {
        ASD_LOG(WARN) << "Can not compare param of " << opDesc1.opName;
    }

    return true;
}
} // namespace AclTransformer
