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
#ifndef ACLTRANSFORMER_PLANV2_H
#define ACLTRANSFORMER_PLANV2_H
#include <memory>
#include <vector>
#include <asdops/utils/status/status.h>
#include "acltransformer/handle.h"
#include "acltransformer/variant_pack.h"
#include "acltransformer/runner_variant_pack.h"

namespace AclTransformer {
class Runner;

class PlanV2 {
public:
    PlanV2();
    ~PlanV2();
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

private:
    std::vector<char> hostTilingBuffer_;
    RunnerVariantPack runnerVariantPack_;
};
} // namespace AclTransformer
#endif