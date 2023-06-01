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
#ifndef ACLTRANSFORMER_OPERATION_H
#define ACLTRANSFORMER_OPERATION_H
#include <vector>
#include <string>
#include "acltransformer/handle.h"
#include "acltransformer/variant_pack.h"
#include "asdops/utils/status/status.h"
#include "asdops/tensor_desc.h"

namespace AclTransformer {
class Runner;
class RunnerBuilder;

class Operation {
public:
    Operation(const std::string &name);
    virtual ~Operation();
    std::string GetName() const;
    virtual AsdOps::Status InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                      std::vector<AsdOps::TensorDesc> &outTensorDescs) = 0;
    AsdOps::Status Setup(VariantPack &variantPack);
    uint64_t GetWorkspaceSize();
    AsdOps::Status Execute(Handle &handle, VariantPack &variantPack);

protected:
    Runner *CreateBestRunner(const VariantPack &variantPack);
    virtual RunnerBuilder *FindBestRunnerBuilder(const VariantPack &variantPack) = 0;
    friend class PlanBuilder;

protected:
    Runner *runner_ = nullptr;
    std::vector<RunnerBuilder *> runnerBuilders_;
    std::string name_;
};
} // namespace AclTransformer
#endif