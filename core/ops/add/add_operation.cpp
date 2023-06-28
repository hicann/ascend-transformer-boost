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
#include "acltransformer/ops/add_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "add_ops_runner_builder.h"
#include "add_torch_runner_builder.h"

namespace AclTransformer {
AddOperation::AddOperation(const AddParam &param) : Operation("AddOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new AddOpsRunnerBuilder(param_), new AddTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new AddOpsRunnerBuilder(param_)};
#endif
}

AddOperation::~AddOperation() {}

uint64_t AddOperation::GetInTensorCount() const { return 2; }

uint64_t AddOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status AddOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *AddOperation::FindBestRunnerBuilder() const
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsAddOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer