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
#include "acltransformer/ops/norm_operation.h"
#include "acltransformer/config.h"
#include "norm_torch_runner_builder.h"
#include "norm_ops_runner_builder.h"

namespace AclTransformer {
NormOperation::NormOperation(const NormParam &param) : Operation("NormOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new NormOpsRunnerBuilder(param_), new NormTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new NormOpsRunnerBuilder(param_)};
#endif
}

NormOperation::~NormOperation() {}

AsdOps::Status NormOperation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                         AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensors.size() != 3) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 3");
    }

    outTensorDescs.resize(1);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *NormOperation::FindBestRunnerBuilder(const VariantPack &variantPack)
{
#ifdef USE_TORCH_RUNNER
    size_t index = Config::IsNormOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer