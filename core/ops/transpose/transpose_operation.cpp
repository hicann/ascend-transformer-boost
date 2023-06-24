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
#include "acltransformer/ops/transpose_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "transpose_ops_runner_builder.h"
#include "transpose_torch_runner_builder.h"

namespace AclTransformer {
TransposeOperation::TransposeOperation(const TransposeParam &param) : Operation("TransposeOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new TransposeOpsRunnerBuilder(param_), new TransposeTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new TransposeOpsRunnerBuilder(param_)};
#endif
}

TransposeOperation::~TransposeOperation() {}

AsdOps::Status TransposeOperation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensors.size() != 1) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 1");
    }

    outTensorDescs.resize(1);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    std::swap(outTensorDescs.at(0).dims[param_.dimA], outTensorDescs.at(0).dims[param_.dimB]);
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *TransposeOperation::FindBestRunnerBuilder()
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsTransposeOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer