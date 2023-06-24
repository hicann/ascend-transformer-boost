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
#include "acltransformer/ops/rms_norm_operation.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "rms_norm_ops_runner_builder.h"
#include "rms_norm_torch_runner_builder.h"

namespace AclTransformer {
RmsNormOperation::RmsNormOperation(const RmsNormParam &param) : Operation("RmsNormOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new RmsNormOpsRunnerBuilder(param_), new RmsNormTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new RmsNormOpsRunnerBuilder(param_)};
#endif
}

RmsNormOperation::~RmsNormOperation() {}

AsdOps::Status RmsNormOperation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensors.size() != 2) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 2");
    }

    outTensorDescs.resize(1);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    ASD_LOG(INFO) << "outTensor dtype:" << outTensorDescs.at(0).dtype;
    ASD_LOG(INFO) << "outTensor format:" << outTensorDescs.at(0).format;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *RmsNormOperation::FindBestRunnerBuilder()
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsRmsNormOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer