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
#include "acltransformer/ops/self_attention_operation.h"
#include "acltransformer/config.h"
#include "self_attention_torch_runner_builder.h"
#include "self_attention_ops_runner_builder.h"

namespace AclTransformer {
SelfAttentionOperation::SelfAttentionOperation(const SelfAttentionParam &param)
    : Operation("SelfAttentionOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new SelfAttentionOpsRunnerBuilder(param_), new SelfAttentionTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new SelfAttentionOpsRunnerBuilder(param_)};
#endif
}

SelfAttentionOperation::~SelfAttentionOperation() {}

AsdOps::Status SelfAttentionOperation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensors.size() != 4) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 2");
    }

    outTensorDescs.resize(1);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SelfAttentionOperation::FindBestRunnerBuilder(const VariantPack &variantPack)
{
#ifdef USE_TORCH_RUNNER
    size_t index = Config::IsSelfAttentionOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer