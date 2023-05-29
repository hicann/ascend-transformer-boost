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
#include "acltransformer/ops/self_attention_operation.h"
#include "self_attention_torch_runner_builder.h"
#include "self_attention_ops_runner_builder.h"

namespace AclTransformer {
SelfAttentionOperation::SelfAttentionOperation(const SelfAttentionParam &param)
    : Operation("SelfAttentionOperation"), param_(param)
{
    runnerBuilders_ = {new SelfAttentionOpsRunnerBuilder(param_), new SelfAttentionTorchRunnerBuilder(param_)};
}

SelfAttentionOperation::~SelfAttentionOperation() {}

AsdOps::Status SelfAttentionOperation::InferShape(const std::vector<AsdOps::TensorDesc> &inTensorDescs,
                                                  std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensorDescs.size() != 4) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 2");
    }

    outTensorDescs.resize(1);
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SelfAttentionOperation::FindBestRunnerBuilder(const VariantPack &variantPack)
{
    return runnerBuilders_.at(1);
}
} // namespace AclTransformer