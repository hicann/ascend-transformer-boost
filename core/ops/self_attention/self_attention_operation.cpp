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
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "self_attention_torch_runner_builder.h"
#include "self_attention_ops_runner_builder.h"
#include <asdops/utils/log/log.h>

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

uint64_t SelfAttentionOperation::GetInTensorCount() const { return 4; }

uint64_t SelfAttentionOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status SelfAttentionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                      AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    if (param_.model == "openbert") {
        outTensorDescs.at(0) = inTensors.at(0).desc;
    } else if (param_.model == "chatglm6b") {
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(0).dims.clear();
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1));
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(2) * inTensors.at(0).desc.dims.at(3));
    } else if (param_.model == "gptneox20b") {
        // input [bs, hn, sq, hs]
        // output [bs, sq, hn * hs]
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(0).dims.clear();
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(2));
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1) * inTensors.at(0).desc.dims.at(3));
        outTensorDescs.at(0).format = AsdOps::TENSOR_FORMAT_ND;
    }
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SelfAttentionOperation::FindBestRunnerBuilder() const
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsSelfAttentionOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer