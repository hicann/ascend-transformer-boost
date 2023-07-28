

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
#include "post_torch_runner.h"
#ifdef USE_TORCH_RUNNER
#include <ATen/ATen.h>
#include "acltransformer/torch/torch_util.h"
#endif
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include "acltransformer/utils/tensor_util.h"

namespace AclTransformer {
PostTorchRunner::PostTorchRunner(const PostParam &param) : Runner("PostTorchRunner"), param_(param) {}

PostTorchRunner::~PostTorchRunner() {}

AsdOps::Status PostTorchRunner::ExecuteImpl(Handle &handle, RunnerVariantPack &runnerVariantPack)
{
    if (runnerVariantPack.inTensors.size() != 1) {
        return AsdOps::Status::FailStatus(1, "PostTorchRunner inTensor num error!");
    }
#ifdef USE_TORCH_RUNNER
    ASD_LOG(INFO) << "PostTorchRunner start";
    at::Tensor atInTensor = TorchUtil::AsdOpsTensor2AtTensor(handle, runnerVariantPack.inTensors[0]);
    ASD_LOG(INFO) << "receive inputs";

    at::Tensor scores = atInTensor / param_.temperature;
    int64_t top_k = std::min(param_.top_k, scores.sizes()[1]);

    auto result = torch::sort(scores, -1, true);
    at::Tensor valuescur = std::get<0>(result).slice(-1, 0, top_k);
    at::Tensor indices = std::get<1>(result).slice(-1, 0, top_k);

    at::Tensor values = torch::flip(valuescur, {1});
    at::Tensor curprobs = torch::softmax(values, -1);
    at::Tensor cumulative_probs = torch::cumsum(curprobs, -1);
    at::Tensor sorted_indices_to_remove = cumulative_probs <= (1 - param_.top_p);

    if(param_.min_tokens_to_keep > 1){
        sorted_indices_to_remove.slice(-1 , -param_.min_tokens_to_keep, -1) = 0;
    }

    at::Tensor output = torch::masked_fill(values, sorted_indices_to_remove, param_.filter_value);

    int n_dim = output.dim();
    at::Tensor probs = torch::softmax(output, n_dim - 1);

    int n_samples = 1;
    bool replacement = true;
    at::Tensor next_tokens = torch::multinomial(probs, n_samples, replacement);
    at::Tensor atOutTensor = next_tokens;

    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, atOutTensor, runnerVariantPack.outTensors[0]);
    TorchUtil::CopyAtTensor2AsdOpsTensor(handle.stream, indices, runnerVariantPack.outTensors[1]);
    return AsdOps::Status::OkStatus();
#else
    return AsdOps::Status::FailStatus(1, "USE_TORCH_RUNNER not define");
#endif
}
} // namespace AclTransformer