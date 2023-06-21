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
#include "acltransformer/ops/self_attention_kv_cache_operation.h"
#include "acltransformer/config.h"
#include "self_attention_kv_cache_ops_runner_builder.h"
#include "self_attention_kv_cache_torch_runner_builder.h"

namespace AclTransformer {
SelfAttentionKvCacheOperation::SelfAttentionKvCacheOperation(const SelfAttentionKvCacheParam &param)
    : Operation("SelfAttentionKvCacheOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new SelfAttentionKvCacheOpsRunnerBuilder(param_),
                       new SelfAttentionKvCacheTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new SelfAttentionKvCacheOpsRunnerBuilder(param_)};
#endif
}

SelfAttentionKvCacheOperation::~SelfAttentionKvCacheOperation() {}

AsdOps::Status SelfAttentionKvCacheOperation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                         AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    // in : Q K V attention_mast pastK pastV [seq_len, batch, head_num, head_size]
    // out : out presentK presentV [seq_len, batch, head_num * head_size]
    if (inTensors.size() != 6) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 6");
    }

    outTensorDescs.resize(3);
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims.clear();
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1));
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(2) * inTensors.at(0).desc.dims.at(3));
    outTensorDescs.at(1) = inTensors.at(4).desc;
    outTensorDescs.at(1).dims.at(0) = outTensorDescs.at(1).dims.at(0) + 1;
    outTensorDescs.at(2) = inTensors.at(5).desc;
    outTensorDescs.at(2).dims.at(0) = outTensorDescs.at(2).dims.at(0) + 1;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SelfAttentionKvCacheOperation::FindBestRunnerBuilder(const VariantPack &variantPack)
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsSelfAttentionKVCacheOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer