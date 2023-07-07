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
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "self_attention_kv_cache_fusion_ops_runner_builder.h"
#include "acltransformer/ops/self_attention_kv_cache_fusion_operation.h"

namespace AclTransformer {
SelfAttentionKvCacheFusionOperation::SelfAttentionKvCacheFusionOperation(const SelfAttentionKvCacheFusionParam &param)
    : Operation("SelfAttentionKvCacheFusionOperation"), param_(param)
{
    runnerBuilders_ = {new SelfAttentionKvCacheFusionOpsRunnerBuilder(param_)};
}

SelfAttentionKvCacheFusionOperation::~SelfAttentionKvCacheFusionOperation() {}

uint64_t SelfAttentionKvCacheFusionOperation::GetInTensorCount() const { return 9; }

uint64_t SelfAttentionKvCacheFusionOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status SelfAttentionKvCacheFusionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
    AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    // in : Q K V attention_mast pastK pastV [seq_len, batch, head_num, head_size]
    // out : out from flas attention [seq_len , batch, head_num, head_size]
    outTensorDescs.resize(GetOutTensorCount());
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims.clear();
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
    outTensorDescs.at(0).dims.push_back(1); // batch == 1
    outTensorDescs.at(0).dims.push_back(param_.headNum);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1) / param_.headNum);
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SelfAttentionKvCacheFusionOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer