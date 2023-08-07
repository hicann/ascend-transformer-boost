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
#include "acltransformer/ops/position_embedding_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "position_embedding_ops_runner_builder.h"
#include "position_embedding_torch_runner_builder.h"

static constexpr int64_t GLM2_IN_TENSOR_SIZE = 2;
static constexpr int64_t DEFAULT_IN_TENSOR_SIZE = 4;
namespace AclTransformer {
PositionEmbeddingOperation::PositionEmbeddingOperation(const PositionEmbeddingParam &param)
    : Operation("PositionEmbeddingOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new PositionEmbeddingOpsRunnerBuilder(param_), new PositionEmbeddingTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new PositionEmbeddingOpsRunnerBuilder(param_)};
#endif
}

PositionEmbeddingOperation::~PositionEmbeddingOperation() {}

uint64_t PositionEmbeddingOperation::GetInTensorCount() const 
{ 
    return (param_.model == "chatglm2_6b") ? GLM2_IN_TENSOR_SIZE : DEFAULT_IN_TENSOR_SIZE; 
}

uint64_t PositionEmbeddingOperation::GetOutTensorCount() const { return 3; }

AsdOps::Status PositionEmbeddingOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                          AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    if (param_.model == "chatglm2_6b"){
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(0).dims.clear();
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1));
        outTensorDescs.at(0).dims.push_back(param_.numHeadPerPartition);
        outTensorDescs.at(0).dims.push_back(param_.hiddenSizePerHead);

        outTensorDescs.at(1) = inTensors.at(0).desc;
        outTensorDescs.at(1).dims.clear();
        outTensorDescs.at(1).dims.push_back(inTensors.at(0).desc.dims.at(0));
        outTensorDescs.at(1).dims.push_back(inTensors.at(0).desc.dims.at(1));
        outTensorDescs.at(1).dims.push_back(param_.numGroupsPerPartition);
        outTensorDescs.at(1).dims.push_back(param_.hiddenSizePerHead);
        outTensorDescs.at(2) = outTensorDescs.at(1);
    } else{
        // in : Q,[seq_len, batch, all_head_size]   position_ids,[]  cos_table,[]  sin_table[]
        // out : Q ,[seq_len, batch, head_num, head_size]
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(0).dims.clear();
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1));
        outTensorDescs.at(0).dims.push_back(param_.headNum);
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(2) / param_.headNum / 3); // 3=qkv
        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(2) = outTensorDescs.at(0);
    }

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *PositionEmbeddingOperation::FindBestRunnerBuilder() const
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsPositionEmbeddingOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer