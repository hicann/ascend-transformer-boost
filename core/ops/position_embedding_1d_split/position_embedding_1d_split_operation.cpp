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
#include "acltransformer/ops/position_embedding_1d_split_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "position_embedding_1d_split_ops_runner_builder.h"
#include "position_embedding_1d_split_torch_runner_builder.h"

namespace AclTransformer {
PositionEmbedding1dSplitOperation::PositionEmbedding1dSplitOperation(const PositionEmbedding1dSplitParam &param)
    : Operation("PositionEmbedding1dSplitOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new PositionEmbedding1dSplitOpsRunnerBuilder(param_),
                       new PositionEmbedding1dSplitTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new PositionEmbedding1dSplitTorchRunnerBuilder(param_)};
#endif
}

PositionEmbedding1dSplitOperation::~PositionEmbedding1dSplitOperation() {}

AsdOps::Status PositionEmbedding1dSplitOperation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                             AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    // in : Q,[batch, seq_len, all_head_size]   position_ids,[]  cos_table,[]  sin_table[]
    // out : Q ,[seq_len, batch, head_num, head_size]
    if (inTensors.size() != 4) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 4 ");
    }

    outTensorDescs.resize(1);

    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims.clear();
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[1]);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[0]);
    outTensorDescs.at(0).dims.push_back(param_.headNum);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[2] / param_.headNum);

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *PositionEmbedding1dSplitOperation::FindBestRunnerBuilder()
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsPositionEmbedding1dSplitOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer