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
#include "acltransformer/ops/position_embedding_fusion_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "position_embedding_fusion_ops_runner_builder.h"

namespace AclTransformer {
PositionEmbeddingFusionOperation::PositionEmbeddingFusionOperation(const PositionEmbeddingFusionParam &param)
    : Operation("PositionEmbeddingFusionOperation"), param_(param)
{
    runnerBuilders_ = {new PositionEmbeddingFusionOpsRunnerBuilder(param_)};
}

PositionEmbeddingFusionOperation::~PositionEmbeddingFusionOperation() {}

uint64_t PositionEmbeddingFusionOperation::GetInTensorCount() const { return inTensorSize; }

uint64_t PositionEmbeddingFusionOperation::GetOutTensorCount() const { return outTensorSize; }

AsdOps::Status
PositionEmbeddingFusionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                 AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    // in : Q,[seq_len, batch, all_head_size]   position_ids,[]  cos_table,[]  sin_table[]
    // out : Q ,[seq_len * batch, head_num * head_size]
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims.at(0) * inTensors.at(0).desc.dims.at(1),
                                 inTensors.at(0).desc.dims.at(2) / kqvSliceSize}; // 2=index
    outTensorDescs.at(1) = outTensorDescs.at(0);
    outTensorDescs.at(2) = outTensorDescs.at(0); // 2=index

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *PositionEmbeddingFusionOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer