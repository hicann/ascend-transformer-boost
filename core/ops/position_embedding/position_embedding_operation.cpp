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
#include "acltransformer/ops/position_embedding_operation.h"
#include "position_embedding_torch_runner_builder.h"

namespace AclTransformer {
PositionEmbeddingOperation::PositionEmbeddingOperation(const PositionEmbeddingParam  &param)
    : Operation("PositionEmbeddingOperation"), param_(param)
{
    runnerBuilders_ = {new PositionEmbeddingTorchRunnerBuilder(param_)};
}

PositionEmbeddingOperation::~PositionEmbeddingOperation() {}

AsdOps::Status PositionEmbeddingOperation::InferShape(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs)
{
    // in : Q,[seq_len, batch, all_head_size]   position_ids,[]  cos_table,[]  sin_table[]
    // out : Q ,[seq_len, batch, head_num, head_size]
    if (inTensors.size() != 4) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 4 ");
    }

    outTensorDescs.resize(2);

    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims.clear();
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1));
    outTensorDescs.at(0).dims.push_back(param_.headNum);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(2) / param_.headNum);
    outTensorDescs.at(1) = outTensorDescs.at(0);

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *PositionEmbeddingOperation::FindBestRunnerBuilder(const VariantPack &variantPack)
{
    return runnerBuilders_.at(0);
}
} // namespace AclTransformer