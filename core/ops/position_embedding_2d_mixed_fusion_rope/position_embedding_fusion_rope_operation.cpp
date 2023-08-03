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
//#include "acltransformer/ops/position_embedding_fusion_gather_operation.h"
#include "acltransformer/ops/position_embedding_fusion_rope_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "position_embedding_fusion_rope_ops_runner_builder.h"

namespace AclTransformer {
OptRopeOperation::OptRopeOperation(const PositionEmbeddingFusionParam &param)
    : Operation("OptRopeOperation"), param_(param)
{
    runnerBuilders_ = {new PositionEmbeddingFusionRopeOpsRunnerBuilder(param_)};
}

OptRopeOperation::~OptRopeOperation() {}

uint64_t OptRopeOperation::GetInTensorCount() const { return IN_TENSOR_SIZE; }

uint64_t OptRopeOperation::GetOutTensorCount() const { return OUT_TENSOR_SIZE; }

AsdOps::Status
OptRopeOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[0]);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[1]);
    outTensorDescs.at(0).dims.push_back(param_.headNum);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[2] / param_.headNum / KQV_SLICE_SIZE);

    outTensorDescs.at(1) = outTensorDescs.at(0);
    outTensorDescs.at(2) = outTensorDescs.at(0); // 2=index
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *OptRopeOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer