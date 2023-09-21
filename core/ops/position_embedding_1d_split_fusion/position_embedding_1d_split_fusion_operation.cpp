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
#include "acltransformer/ops/position_embedding_1d_fusion_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "position_embedding_1d_split_fusion_ops_runner_builder.h"

namespace AclTransformer {

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 2;

static const uint64_t LLAMA_IN_TENSOR_COUNT = 4;
static const uint64_t LLAMA_OUT_TENSOR_COUNT = 3;

PositionEmbedding1dSplitFusionOperation::PositionEmbedding1dSplitFusionOperation(const PositionEmbedding1dFusionParam &param)
    : Operation("PositionEmbedding1dSplitFusionOperation"), param_(param)
{
    runnerBuilders_ = {new PositionEmbedding1dFusionOpsRunnerBuilder(param_)};
}

PositionEmbedding1dSplitFusionOperation::~PositionEmbedding1dSplitFusionOperation() {}

uint64_t PositionEmbedding1dSplitFusionOperation::GetInTensorCount() const
{
    return (param_.model == "llama13b") ? LLAMA_IN_TENSOR_COUNT : IN_TENSOR_COUNT;
}

uint64_t PositionEmbedding1dSplitFusionOperation::GetOutTensorCount() const
{
    return (param_.model == "llama13b") ? LLAMA_OUT_TENSOR_COUNT : OUT_TENSOR_COUNT;
}

AsdOps::Status
PositionEmbedding1dSplitFusionOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    if (param_.model == "llama13b") {
        outTensorDescs.resize(GetOutTensorCount());
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(0).dims.clear();
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[0]);
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[1] / 3);
        outTensorDescs.at(1) = inTensors.at(0).desc;
        outTensorDescs.at(1).dims.clear();
        outTensorDescs.at(1).dims.push_back(inTensors.at(0).desc.dims[0]);
        outTensorDescs.at(1).dims.push_back(inTensors.at(0).desc.dims[1] / 3);
        outTensorDescs.at(2) = inTensors.at(0).desc;
        outTensorDescs.at(2).dims.clear();
        outTensorDescs.at(2).dims.push_back(inTensors.at(0).desc.dims[0]);
        outTensorDescs.at(2).dims.push_back(inTensors.at(0).desc.dims[1] / 3);
    } else {
        outTensorDescs.resize(GetOutTensorCount());
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(0).dims.clear();
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[0]);
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[1]);
        outTensorDescs.at(1) = inTensors.at(1).desc;
        outTensorDescs.at(1).dims.clear();
        outTensorDescs.at(1).dims.push_back(inTensors.at(1).desc.dims[0]);
        outTensorDescs.at(1).dims.push_back(inTensors.at(1).desc.dims[1]);
    }

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *PositionEmbedding1dSplitFusionOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer