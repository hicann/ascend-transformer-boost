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
RopeOperation::RopeOperation(const PositionEmbeddingFusionParam &param)
    : Operation("RopeOperation"), param_(param)
{
    runnerBuilders_ = {new PositionEmbeddingFusionOpsRunnerBuilder(param_)};
}

RopeOperation::~RopeOperation() {}

uint64_t RopeOperation::GetInTensorCount() const { return inTensorSize; }

uint64_t RopeOperation::GetOutTensorCount() const { return outTensorSize; }

AsdOps::Status
RopeOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                              AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    if (param_.model == "gptneox20b") {
        // gptnexo20b in : QKV [bs, sq, 3 * all_hs], positionIds [bs, sql], cosTable, sinTable [sq, rd], seqLen [bs]
        // out: q, k, v [bs, hn, sq, hs]
        outTensorDescs.at(0) = inTensors.at(0).desc;
        outTensorDescs.at(0).dims.clear();
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(0));
        outTensorDescs.at(0).dims.push_back(param_.headNum);
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(1));
        outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims.at(2) / param_.headNum / 3);
        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(2) = outTensorDescs.at(0);
    }
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[0]);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[1]);
    outTensorDescs.at(0).dims.push_back(param_.headNum);
    outTensorDescs.at(0).dims.push_back(inTensors.at(0).desc.dims[2] / param_.headNum / kqvSliceSize);

    outTensorDescs.at(1) = outTensorDescs.at(0);
    outTensorDescs.at(2) = outTensorDescs.at(0); // 2=index
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *RopeOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer