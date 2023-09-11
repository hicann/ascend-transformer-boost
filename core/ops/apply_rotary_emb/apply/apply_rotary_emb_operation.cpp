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
#include "acltransformer/ops/apply_rotary_emb_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/config.h"
#include "apply_rotary_emb_builder.h"

namespace AclTransformer {
ApplyRotaryEmbOperation::ApplyRotaryEmbOperation(const ApplayRotaryEmbParam &param) : Operation("ApplyRotaryEmbOpration")
{
    runnerBuilders_ = {new ApplyRotaryEmbRunnerBuilder()};
}

ApplyRotaryEmbOperation::~ApplyRotaryEmbOperation() {}

uint64_t ApplyRotaryEmbOperation::GetInTensorCount() const { return 3; }

uint64_t ApplyRotaryEmbOperation::GetOutTensorCount() const { return 2; }

AsdOps::Status ApplyRotaryEmbOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    ASD_LOG(INFO) << "ApplyRotaryEmbOperation::InferShapeImpl start";
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(1) = inTensors.at(1).desc;
    ASD_LOG(INFO) << "ApplyRotaryEmbOperation::InferShapeImpl end";
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *ApplyRotaryEmbOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer
