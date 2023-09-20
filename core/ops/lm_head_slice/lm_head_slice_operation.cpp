/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, s
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include <asdops/utils/log/log.h>
#include "acltransformer/ops/lm_head_slice_operation.h"
#include "lm_head_slice_ops_runner_builder.h"

namespace AclTransformer {
LmHeadSliceOperation::LmHeadSliceOperation(const LmHeadSliceParam &param) : Operation("LmHeadSliceOperation"), param_(param)
{
    runnerBuilders_ = {new LmHeadSliceOpsRunnerBuilder(param_)};
}

LmHeadSliceOperation::~LmHeadSliceOperation() {}

uint64_t LmHeadSliceOperation::GetInTensorCount() const { return 1; }

uint64_t LmHeadSliceOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status LmHeadSliceOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                            AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[1], 1, inTensors.at(0).desc.dims[2]};
    ASD_LOG(INFO) << "OutTensor dims = " << outTensorDescs.at(0).dims;
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *LmHeadSliceOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;

    return runnerBuilders_.at(index);
}
} // namespace AclTransformer