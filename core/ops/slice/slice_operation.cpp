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
#include "acltransformer/ops/slice_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include <asdops/types.h>
#include "acltransformer/config.h"
#include "slice_ops_runner_builder.h"

namespace AclTransformer {
SliceOperation::SliceOperation(const SliceParam &param) : Operation("SliceOperation"), param_(param)
{
    runnerBuilders_ = {new SliceOpsRunnerBuilder(param_)};
}

SliceOperation::~SliceOperation() {}

uint64_t SliceOperation::GetInTensorCount() const { return 1; }

uint64_t SliceOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status SliceOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &xTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    auto xTensorDims = xTensors.at(0).desc.dims.size();
    outTensorDescs.at(0) = xTensors.at(0).desc;
    if (param_.offsets.size() != xTensorDims) {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INVALID_VALUE, "Wrong input offsets");
    }
    if (param_.size.size() != xTensorDims) {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INVALID_VALUE, "Wrong input size");
    }

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SliceOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer
