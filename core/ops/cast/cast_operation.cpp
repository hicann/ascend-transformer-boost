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
#include "acltransformer/ops/cast_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "cast_ops_runner_builder.h"

namespace AclTransformer {
CastOperation::CastOperation() : Operation("CastOperation")
{
    runnerBuilders_ = {new CastOpsRunnerBuilder()};
}

CastOperation::~CastOperation() {}

uint64_t CastOperation::GetInTensorCount() const { return 1; }

uint64_t CastOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status CastOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensors.at(0).desc;
    if (inTensors.at(0).desc.dtype == AsdOps::TensorDType::TENSOR_DTYPE_FLOAT16) {
        outTensorDescs.at(0).dtype = AsdOps::TensorDType::TENSOR_DTYPE_FLOAT;
    } else {
        outTensorDescs.at(0).dtype = AsdOps::TensorDType::TENSOR_DTYPE_FLOAT16;
    }
    return AsdOps::Status::OkStatus();
}

RunnerBuilder *CastOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer
