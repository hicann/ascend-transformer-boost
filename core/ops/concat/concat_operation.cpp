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
#include "acltransformer/ops/concat_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include <asdops/types.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/config.h"
#include "concat_ops_runner_builder.h"

namespace AclTransformer {
ConcatOperation::ConcatOperation(const ConcatParam &param) : Operation("ConcatOperation"), param_(param)
{
    runnerBuilders_ = {new ConcatOpsRunnerBuilder(param_)};
}

ConcatOperation::~ConcatOperation() {}

uint64_t ConcatOperation::GetInTensorCount() const { return 2; }

uint64_t ConcatOperation::GetOutTensorCount() const { return 1; }

static const char *OP_NAME = "ConcatOperation";

AsdOps::Status ConcatOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    if (inTensors.size() != static_cast<size_t>(GetInTensorCount())) {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INVALID_VALUE, "input tensor number should be 2");
    }
    if (outTensorDescs.size() != static_cast<size_t>(GetOutTensorCount())) {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INVALID_VALUE, "output tensor number should be 1");
    }

    AsdOps::TensorDType dtype0 = inTensors.at(0).desc.dtype;
    AsdOps::TensorDType dtype1 = inTensors.at(1).desc.dtype;
    AsdOps::SVector<int64_t> dims = inTensors.at(0).desc.dims;
    AsdOps::SVector<int64_t> dims1 = inTensors.at(1).desc.dims;
    AsdOps::TensorFormat format = inTensors.at(0).desc.format;
    if (dtype0 == AsdOps::TENSOR_DTYPE_FLOAT16 && dtype1 == AsdOps::TENSOR_DTYPE_FLOAT16) {
        dims.at(param_.concatDim) = dims.at(param_.concatDim) + dims1.at(param_.concatDim);
        outTensorDescs.at(0) = {AsdOps::TENSOR_DTYPE_FLOAT16, format, dims};
        return AsdOps::Status::OkStatus();
    } else {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INFERSHAPE_ERROR, "Unsupported input descriptor.");
    }
}

RunnerBuilder *ConcatOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer
