/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 *  * Licensed under the Apache License, Version 2.0 (the "License");
 *  * you may not use this file except in compliance with the License.
 *  * You may obtain a copy of the License at
 *  *
 *  * http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  * See the License for the specific language governing permissions and
 *  * limitations under the License.
 *  */
#include "acltransformer/ops/sort_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "sort_ops_runner_builder.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
SortOperation::SortOperation(const SortParam &param) : Operation("SortOperation"), param_(param)
{
    runnerBuilders_ = {new SortOpsRunnerBuilder(param)};
}

SortOperation::~SortOperation() {}

uint64_t SortOperation::GetInTensorCount() const { return 1; }

uint64_t SortOperation::GetOutTensorCount() const { return 2; }

AsdOps::Status SortOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                             AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    if (param_.num.size() == 0) {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INVALID_VALUE, "TopK num invalid.");
    }
    const AsdOps::SVector<int64_t> &dims0 = inTensors.at(0).desc.dims;
    if (dims0[dims0.size() - 1] < param_.num[0]) {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INVALID_VALUE,
                                          "TopK input tensor last dim at least k.");
    }
    AsdOps::SVector<int64_t> outDim = dims0;
    outDim[dims0.size() - 1] = param_.num[0];

    AsdOps::TensorDType dtype0 = inTensors.at(0).desc.dtype;
    AsdOps::TensorFormat format = inTensors.at(0).desc.format;
    outTensorDescs.at(0) = {dtype0, format, outDim};
    outTensorDescs.at(1) = {AsdOps::TensorDType::TENSOR_DTYPE_INT32, AsdOps::TensorFormat::TENSOR_FORMAT_ND, outDim};

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *SortOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer