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
#include "acltransformer/ops/gather_operation.h"
#include <asdops/utils/singleton/singleton.h>
#include <asdops/types.h>
#include <asdops/utils/log/log.h>
#include "acltransformer/config.h"
#include "gather_ops_runner_builder.h"

namespace AclTransformer {
GatherOperation::GatherOperation(const GatherParam &param) : Operation("GatherOperation"), param_(param)
{
    runnerBuilders_ = {new GatherOpsRunnerBuilder(param_)};
}

GatherOperation::~GatherOperation() {}

uint64_t GatherOperation::GetInTensorCount() const { return 2; }

uint64_t GatherOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status GatherOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                  AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    if (inTensors.size() != static_cast<size_t>(GetInTensorCount())) {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INVALID_VALUE, "input tensor number should be 2");
    }
    if (outTensorDescs.size() != static_cast<size_t>(GetOutTensorCount())) {
        return AsdOps::Status::FailStatus(AsdOps::ErrorType::ERROR_INVALID_VALUE, "input tensor number should be 1");
    }

    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;

    AsdOps::SVector<int64_t> tensorDims = inTensors.at(0).desc.dims;
    AsdOps::SVector<int64_t> indicesDims = inTensors.at(1).desc.dims;
    AsdOps::SVector<int64_t> outTensorDims;

    for (size_t i = 0; i < static_cast<size_t>(param_.axis) && i < tensorDims.size(); i++) {
        outTensorDims.push_back(tensorDims[i]);
    }
    for (size_t i = 0; i < indicesDims.size(); ++i) {
        outTensorDims.push_back(indicesDims[i]);
    }
    for (size_t i = param_.axis + 1; i < tensorDims.size(); ++i) {
        outTensorDims.push_back(tensorDims[i]);
    }

    outTensorDescs.at(0).dims = outTensorDims;

    for (size_t i = 0; i < outTensorDims.size(); i++) {
        ASD_LOG(DEBUG) << "GatherV2InferShape OutTensor dims[" << i << "] = "
                       << outTensorDims[i];
    }

    return AsdOps::Status::OkStatus();
}

RunnerBuilder *GatherOperation::FindBestRunnerBuilder() const { return runnerBuilders_.at(0); }
} // namespace AclTransformer
