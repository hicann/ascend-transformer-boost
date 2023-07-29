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
#include "acltransformer/ops/linear_operation.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/svector/svector.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "linear_ops_runner_builder.h"
#include "linear_torch_runner_builder.h"

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

namespace AclTransformer {
LinearOperation::LinearOperation(const LinearParam &param) : Operation("LinearOperation"), param_(param)
{
#ifdef USE_TORCH_RUNNER
    runnerBuilders_ = {new LinearOpsRunnerBuilder(param_), new LinearTorchRunnerBuilder(param_)};
#else
    runnerBuilders_ = {new LinearOpsRunnerBuilder(param_)};
#endif
}

LinearOperation::~LinearOperation() {}

uint64_t LinearOperation::GetInTensorCount() const { return param_.hasBias ? DIM_3 : DIM_2; }

uint64_t LinearOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status LinearOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                               AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    // in * weight + bias
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    if (param_.transposeB) {
        outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1],
                                 inTensors.at(1).desc.dims[1]}; 
    } else {
        outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1],
                                 inTensors.at(1).desc.dims[0]}; 
    }
    return AsdOps::Status::OkStatus();
}

bool LinearOperation::IsConsistent(const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs,
                                   AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    ASDOPS_CHECK_TRUE(inTensorDescs.size() == GetInTensorCount(), return false);
    ASDOPS_CHECK_TRUE(outTensorDescs.size() == static_cast<size_t>(DIM_1), return false);
    auto inTensorDescA = inTensorDescs[0];
    auto inTensorDescB = inTensorDescs[1];
    ASDOPS_CHECK_TRUE(inTensorDescA.dims.size() == DIM_2 || inTensorDescA.dims.size() == DIM_3, return false);
    ASDOPS_CHECK_TRUE(inTensorDescB.dims.size() == DIM_2 || inTensorDescB.dims.size() == DIM_3, return false);
    int64_t batchA = GetTensorBatch(inTensorDescA);
    int64_t batchB = GetTensorBatch(inTensorDescB);
    if (batchA > 1 && batchB > 1) {
        ASDOPS_CHECK_TRUE(batchB == batchA, return false);
    }
    return true;
}

int64_t LinearOperation::GetTensorBatch(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return DIM_1;
    }
    return tensorDesc.dims[DIM_0];
}

int64_t LinearOperation::GetTensorH(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return tensorDesc.dims[DIM_0];
    }
    return tensorDesc.dims[DIM_1];
}

int64_t LinearOperation::GetTensorW(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return tensorDesc.dims[DIM_1];
    }
    return tensorDesc.dims[DIM_2];
}

RunnerBuilder *LinearOperation::FindBestRunnerBuilder() const
{
#ifdef USE_TORCH_RUNNER
    size_t index = AsdOps::GetSingleton<Config>().IsLinearOpsRunnerEnable() ? 0 : 1;
#else
    size_t index = 0;
#endif
    return runnerBuilders_.at(index);
}

} // namespace AclTransformer