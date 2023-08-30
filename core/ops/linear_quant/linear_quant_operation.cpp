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
#include "acltransformer/ops/linear_quant_operation.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/svector/svector.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "linear_quant_ops_runner_builder.h"

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

namespace AclTransformer {
LinearQuantOperation::LinearQuantOperation(const LinearQuantParam &param)
    : Operation("LinearQuantOperation"), param_(param)
{
    runnerBuilders_ = {new LinearQuantOpsRunnerBuilder(param_)};
}

LinearQuantOperation::~LinearQuantOperation() {}

uint64_t LinearQuantOperation::GetInTensorCount() const { return 4; }

uint64_t LinearQuantOperation::GetOutTensorCount() const { return 1; }

AsdOps::Status LinearQuantOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                                    AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    if (inTensors.at(1).desc.format == AsdOps::TENSOR_FORMAT_FRACTAL_NZ) {
        outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1],
                                    inTensors.at(1).desc.dims[2]}; // to do shape
    } else {
        outTensorDescs.at(0).dims = {inTensors.at(0).desc.dims[0], inTensors.at(0).desc.dims[1],
                                    inTensors.at(1).desc.dims[0]}; // to do shape
    }
    outTensorDescs.at(0).dtype = AsdOps::TENSOR_DTYPE_FLOAT16;
    return AsdOps::Status::OkStatus();
}

bool LinearQuantOperation::IsConsistent(const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs,
                                        AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    ASDOPS_CHECK_TRUE(inTensorDescs.size() == static_cast<size_t>(DIM_4), return false);
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

int64_t LinearQuantOperation::GetTensorBatch(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return DIM_1;
    }
    return tensorDesc.dims[DIM_0];
}

int64_t LinearQuantOperation::GetTensorH(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return tensorDesc.dims[DIM_0];
    }
    return tensorDesc.dims[DIM_1];
}

int64_t LinearQuantOperation::GetTensorW(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return tensorDesc.dims[DIM_1];
    }
    return tensorDesc.dims[DIM_2];
}

RunnerBuilder *LinearQuantOperation::FindBestRunnerBuilder() const
{
    size_t index = 0;
    return runnerBuilders_.at(index);
}
} // namespace AclTransformer