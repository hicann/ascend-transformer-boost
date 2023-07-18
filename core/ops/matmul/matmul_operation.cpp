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
#include "acltransformer/ops/matmul_operation.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/svector/svector.h>
#include <asdops/utils/singleton/singleton.h>
#include "acltransformer/config.h"
#include "matmul_ops_runner_builder.h"

static constexpr int64_t DIM_0 = 0;
static constexpr int64_t DIM_1 = 1;
static constexpr int64_t DIM_2 = 2;
static constexpr int64_t DIM_3 = 3;

namespace AclTransformer {
MatmulOperation::MatmulOperation(const MatmulParam &param) : Operation("MatmulOperation"), param_(param)
{
    runnerBuilders_ = {new MatmulOpsRunnerBuilder(param_)};
}

MatmulOperation::~MatmulOperation() {}

uint64_t MatmulOperation::GetInTensorCount() const
{
    return DIM_2;
}

uint64_t MatmulOperation::GetOutTensorCount() const
{
    return DIM_1;
}

AsdOps::Status MatmulOperation::InferShapeImpl(const AsdOps::SVector<AsdOps::Tensor> &inTensors,
                                               AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    // input * weight
    outTensorDescs.at(0).dtype = inTensors.at(0).desc.dtype;
    outTensorDescs.at(0).format = inTensors.at(0).desc.format;
    auto inTensorADims = inTensors.at(0).desc.dims.size();
    auto inTensorBDims = inTensors.at(1).desc.dims.size();
    // 当前仅支持2维*2维，3维*3维，3维*2维
    if (inTensorADims == DIM_3) {
        auto outTensorDim0 = inTensors.at(0).desc.dims[0];
        auto outTensorDim1 = param_.transposeA ? inTensors.at(0).desc.dims[inTensorADims - 1]
                                               : inTensors.at(0).desc.dims[inTensorADims - 2];
        auto outTensorDim2 = param_.transposeB ? inTensors.at(1).desc.dims[inTensorBDims - 2]
                                               : inTensors.at(1).desc.dims[inTensorBDims - 1];
        outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1, outTensorDim2};
    } else {
        auto outTensorDim0 = param_.transposeA ? inTensors.at(0).desc.dims[1] : inTensors.at(0).desc.dims[0];
        auto outTensorDim1 = param_.transposeB ? inTensors.at(1).desc.dims[0] : inTensors.at(1).desc.dims[1];
        outTensorDescs.at(0).dims = {outTensorDim0, outTensorDim1};
    }

    return AsdOps::Status::OkStatus();
}

bool MatmulOperation::IsConsistent(const AsdOps::SVector<AsdOps::TensorDesc> &inTensorDescs,
                                   AsdOps::SVector<AsdOps::TensorDesc> &outTensorDescs) const
{
    ASDOPS_CHECK_TRUE(inTensorDescs.size() == static_cast<size_t>(DIM_2), return false);
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

int64_t MatmulOperation::GetTensorBatch(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return DIM_1;
    }
    return tensorDesc.dims[DIM_0];
}

RunnerBuilder *MatmulOperation::FindBestRunnerBuilder() const
{
    return runnerBuilders_.at(0);
}
} // namespace AclTransformer