/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "acltransformer/ops/ffn_operation.h"
#include <asdops/utils/log/log.h>
#include <asdops/utils/svector/svector.h>
#include "ffn_torch_runner_builder.h"
#include "ffn_ops_runner_builder.h"

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

namespace AclTransformer {
FfnOperation::FfnOperation(const FfnParam &param) : Operation("FfnOperation"), param_(param)
{
    runnerBuilders_ = {new FfnOpsRunnerBuilder(param_), new FfnTorchRunnerBuilder(param_)};
}

FfnOperation::~FfnOperation() {}

AsdOps::Status FfnOperation::InferShape(const std::vector<AsdOps::TensorDesc> &inTensorDescs,
                                        std::vector<AsdOps::TensorDesc> &outTensorDescs)
{
    if (inTensorDescs.size() != 3) {
        return AsdOps::Status::FailStatus(1, "inTensorDescs size is not 3");
    }
    outTensorDescs.resize(1);
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dims = {inTensorDescs.at(0).dims[0], inTensorDescs.at(0).dims[1],
                                 inTensorDescs.at(1).dims[0]}; // to do shape

    return AsdOps::Status::OkStatus();
}

bool FfnOperation::IsConsistent(const std::vector<AsdOps::TensorDesc> &inTensorDescs,
                                std::vector<AsdOps::TensorDesc> &outTensorDescs) const
{
    ASDOPS_CHECK_TRUE(inTensorDescs.size() == static_cast<size_t>(DIM_3), return false);
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

int64_t FfnOperation::GetTensorBatch(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return DIM_1;
    }
    return tensorDesc.dims[DIM_0];
}

int64_t FfnOperation::GetTensorH(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return tensorDesc.dims[DIM_0];
    }
    return tensorDesc.dims[DIM_1];
}

int64_t FfnOperation::GetTensorW(const AsdOps::TensorDesc &tensorDesc) const
{
    // make sure dims.size() == 2 or 3
    if (tensorDesc.dims.size() == DIM_2) {
        return tensorDesc.dims[DIM_1];
    }
    return tensorDesc.dims[DIM_2];
}

RunnerBuilder *FfnOperation::FindBestRunnerBuilder(const VariantPack &variantPack) { return runnerBuilders_.at(1); }
} // namespace AclTransformer