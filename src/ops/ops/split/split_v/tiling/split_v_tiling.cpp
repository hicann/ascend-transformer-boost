/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "split_v_tiling.h"

#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"

static constexpr int OUT_TENSOR1 = 0;
static constexpr int OUT_TENSOR2 = 1;
static constexpr int OUT_TENSOR3 = 2;

namespace AsdOps {
Status SplitV2OutputsTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                            const BinHandle &binHandle)
{
    auto &param = AnyCast<OpParam::Split>(launchParam.GetParam());
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut1 = launchParam.GetOutTensor(OUT_TENSOR1).desc;
    const auto &tensorDescOut2 = launchParam.GetOutTensor(OUT_TENSOR2).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.splitSize.size())},
                       param.splitSize.data(), param.splitSize.size() * sizeof(int32_t))
        .AddConstInput(TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.splitVDim.size())},
                       param.splitVDim.data(), param.splitVDim.size() * sizeof(int32_t))
        .AddOutput(tensorDescOut1.dtype, tensorDescOut1.format, tensorDescOut1.dims)
        .AddOutput(tensorDescOut2.dtype, tensorDescOut2.format, tensorDescOut1.dims)
        .AddAttrInt(param.splitNum);
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status SplitV3OutputsTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                            const BinHandle &binHandle)
{
    auto &param = AnyCast<OpParam::Split>(launchParam.GetParam());
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut1 = launchParam.GetOutTensor(OUT_TENSOR1).desc;
    const auto &tensorDescOut2 = launchParam.GetOutTensor(OUT_TENSOR2).desc;
    const auto &tensorDescOut3 = launchParam.GetOutTensor(OUT_TENSOR3).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.splitSize.size())},
                       param.splitSize.data(), param.splitSize.size() * sizeof(int32_t))
        .AddConstInput(TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.splitVDim.size())},
                       param.splitVDim.data(), param.splitVDim.size() * sizeof(int32_t))
        .AddOutput(tensorDescOut1.dtype, tensorDescOut1.format, tensorDescOut1.dims)
        .AddOutput(tensorDescOut2.dtype, tensorDescOut2.format, tensorDescOut2.dims)
        .AddOutput(tensorDescOut3.dtype, tensorDescOut3.format, tensorDescOut3.dims)
        .AddAttrInt(param.splitNum);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps