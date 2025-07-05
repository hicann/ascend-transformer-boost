/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "elewise_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/bf16/bf16_t.h>
#include <mki/utils/fp16/fp16_t.h>
#include "asdops/params/elewise.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status ElewiseCommonTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                           const BinHandle &binHandle)
{
    const auto &tensorDesc = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .AddInput(tensorDesc.dtype, tensorDesc.format, tensorDesc.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status AddTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Add")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status SinTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                 const BinHandle &binHandle)
{
    const auto &tensorDesc = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Sin")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc.dtype, tensorDesc.format, tensorDesc.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status CastTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const BinHandle &binHandle)
{
    const auto &tensorDesc = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;
    auto param = AnyCast<OpParam::Elewise>(launchParam.GetParam());
    int dstType = param.outTensorType;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Cast")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc.dtype, tensorDesc.format, tensorDesc.dims)
        .AddAttrInt(dstType)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status MulTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                           const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;
    const auto &param = AnyCast<OpParam::Elewise>(launchParam.GetParam());

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Mul")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
        .AddAttrFloat(param.varAttr);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status CosTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                 const BinHandle &binHandle)
{
    const auto &tensorDesc = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Cos")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc.dtype, tensorDesc.format, tensorDesc.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status MulsTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                  const BinHandle &binHandle)
{
    const auto &tensorDesc = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;
    const auto &param = AnyCast<OpParam::Elewise>(launchParam.GetParam());

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Muls")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc.dtype, tensorDesc.format, tensorDesc.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
        .AddAttrFloat(param.varAttr);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status DynamicQuantAptTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                     const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut0 = launchParam.GetOutTensor(0).desc;
    const auto &tensorDescOut1 = launchParam.GetOutTensor(1).desc;
    const auto &param = AnyCast<OpParam::Elewise>(launchParam.GetParam());

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("DynamicQuant")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddOutput(tensorDescOut0.dtype, tensorDescOut0.format, tensorDescOut0.dims)
        .AddOutput(tensorDescOut1.dtype, tensorDescOut1.format, tensorDescOut1.dims)
        .AddAttrInt(param.outTensorType);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status QuantPerTensorTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                     const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut0 = launchParam.GetOutTensor(0).desc;
    const auto &param = AnyCast<OpParam::Elewise>(launchParam.GetParam());

    float scale = param.inputScale;
    int offset = param.inputOffset;
    int dstType = param.outTensorType;
    SVector<int64_t> paramShape = {static_cast<int64_t>(1)};

    if (tensorDesc0.dtype == TENSOR_DTYPE_FLOAT16) {
        fp16_t castScale;
        fp16_t castOffset;

        castScale = static_cast<fp16_t>(scale);
        castOffset = static_cast<fp16_t>(offset);
        SVector<fp16_t> ScaleVector = {castScale};
        SVector<fp16_t> OffsetVector = {castOffset};
        const auto &ScalePointer = ScaleVector;
        const auto &OffsetPointer = OffsetVector;

        auto runner = AsdOpsGeRt::TbeTilingRunner()
            .SetName("AscendQuantV2")
            .SetKernelName(kernelName)
            .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
            .AddConstInput(TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, paramShape, ScalePointer.data(), sizeof(fp16_t))
            .AddConstInput(TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, paramShape, OffsetPointer.data(), sizeof(fp16_t))
            .AddOutput(tensorDescOut0.dtype, tensorDescOut0.format, tensorDescOut0.dims)
            .AddAttrBool(false)
            .AddAttrStr("round")
            .AddAttrInt(dstType)
            .AddAttrInt(-1);

        return GetTilingFromRunner(kernelInfo, runner, binHandle);
    } else if (tensorDesc0.dtype == TENSOR_DTYPE_BF16) {
        bf16_t castScale;
        bf16_t castOffset;

        castScale = static_cast<bf16_t>(scale);
        castOffset = static_cast<bf16_t>(offset);
        SVector<bf16_t> ScaleVector = {castScale};
        SVector<bf16_t> OffsetVector = {castOffset};
        const auto &ScalePointer = ScaleVector;
        const auto &OffsetPointer = OffsetVector;

        auto runner = AsdOpsGeRt::TbeTilingRunner()
            .SetName("AscendQuantV2")
            .SetKernelName(kernelName)
            .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
            .AddConstInput(TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, paramShape, ScalePointer.data(), sizeof(bf16_t))
            .AddConstInput(TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, paramShape, OffsetPointer.data(), sizeof(bf16_t))
            .AddOutput(tensorDescOut0.dtype, tensorDescOut0.format, tensorDescOut0.dims)
            .AddAttrBool(false)
            .AddAttrStr("round")
            .AddAttrInt(dstType)
            .AddAttrInt(-1);

        return GetTilingFromRunner(kernelInfo, runner, binHandle);
    } else {
        return Status::FailStatus(-1, "ElewiseOperationQuant: xDtype should be float16/bf16.");
    }
}

Status RealDivTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                     const BinHandle &binHandle)
{
    // return Status::OkStatus();

    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("RealDiv")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status BroadcastCommonTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps