/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_910_95_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"
#include "kernels/matmul/tiling/tiling_data.h"

namespace AsdOps {
Status MatMul91095Tiling(
    const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo, const BinHandle &binHandle)
{
    return Status::OkStatus();
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    size_t inTensorIndex = 0;
    const TensorDesc &x1TensorDesc = launchParam.GetInTensor(inTensorIndex++).desc;
    const TensorDesc &x2TensorDesc = launchParam.GetInTensor(inTensorIndex++).desc;
    const TensorDesc &outTensorDesc = launchParam.GetOutTensor(0).desc;

    if (opParam.withBias) {
        const TensorDesc &biasTensorDesc = launchParam.GetInTensor(inTensorIndex++).desc;
        auto runner = AsdOpsGeRt::TbeTilingRunner()
                          .SetName("MatMulV3")
                          .SetKernelName(kernelName)
                          .AddInput(x1TensorDesc.dtype, x1TensorDesc.format, x1TensorDesc.dims)
                          .AddInput(x2TensorDesc.dtype, x2TensorDesc.format, x2TensorDesc.dims)
                          .AddInput(biasTensorDesc.dtype, biasTensorDesc.format, biasTensorDesc.dims)
                          .AddOutput(outTensorDesc.dtype, outTensorDesc.format, outTensorDesc.dims)
                          .AddAttrBool(opParam.transposeA)
                          .AddAttrBool(opParam.transposeB)
                          .AddAttrInt(0)
                          .AddAttrBool(false)
                          .AddAttrInt64(0);  // 0x40 is high precision
        return GetTilingFromRunner(kernelInfo, runner, binHandle);
    }
    auto runner = AsdOpsGeRt::TbeTilingRunner()
                      .SetName("MatMulV3")
                      .SetKernelName(kernelName)
                      .AddInput(x1TensorDesc.dtype, x1TensorDesc.format, x1TensorDesc.dims)
                      .AddInput(x2TensorDesc.dtype, x2TensorDesc.format, x2TensorDesc.dims)
                      .AddOutput(outTensorDesc.dtype, outTensorDesc.format, outTensorDesc.dims)
                      .AddAttrBool(opParam.transposeA)
                      .AddAttrBool(opParam.transposeB)
                      .AddAttrInt(0)
                      .AddAttrBool(false)
                      .AddAttrInt64(0);  // 0x40 is high precision
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status BatchMatMul91095Tiling(
    const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo, const BinHandle &binHandle)
{
    return Status::OkStatus();
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    size_t inTensorIndex = 0;
    const TensorDesc &x1TensorDesc = launchParam.GetInTensor(inTensorIndex++).desc;
    const TensorDesc &x2TensorDesc = launchParam.GetInTensor(inTensorIndex++).desc;
    const TensorDesc &outTensorDesc = launchParam.GetOutTensor(0).desc;

    if (opParam.withBias) {
        const TensorDesc &biasTensorDesc = launchParam.GetInTensor(inTensorIndex++).desc;
        auto runner = AsdOpsGeRt::TbeTilingRunner()
                          .SetName("BatchMatMulV3")
                          .SetKernelName(kernelName)
                          .AddInput(x1TensorDesc.dtype, x1TensorDesc.format, x1TensorDesc.dims)
                          .AddInput(x2TensorDesc.dtype, x2TensorDesc.format, x2TensorDesc.dims)
                          .AddInput(biasTensorDesc.dtype, biasTensorDesc.format, biasTensorDesc.dims)
                          .AddOutput(outTensorDesc.dtype, outTensorDesc.format, outTensorDesc.dims)
                          .AddAttrBool(opParam.transposeA)
                          .AddAttrBool(opParam.transposeB)
                          .AddAttrInt(0)
                          .AddAttrBool(false)
                          .AddAttrInt64(0);  // 0x40 is high precision
        return GetTilingFromRunner(kernelInfo, runner, binHandle);
    }
    auto runner = AsdOpsGeRt::TbeTilingRunner()
                      .SetName("BatchMatMulV3")
                      .SetKernelName(kernelName)
                      .AddInput(x1TensorDesc.dtype, x1TensorDesc.format, x1TensorDesc.dims)
                      .AddInput(x2TensorDesc.dtype, x2TensorDesc.format, x2TensorDesc.dims)
                      .AddOutput(outTensorDesc.dtype, outTensorDesc.format, outTensorDesc.dims)
                      .AddAttrBool(opParam.transposeA)
                      .AddAttrBool(opParam.transposeB)
                      .AddAttrInt(0)
                      .AddAttrBool(false)
                      .AddAttrInt64(0);  // 0x40 is high precision
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
}  // namespace AsdOps