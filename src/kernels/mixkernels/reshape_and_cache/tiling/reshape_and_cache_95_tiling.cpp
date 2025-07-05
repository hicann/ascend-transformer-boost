/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <climits>
#include "reshape_and_cache_95_tiling.h"
#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "reshape_and_cache_tiling_dependency.h"
#include "tbe_tiling_runner.h"

namespace AtbOps {
using namespace Mki;
Status ReshapeAndCache95Tiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                     const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDesc2 = launchParam.GetInTensor(2).desc;
    const auto &tensorDesc3 = launchParam.GetInTensor(3).desc;
    const auto &tensorDesc4 = launchParam.GetInTensor(4).desc;
    const auto &tensorDescOut0 = launchParam.GetOutTensor(0).desc;
    const auto &tensorDescOut1 = launchParam.GetOutTensor(1).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("ScatterPaKvCache")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddInput(tensorDesc2.dtype, tensorDesc2.format, tensorDesc2.dims)
        .AddInput(tensorDesc3.dtype, tensorDesc3.format, tensorDesc3.dims)
        .AddInput(tensorDesc4.dtype, tensorDesc4.format, tensorDesc4.dims)
        .AddOutput(tensorDescOut0.dtype, tensorDescOut0.format, tensorDescOut0.dims)
        .AddOutput(tensorDescOut1.dtype, tensorDescOut1.format, tensorDescOut1.dims);

    return AsdOps::GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status ReshapeAndCacheCompressAlibi95Tiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                     const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDesc2 = launchParam.GetInTensor(2).desc;
    const auto &tensorDesc3 = launchParam.GetInTensor(3).desc;
    const auto &tensorDesc4 = launchParam.GetInTensor(4).desc;
    const auto &tensorDesc5 = launchParam.GetInTensor(5).desc;
    const auto &tensorDesc6 = launchParam.GetInTensor(6).desc;
    const auto &tensorDesc7 = launchParam.GetInTensor(7).desc;
    const auto &tensorDescOut0 = launchParam.GetOutTensor(0).desc;
    const auto &tensorDescOut1 = launchParam.GetOutTensor(1).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("ScatterPaKvCache")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddInput(tensorDesc2.dtype, tensorDesc2.format, tensorDesc2.dims)
        .AddInput(tensorDesc3.dtype, tensorDesc3.format, tensorDesc3.dims)
        .AddInput(tensorDesc4.dtype, tensorDesc4.format, tensorDesc4.dims)
        .AddInput(tensorDesc5.dtype, tensorDesc5.format, tensorDesc5.dims)
        .AddInput(tensorDesc6.dtype, tensorDesc6.format, tensorDesc6.dims)
        .AddInput(tensorDesc7.dtype, tensorDesc7.format, tensorDesc7.dims)
        .AddOutput(tensorDescOut0.dtype, tensorDescOut0.format, tensorDescOut0.dims)
        .AddOutput(tensorDescOut1.dtype, tensorDescOut1.format, tensorDescOut1.dims);

    return AsdOps::GetTilingFromRunner(kernelInfo, runner, binHandle);
}
}