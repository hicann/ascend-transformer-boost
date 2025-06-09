/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm_lingqu_tiling.h"
#include <cstring>
#include <mki/utils/env/env.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "tbe_tiling_runner.h"
#include "asdops/params/norm.h"

namespace AsdOps {
Status RmsNormLingQuTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo, const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDescOut0 = launchParam.GetOutTensor(0).desc;
    const auto &tensorDescOut1 = launchParam.GetOutTensor(1).desc;
    const auto &param = AnyCast<OpParam::Norm>(launchParam.GetParam());
    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("RmsNorm")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddOutput(tensorDescOut0.dtype, tensorDescOut0.format, tensorDescOut0.dims)
        .AddOutput(tensorDescOut1.dtype, tensorDescOut1.format, tensorDescOut1.dims)
        .AddAttrFloat(param.epsilon);
        return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps