/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_ELEWISE_TILING_H
#define ASCEND_OPS_ELEWISE_TILING_H

#include <mki/bin_handle.h>
#include <mki/launch_param.h>
#include <mki/kernel_info.h>

namespace AsdOps {
using namespace Mki;
Status ElewiseCommonTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                           const BinHandle &binHandle);
Status AddTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const BinHandle &binHandle);
Status CastTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const BinHandle &binHandle);
Status CosTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                 const BinHandle &binHandle);
Status MulsTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                  const BinHandle &binHandle);
Status BroadcastCommonTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const BinHandle &binHandle);
} // namespace AsdOps

#endif