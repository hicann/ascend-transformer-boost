/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "kernels/matmul/tiling/matmul_nd_tiling.h"
#include "kernels/matmul/tiling/matmul_nz_tiling.h"

namespace AsdOps {
class QuantBatchMatmulI8Kernel : public KernelBase {
public:
    explicit QuantBatchMatmulI8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return QuantBatchMatmulNdTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};
REG_KERNEL_BASE(QuantBatchMatmulI8Kernel);
} // namespace AsdOps
