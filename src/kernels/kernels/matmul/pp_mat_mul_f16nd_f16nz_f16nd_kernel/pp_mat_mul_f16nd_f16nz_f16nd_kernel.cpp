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
#include "kernels/matmul/tiling/pp_matmul_tiling.h"
#include "kernels/matmul/tiling/tiling_data.h"
#include "kernels/matmul/common/common.h"
#include "kernels/matmul/common/common_tiling.h"
#include "sink_common.h"

namespace AsdOps {
class PpMatMulF16NDF16NZF16NDKernel : public KernelBase {
public:
    explicit PpMatMulF16NDF16NZF16NDKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B, "platform not support",
                     return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 4, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed", return false);

        const auto &descA = launchParam.GetInTensor(0).desc;
        const auto &descB = launchParam.GetInTensor(1).desc;

        MKI_CHECK(descA.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(descA.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);

        MKI_CHECK(descB.format == TENSOR_FORMAT_FRACTAL_NZ, "tensor format invalid", return false);
        MKI_CHECK(descB.dims.size() == 4, "tensor dims invalid", return false);
        MKI_CHECK(descB.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);

        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        if (opParam.matmulType == OpParam::MatMul::MatMulType::MATMUL_EIN_SUM) {
            MKI_CHECK(descA.dims.size() == 3, "Matrix A must be a 3D-tensor in MATMUL_EIN_SUM scenario.", return false);
            MKI_CHECK(descA.dims[1] == descB.dims[0], "tensor dims invalid: batchA != batchB", return false);
        } else {
            if (descA.dims.size() == 2) { // 2: The first input is a two-dimensional tensor while batch size is one.
                MKI_CHECK(descB.dims[0] == 1, "tensor dims invalid", return false);
            } else {
                MKI_CHECK(descA.dims.size() == 3, "tensor dims invalid", return false);
                MKI_CHECK(descA.dims[0] == descB.dims[0], "tensor dims invalid", return false);
            }
        }

        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return optiling::CallGeTiling("PpMatMul", *GetBinHandle(), launchParam,
                                      AsdOps::GetMkiSpecificAttr<OpParam::MatMul>, kernelInfo_);
    }
};
REG_KERNEL_BASE(PpMatMulF16NDF16NZF16NDKernel);
} // namespace AsdOps
