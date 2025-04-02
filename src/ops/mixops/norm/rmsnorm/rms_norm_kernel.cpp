/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
#include "mixops/norm/common/common_tiling_data.h"
#include "mixops/norm/rmsnorm/tiling/rms_norm_tiling.h"

namespace AtbOps {
class RmsNormKernel : public KernelBase {
public:
    explicit RmsNormKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RmsNormCommonTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return RmsNormTiling(launchParam, kernelInfo_);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
            "norm: param type invalid", return false);
        AtbOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AtbOps::OpParam::Norm::NormType type = param.normType;
        MKI_CHECK(type == AtbOps::OpParam::Norm::RMS_NORM, "rmsnorm: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        TensorDType dtype0 = launchParam.GetInTensor(0).desc.dtype;
        TensorDType dtype1 = launchParam.GetInTensor(1).desc.dtype;
        TensorFormat format0 = launchParam.GetInTensor(0).desc.format;
        TensorFormat format1 = launchParam.GetInTensor(1).desc.format;
        MKI_CHECK((dtype0 == TENSOR_DTYPE_FLOAT16 || dtype0 == TENSOR_DTYPE_BF16) && dtype1 == dtype0,
            "input dtype invalid", return false);
        MKI_CHECK(format0 == TENSOR_FORMAT_ND && format1 == format0,
            "input format invalid", return false);
        uint32_t inTensor0Row = launchParam.GetInTensor(0).desc.dims.size();
        MKI_CHECK(inTensor0Row != 0, "the dimension of input0 should not be 0", return false);
        uint32_t inTensor0Col = launchParam.GetInTensor(0).desc.dims[inTensor0Row - 1];
        MKI_CHECK(inTensor0Col != 0, "the dimension of input0 should not be 0", return false);
        uint32_t inTensor1Row = launchParam.GetInTensor(1).desc.dims.size();
        MKI_CHECK(inTensor1Row != 0, "the dimension of input1 should not be 0", return false);
        uint32_t inTensor1Col = launchParam.GetInTensor(1).desc.dims[inTensor1Row - 1];
        MKI_CHECK(inTensor1Col != 0, "the dimension of input1 should not be 0", return false);
        MKI_CHECK((inTensor0Col % 16) == 0, "input0 is not a multiple of 16", return false);
        return true;
    }
};
REG_KERNEL_BASE(RmsNormKernel);
} // namespace AtbOps