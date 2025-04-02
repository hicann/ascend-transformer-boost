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
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
#include "tiling/cast_tiling.h"
#include "tiling/tiling_data.h"

namespace AtbOps {
// CastWide
class CastWideKernel : public KernelBase {
public:
    explicit CastWideKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_CAST, "cast: param type invalid", return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto outTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "input format invalid", return false);
        MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "output format invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(CastTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return CastCommonTiling(launchParam, kernelInfo_);
    }
};

REG_KERNEL_BASE(CastWideKernel);
} // namespace AtbOps