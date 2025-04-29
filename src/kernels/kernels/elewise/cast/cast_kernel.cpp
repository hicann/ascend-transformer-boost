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
#include "asdops/params/params.h"
#include "kernels/elewise/tiling/elewise_tiling.h"
#include "tiling/cast_tiling.h"
#include "tiling/tiling_data.h"

namespace AsdOps {
class CastKernel : public KernelBase {
public:
    explicit CastKernel(const std::string &kernelName, const BinHandle *handle) noexcept
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
    
    Status InitImpl(const LaunchParam &launchParam) override
    {
        return ElewiseCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// CastF16toF32
class CastF16toF32Kernel : public CastKernel {
public:
    explicit CastF16toF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CastKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(CastKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(CastF16toF32Kernel);

// CastI64toI32
class CastI64toI32Kernel : public CastKernel {
public:
    explicit CastI64toI32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CastKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(CastKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(CastI64toI32Kernel);

// CastF32toF16
class CastF32toF16Kernel : public CastKernel {
public:
    explicit CastF32toF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CastKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(CastKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(CastF32toF16Kernel);

// CastI32toF16
class CastI32toF16Kernel : public CastKernel {
public:
    explicit CastI32toF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CastKernel(kernelName, handle)
    {
    }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(CastKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(CastI32toF16Kernel);

// CastI32toI64
class CastI32toI64Kernel : public CastKernel {
public:
    explicit CastI32toI64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CastKernel(kernelName, handle)
    {
    }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(CastKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(CastI32toI64Kernel);

// CastWide
class CastWideKernel : public CastKernel {
public:
    explicit CastWideKernel(const std::string &kernelName, const BinHandle *handle) : CastKernel(kernelName, handle) {}

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
} // namespace AsdOps