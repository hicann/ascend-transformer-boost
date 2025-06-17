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
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "dynamic_quant_tiling/dynamic_quant_tiling.h"
#include "dynamic_quant_tiling/tiling_data.h"
#include "kernels/elewise/tiling/elewise_tiling.h"

namespace AsdOps {
class DynamicQuantKernel : public KernelBase {
public:
    explicit DynamicQuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        const int64_t outTensorIdxTwo = 2;
        const int64_t outTensorNumThree = 3;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == outTensorNumThree, "output num invalid", return false);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_DYNAMIC_QUANT, "dynamic quant: param type invalid",
            return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.format == TENSOR_FORMAT_ND,
            "input format invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.format == TENSOR_FORMAT_ND,
            "output1 format invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(1).desc.format == TENSOR_FORMAT_ND,
            "output2 format invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(outTensorIdxTwo).desc.format == TENSOR_FORMAT_ND,
            "output3 format invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(outTensorIdxTwo).desc.dtype == TENSOR_DTYPE_FLOAT,
            "tensor dtype unsupported", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return DynamicQuantTiling(launchParam, kernelInfo_);
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(DynamicQuantTilingData);
    }
};
REG_KERNEL_BASE(DynamicQuantKernel);

class DynamicQuantAptKernel : public KernelBase {
public:
    explicit DynamicQuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        // MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        // MKI_CHECK(launchParam.GetOutTensorCount() == 2, "output num invalid", return false);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_DYNAMIC_QUANT, "dynamic quant: param type invalid",
            return false);
        // MKI_CHECK(launchParam.GetInTensor(0).desc.format == TENSOR_FORMAT_ND,
        //     "input format invalid", return false);
        // MKI_CHECK(launchParam.GetOutTensor(0).desc.format == TENSOR_FORMAT_ND,
        //     "output1 format invalid", return false);
        // MKI_CHECK(launchParam.GetOutTensor(1).desc.format == TENSOR_FORMAT_ND,
        //     "output2 format invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8 ||
            launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_HIFLOAT8 ||
            launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E4M3FN ||
            launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E5M2, 
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT,
            "tensor dtype unsupported", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return DynamicQuantAptTiling(launchParam, kernelInfo_);
    }
};

// DynamicQuantAptBF16toInt8
class DynamicQuantAptBF16toInt8Kernel : public DynamicQuantAptKernel {
public:
    explicit DynamicQuantAptBF16toInt8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DynamicQuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(DynamicQuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(DynamicQuantAptBF16toInt8Kernel);

// DynamicQuantAptBF16toHi8
class DynamicQuantAptBF16toHi8Kernel : public DynamicQuantAptKernel {
public:
    explicit DynamicQuantAptBF16toHi8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DynamicQuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(DynamicQuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_HIFLOAT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(DynamicQuantAptBF16toHi8Kernel);

// DynamicQuantAptBF16toE4M3
class DynamicQuantAptBF16toE4M3Kernel : public DynamicQuantAptKernel {
public:
    explicit DynamicQuantAptBF16toE4M3Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DynamicQuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(DynamicQuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E4M3FN,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(DynamicQuantAptBF16toE4M3Kernel);

// DynamicQuantAptBF16toE5M2
class DynamicQuantAptBF16toE5M2Kernel : public DynamicQuantAptKernel {
public:
    explicit DynamicQuantAptBF16toE5M2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DynamicQuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(DynamicQuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E5M2,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(DynamicQuantAptBF16toE5M2Kernel);

// DynamicQuantAptF16toInt8
class DynamicQuantAptF16toInt8Kernel : public DynamicQuantAptKernel {
public:
    explicit DynamicQuantAptF16toInt8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DynamicQuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(DynamicQuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(DynamicQuantAptF16toInt8Kernel);

// DynamicQuantAptF16toHi8
class DynamicQuantAptF16toHi8Kernel : public DynamicQuantAptKernel {
public:
    explicit DynamicQuantAptF16toHi8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DynamicQuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(DynamicQuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_HIFLOAT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(DynamicQuantAptF16toHi8Kernel);

// DynamicQuantAptF16toE4M3
class DynamicQuantAptF16toE4M3Kernel : public DynamicQuantAptKernel {
public:
    explicit DynamicQuantAptF16toE4M3Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DynamicQuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(DynamicQuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E4M3FN,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(DynamicQuantAptF16toE4M3Kernel);

// DynamicQuantAptF16toE5M2
class DynamicQuantAptF16toE5M2Kernel : public DynamicQuantAptKernel {
public:
    explicit DynamicQuantAptF16toE5M2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : DynamicQuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(DynamicQuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E5M2,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(DynamicQuantAptF16toE5M2Kernel);
} // namespace AsdOps