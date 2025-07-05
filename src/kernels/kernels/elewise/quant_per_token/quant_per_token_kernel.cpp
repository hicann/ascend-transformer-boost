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
#include "kernels/elewise/tiling/elewise_tiling.h"

namespace AsdOps {
class QuantPerTokenKernel : public KernelBase {
public:
    explicit QuantPerTokenKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 2, "output num invalid", return false);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_QUANT_PER_TOKEN, "quant_per_token: param type invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return QuantPerTokenTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};


// QuantPerTokenBF16toInt8
class QuantPerTokenBF16toInt8Kernel : public QuantPerTokenKernel {
public:
    explicit QuantPerTokenBF16toInt8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantPerTokenKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantPerTokenKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTokenBF16toInt8Kernel);

// QuantPerTokenBF16toHi8
class QuantPerTokenBF16toHi8Kernel : public QuantPerTokenKernel {
public:
    explicit QuantPerTokenBF16toHi8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantPerTokenKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantPerTokenKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_HIFLOAT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTokenBF16toHi8Kernel);

// QuantPerTokenBF16toE4M3
class QuantPerTokenBF16toE4M3Kernel : public QuantPerTokenKernel {
public:
    explicit QuantPerTokenBF16toE4M3Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantPerTokenKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantPerTokenKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E4M3FN,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTokenBF16toE4M3Kernel);

// QuantPerTokenBF16toE5M2
class QuantPerTokenBF16toE5M2Kernel : public QuantPerTokenKernel {
public:
    explicit QuantPerTokenBF16toE5M2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantPerTokenKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantPerTokenKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E5M2,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTokenBF16toE5M2Kernel);

// QuantPerTokenF16toInt8
class QuantPerTokenF16toInt8Kernel : public QuantPerTokenKernel {
public:
    explicit QuantPerTokenF16toInt8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantPerTokenKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantPerTokenKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTokenF16toInt8Kernel);

// QuantPerTokenF16toHi8
class QuantPerTokenF16toHi8Kernel : public QuantPerTokenKernel {
public:
    explicit QuantPerTokenF16toHi8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantPerTokenKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantPerTokenKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_HIFLOAT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTokenF16toHi8Kernel);

// QuantPerTokenF16toE4M3
class QuantPerTokenF16toE4M3Kernel : public QuantPerTokenKernel {
public:
    explicit QuantPerTokenF16toE4M3Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantPerTokenKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantPerTokenKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E4M3FN,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTokenF16toE4M3Kernel);

// QuantPerTokenF16toE5M2
class QuantPerTokenF16toE5M2Kernel : public QuantPerTokenKernel {
public:
    explicit QuantPerTokenF16toE5M2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantPerTokenKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantPerTokenKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E5M2,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTokenF16toE5M2Kernel);
} // namespace AsdOps