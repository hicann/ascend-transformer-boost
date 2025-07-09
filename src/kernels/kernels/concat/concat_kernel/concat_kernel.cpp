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
#include "asdops/params/concat.h"
#include "kernels/concat/tiling/concat_tiling.h"
#include <mki/utils/platform/platform_info.h>

namespace AsdOps {
using namespace Mki;
class ConcatKernel : public KernelBase {
public:
    explicit ConcatKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Concat),
            "transpose: param type invalid", return false);
        OpParam::Concat param = AnyCast<OpParam::Concat>(launchParam.GetParam());
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        MKI_CHECK(dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_FLOAT || dtype == TENSOR_DTYPE_BF16,
                  "input data type error", return false);
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        if (param.concatDim < 0) {
            int realConcatDim = static_cast<int>(dims.size()) + param.concatDim;
            MKI_CHECK((realConcatDim >= 0), "Incorrect concatDim.", return false);
        } else {
            MKI_CHECK((param.concatDim < static_cast<int>(dims.size())), "ConcatDim Oversize.", return false);
        }

        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return Concat2InputsTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

class ConcatF16Input2Kernel : public ConcatKernel {
public:
    explicit ConcatF16Input2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ConcatF16Input2Kernel);

class ConcatF32Input2Kernel : public ConcatKernel {
public:
    explicit ConcatF32Input2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ConcatF32Input2Kernel);

class ConcatAptKernel : public KernelBase {
public:
    explicit ConcatAptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        size_t inputNum = launchParam.GetInputLenCount() > 0 ? launchParam.GetInputLenCount() : launchParam.GetInTensorCount();
        size_t outputNum = launchParam.GetOutputLenCount() > 0 ? launchParam.GetOutputLenCount() : launchParam.GetOutTensorCount();
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Concat),
            "transpose: param type invalid", return false);
        OpParam::Concat param = AnyCast<OpParam::Concat>(launchParam.GetParam());
        MKI_CHECK(inputNum == 1, "input num invalid", return false);
        MKI_CHECK(outputNum == 1, "output num invalid", return false);
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        MKI_CHECK(dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_FLOAT || dtype == TENSOR_DTYPE_INT8
                || dtype == TENSOR_DTYPE_INT16 || dtype == TENSOR_DTYPE_INT32 || dtype == TENSOR_DTYPE_INT64
                || dtype == TENSOR_DTYPE_BF16, "input data type error", return false);
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        if (param.concatDim < 0) {
            int realConcatDim = static_cast<int>(dims.size()) + param.concatDim;
            MKI_CHECK((realConcatDim >= 0), "Incorrect concatDim.", return false);
        } else {
            MKI_CHECK((param.concatDim < static_cast<int>(dims.size())), "ConcatDim Oversize.", return false);
        }

        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return ConcatAptTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// ConcatI8Apt
class ConcatI8AptKernel : public ConcatAptKernel {
public:
    explicit ConcatI8AptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ConcatAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(ConcatI8AptKernel);

// ConcatI16Apt
class ConcatI16AptKernel : public ConcatAptKernel {
public:
    explicit ConcatI16AptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ConcatAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT16,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(ConcatI16AptKernel);

class ConcatF16AptKernel : public ConcatAptKernel {
public:
    explicit ConcatF16AptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ConcatAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(ConcatF16AptKernel);

class ConcatBF16AptKernel : public ConcatAptKernel {
public:
    explicit ConcatBF16AptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ConcatAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(ConcatBF16AptKernel);

class ConcatI32AptKernel : public ConcatAptKernel {
public:
    explicit ConcatI32AptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ConcatAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(ConcatI32AptKernel);

class ConcatF32AptKernel : public ConcatAptKernel {
public:
    explicit ConcatF32AptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ConcatAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(ConcatF32AptKernel);

class ConcatI64AptKernel : public ConcatAptKernel {
public:
    explicit ConcatI64AptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ConcatAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(ConcatI64AptKernel);
} // namespace AsdOps