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
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/bf16/bf16_t.h>
#include <mki/utils/fp16/fp16_t.h>
#include "asdops/params/params.h"
#include "quant_tiling/quant_tiling.h"
#include "quant_tiling/tiling_data.h"
#include "kernels/elewise/tiling/elewise_tiling.h"

namespace AsdOps {
class QuantKernel : public KernelBase {
public:
    explicit QuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto outTensor = launchParam.GetOutTensor(0);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_QUANT, "quant: param type invalid", return false);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "input format invalid", return false);
        MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "output format invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return QuantF16Tiling(launchParam, kernelInfo_);
    }
};

// QuantF16
class QuantF16Kernel : public QuantKernel {
public:
    explicit QuantF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "tensor dtype unsupported", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(QuantF16TilingData);
    }
};
REG_KERNEL_BASE(QuantF16Kernel);

// QuantAptF16
class QuantAptF16Kernel : public QuantKernel {
public:
    explicit QuantAptF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "tensor dtype unsupported", return false);
        return true;
    }

    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return Status::OkStatus();
    }
};
REG_KERNEL_BASE(QuantAptF16Kernel);

//QuantAptKernel for 95
class QuantAptKernel : public KernelBase {
public:
    explicit QuantAptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto outTensor = launchParam.GetOutTensor(0);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_QUANT, "quant: param type invalid", return false);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "input format invalid", return false);
        MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "output format invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        size_t constTensorSize = Utils::GetConstTensorSize<fp16_t>({static_cast<fp16_t>(1)});
        return launchBufferSize_ + constTensorSize * 2;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = QuantPerTensorTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        
        auto param = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        float scale = param.inputScale;
        int offset = param.inputOffset;
        
        const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
        SVector<float> ScaleVector = {scale};
        SVector<float> OffsetVector = {static_cast<float>(offset)};
        if(tensorDesc0.dtype == TENSOR_DTYPE_FLOAT16){
            kernelInfo_.SetConstTensorOffset(launchBufferSize_);
            kernelInfo_.AddConstTensorData<float, fp16_t>(1, ScaleVector);
            kernelInfo_.AddConstTensorData<float, fp16_t>(2, OffsetVector);
            return Status::OkStatus();
        } else if (tensorDesc0.dtype == TENSOR_DTYPE_BF16) {
            kernelInfo_.SetConstTensorOffset(launchBufferSize_);
            kernelInfo_.AddConstTensorData<float, bf16_t>(1, ScaleVector);
            kernelInfo_.AddConstTensorData<float, bf16_t>(2, OffsetVector);
            return Status::OkStatus();
        }
        return Status::FailStatus(-1, "inpux dtype is not fp16 or bf16!");
    }
};

// QuantPerTensorBF16toHi8
class QuantPerTensorBF16toHi8Kernel : public QuantAptKernel {
public:
    explicit QuantPerTensorBF16toHi8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_HIFLOAT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTensorBF16toHi8Kernel);

// QuantPerTensorBF16toE4M3
class QuantPerTensorBF16toE4M3Kernel : public QuantAptKernel {
public:
    explicit QuantPerTensorBF16toE4M3Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E4M3FN,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTensorBF16toE4M3Kernel);

// QuantPerTensorBF16toE5M2
class QuantPerTensorBF16toE5M2Kernel : public QuantAptKernel {
public:
    explicit QuantPerTensorBF16toE5M2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E5M2,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTensorBF16toE5M2Kernel);

// QuantPerTensorF16toHi8
class QuantPerTensorF16toHi8Kernel : public QuantAptKernel {
public:
    explicit QuantPerTensorF16toHi8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_HIFLOAT8,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTensorF16toHi8Kernel);

// QuantPerTensorF16toE4M3
class QuantPerTensorF16toE4M3Kernel : public QuantAptKernel {
public:
    explicit QuantPerTensorF16toE4M3Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E4M3FN,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTensorF16toE4M3Kernel);

// QuantPerTensorF16toE5M2
class QuantPerTensorF16toE5M2Kernel : public QuantAptKernel {
public:
    explicit QuantPerTensorF16toE5M2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : QuantAptKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(QuantAptKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT8_E5M2,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(QuantPerTensorF16toE5M2Kernel);

} // namespace AsdOps