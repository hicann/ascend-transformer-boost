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
#include <mki/utils/log/log.h>
#include <mki_loader/op_register.h>
#include "asdops/params/params.h"
#include "kernels/matmul/tiling/matmul_910_95_tiling.h"

namespace AsdOps {
using namespace Mki;
class MatMul91095Kernel : public KernelBase {
public:
    explicit MatMul91095Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return MatMul91095Tiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }

    Status Init(const LaunchParam &launchParam) override
    {
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        LaunchParam newLaunchParam = launchParam;
        newLaunchParam.AddInTensor({});
        if (!opParam.withBias) {
            newLaunchParam.AddInTensor({});
        }
        return KernelBase::Init(launchParam);
    }

    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        LaunchParam newLaunchParam = launchParam;
        newLaunchParam.AddInTensor({});
        if (!opParam.withBias) {
            newLaunchParam.AddInTensor({});
        }
        return KernelBase::Run(newLaunchParam, runInfo);
    }
};

class MatMulX1Float16NdX2Float16NdBiasFloat16Nd91095Kernel : public MatMul91095Kernel {
public:
    explicit MatMulX1Float16NdX2Float16NdBiasFloat16Nd91095Kernel(const std::string &kernelName,
                                                                  const BinHandle *handle) noexcept
        : MatMul91095Kernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(MatMulX1Float16NdX2Float16NdBiasFloat16Nd91095Kernel);

class MatMulX1Float16NdX2Float16NdBiasFloatNd91095Kernel : public MatMul91095Kernel {
public:
    explicit MatMulX1Float16NdX2Float16NdBiasFloatNd91095Kernel(const std::string &kernelName,
                                                                const BinHandle *handle) noexcept
        : MatMul91095Kernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(MatMulX1Float16NdX2Float16NdBiasFloatNd91095Kernel);

class MatMulX1Bf16NdX2Bf16NdBiasBf16Nd91095Kernel : public MatMul91095Kernel {
public:
    explicit MatMulX1Bf16NdX2Bf16NdBiasBf16Nd91095Kernel(const std::string &kernelName,
                                                         const BinHandle *handle) noexcept
        : MatMul91095Kernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(MatMulX1Bf16NdX2Bf16NdBiasBf16Nd91095Kernel);

class MatMulX1Bf16NdX2Bf16NdBiasFloatNd91095Kernel : public MatMul91095Kernel {
public:
    explicit MatMulX1Bf16NdX2Bf16NdBiasFloatNd91095Kernel(const std::string &kernelName,
                                                          const BinHandle *handle) noexcept
        : MatMul91095Kernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(MatMulX1Bf16NdX2Bf16NdBiasFloatNd91095Kernel);
} // namespace AsdOps
