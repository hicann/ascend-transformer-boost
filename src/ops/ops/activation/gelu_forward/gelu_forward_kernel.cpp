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
#include "ops/activation/gelu_forward/tiling/gelu_tiling.h"
#include "ops/activation/gelu_forward/tiling/tiling_data.h"
namespace AsdOps {
using namespace Mki;
// GeluForward
class GeluForwardKernel : public KernelBase {
public:
    explicit GeluForwardKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Activation),
                     "gelu_forward: param type invalid", return false);
        auto opParam = AnyCast<OpParam::Activation>(launchParam.GetParam());
        OpParam::Activation::ActivationType type = opParam.activationType;
        MKI_CHECK(type == OpParam::Activation::ACTIVATION_GELU, "activation type invalid",
                     return false);

        TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
        TensorDType outDtype = launchParam.GetOutTensor(0).desc.dtype;
        MKI_CHECK(inDtype == TENSOR_DTYPE_FLOAT || inDtype == TENSOR_DTYPE_FLOAT16 || inDtype == TENSOR_DTYPE_BF16,
                     "Input dtype invalid, should be float or float16 or bf16", return false);
        MKI_CHECK(outDtype == TENSOR_DTYPE_FLOAT || outDtype == TENSOR_DTYPE_FLOAT16 ||
                         outDtype == TENSOR_DTYPE_BF16,
                     "Output dtype invalid, should be float or float16 or bf16", return false);
        MKI_CHECK(inDtype == outDtype,
                     "Input dtype should be equal with output dtype", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(GeluForwardTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return GeluForwardTiling(launchParam, kernelInfo_);
    }
};
// 16 512

REG_KERNEL_BASE(GeluForwardKernel);
} // namespace AsdOps