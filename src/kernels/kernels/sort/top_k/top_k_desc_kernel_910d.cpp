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
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>
#include "asdops/params/sort.h"
#include "kernels/sort/tiling/sort_tiling_910d.h"

namespace AsdOps {
constexpr uint64_t TENSOR_TOPK_CONST_IDX = 1;
constexpr uint64_t TENSOR_TOPK_WORKSPACE_IDX = 6;

class TopKDescKernel910D : public KernelBase {
public:
    explicit TopKDescKernel910D(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Sort),
            "sort: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 2, "output num invalid", return false);
        return true;
    }

    /*uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Sort),
            "sort: param type invalid", return 0);
        const auto &param = AnyCast<OpParam::Sort>(launchParam.GetParam());
        size_t constTensorSize = Utils::GetConstTensorSize<int32_t>(param.num);
        return launchBufferSize_ + constTensorSize;
    }*/

    Status InitImpl(const LaunchParam &launchParam) override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Sort),
            "sort: param type invalid", return Status::FailStatus(1));
        auto status = TopKDescTiling910D(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        /*kernelInfo_.SetConstTensorOffset(launchBufferSize_);

        auto &param = AnyCast<OpParam::Sort>(launchParam.GetParam());
        if (PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B) {
            kernelInfo_.SetHwsyncIdx(0);
            kernelInfo_.AddConstTensorData<int32_t>(TENSOR_TOPK_CONST_IDX + 1, param.num);
            kernelInfo_.SetMemsetInfo(TENSOR_TOPK_WORKSPACE_IDX + 1,
                                      kernelInfo_.GetScratchSizes().at(2)); // 第2块workspace
        } else {
            kernelInfo_.AddConstTensorData<int32_t>(TENSOR_TOPK_CONST_IDX, param.num);
            kernelInfo_.SetMemsetInfo(TENSOR_TOPK_WORKSPACE_IDX,
                                      kernelInfo_.GetScratchSizes().at(2)); // 第2块workspace
        }*/
        return Status::OkStatus();
    }
};

// TopKDescF16Kernel910D
class TopKDescF16Kernel910D : public TopKDescKernel910D {
public:
    explicit TopKDescF16Kernel910D(const std::string &kernelName, const BinHandle *handle) noexcept
        : TopKDescKernel910D(kernelName, handle)
    {
    }
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return AsdOps::Status::OkStatus();
    }
};
REG_KERNEL_BASE(TopKDescF16Kernel910D);

// TopKDescBF16Kernel910D
class TopKDescBF16Kernel910D : public TopKDescKernel910D {
public:
    explicit TopKDescBF16Kernel910D(const std::string &kernelName, const BinHandle *handle) noexcept
        : TopKDescKernel910D(kernelName, handle)
    {
    }
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return AsdOps::Status::OkStatus();
    }
};
REG_KERNEL_BASE(TopKDescBF16Kernel910D);

// TopKDescF32Kernel910D
class TopKDescF32Kernel910D : public TopKDescKernel910D {
public:
    explicit TopKDescF32Kernel910D(const std::string &kernelName, const BinHandle *handle) noexcept
        : TopKDescKernel910D(kernelName, handle)
    {
    }
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override
    {
        return AsdOps::Status::OkStatus();
    }
};
REG_KERNEL_BASE(TopKDescF32Kernel910D);
} // namespace AsdOps