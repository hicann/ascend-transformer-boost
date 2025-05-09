/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <numeric>
#include <mki/kernel_info.h>
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "mixkernels/multi_latent_attention/tiling/mla_tiling.h"
#include "mixkernels/multi_latent_attention/tiling/mla_tiling_dependency.h"
#include "mixkernels/utils/common.h"

namespace AtbOps {
using namespace Mki;
constexpr uint32_t TILINGMIN = 512;
class MLAKernel : public KernelBase {
public:
    explicit MLAKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = Utils::RoundUp((TILING_PARA_SIZE + TILING_HEAD_SIZE) * sizeof(uint32_t), TILINGMIN);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = MLATiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MLA),
            "paged attention: param type invalid", return false);
        auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
        auto batch = param.kvSeqLen.size();
        uint64_t taskNum = param.qSeqLen.data() == nullptr ? batch :
                            std::accumulate(param.qSeqLen.data(),
                                            param.qSeqLen.data() + batch, static_cast<int32_t>(0));
        uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
        uint64_t bufferSize =
            Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE_TP1 * (taskNum - 1) * sizeof(uint32_t) +
            TILING_PARA_SIZE_TP1 * blockDim * blockDim * sizeof(uint32_t) + blockDim * 2 * sizeof(uint32_t),
            TILINGMIN);
        return bufferSize;
    }
};

REG_KERNEL_BASE(MLAKernel);

} // namespace AtbOps

