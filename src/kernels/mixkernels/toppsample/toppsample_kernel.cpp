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
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
#include "mixkernels/utils/common.h"
#include "tiling/toppsample_tiling.h"
#include "tiling/tiling_data.h"
#include "sink_common.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;

namespace AtbOps {
using namespace Mki;
class ToppsampleKernel : public KernelBase {
public:
    explicit ToppsampleKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
            "check outTensor count failed", return false);
        auto inTensor0 = launchParam.GetInTensor(0).desc;
        auto inTensor1 = launchParam.GetInTensor(1).desc;

        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Toppsample),
            "check param type failed!", return false);
        MKI_CHECK(inTensor0.dims.size() == 2, "check inTensor dims failed", return false);
        MKI_CHECK(inTensor1.Numel() == 1 || inTensor1.Numel() == inTensor0.dims[0],
            "batch size is wrong, check inTensor1 or first dim of inTensor0", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(ToppsampleTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto geTiling = optiling::CallGeTiling("TopPSample", *GetBinHandle(), launchParam,
                                      AsdOps::GetMkiSpecificAttr<OpParam::Toppsample>, kernelInfo_);
        kernelInfo.SetMemsetInfo(TENSOR_INPUT_NUM + TENSOR_OUTPUT_NUM, sysWorkspaceSize);
        return geTiling;  
    }
};
REG_KERNEL_BASE(ToppsampleKernel);
} // namespace AtbOps