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
#include "atbops/params/params.h"
#include "mixops/utils/common.h"
#include "tiling/rms_norm_and_rope_and_reshape_and_cache_tiling.h"
#include "tiling/rms_norm_and_rope_and_reshape_and_cache_tiling_data.h"
 
namespace AtbOps {
using namespace Mki;
class RmsNormAndRopeAndReshapeAndCacheKernel : public KernelBase {
public:
    explicit RmsNormAndRopeAndReshapeAndCacheKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
    }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RmsNormAndRopeAndReshapeAndCache),
            "rms_norm_and_rope_and_reshape_and_cache: param type invalid", return false);
        return true;
    }
 
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RmsNormAndRopeAndReshapeAndCacheTilingData);
    }
 
    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = RmsNormAndRopeAndReshapeAndCacheTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        return Status::OkStatus();
    }
};
 
REG_KERNEL_BASE(RmsNormAndRopeAndReshapeAndCacheKernel);
} // namespace AtbOps