/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_post_norm_quant_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "asdops/params/norm.h"
#include "kernels/norm/common/common_tiling.h"

static constexpr uint32_t TBUF_NUM = 21;  // double buffer: 2*fp16(x)+2*fp16(resIn)+fp16(g)+int8(y)+fp16(reOut)+2*fp32
static constexpr uint32_t BYTE_32 = 32;

namespace AsdOps {
constexpr uint32_t RMSPOSTNORMQUANT_TILING_KEY_BASE = 100;
constexpr uint32_t RMSPOSTNORMQUANT_TILING_KEY_DTYPE = 10;    // 0: fp16; 1: bf16
constexpr uint32_t RMSPOSTNORMQUANT_TILING_KEY_WITH_SLICE = 1; // 0: no slice; 1: use slice

Status RmsPostNormQuantTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    RmsNormQuantCommonTilingData *tilingDataPtr =
        reinterpret_cast<RmsNormQuantCommonTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    Status ret = RmsNormQuantTilingGen<RmsNormQuantCommonTilingData>(kernelInfo, tilingDataPtr, launchParam);
    OP_TILING_CHECK_STATUS_RETURN(ret);

    uint32_t ubSize = PlatformInfo::Instance().GetUbSize();   // A2: 192K  310P:256K
    uint32_t maxSliceSize = (ubSize / TBUF_NUM) / BYTE_32 * BYTE_32;
    tilingDataPtr->sliceSize = maxSliceSize;
    MKI_CHECK(tilingDataPtr->numCol <= UINT_MAX - tilingDataPtr->sliceSize, "numCol + sliceSize is invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "numCol + sliceSize is invalid!"));
    if (tilingDataPtr->numCol <= tilingDataPtr->sliceSize) {
        tilingDataPtr->sliceSize = tilingDataPtr->numCol;
    } else {
        tilingDataPtr->sliceNum = (tilingDataPtr->numCol + tilingDataPtr->sliceSize - 1) / tilingDataPtr->sliceSize;
        tilingDataPtr->tailSliceSize = tilingDataPtr->numCol - (tilingDataPtr->sliceNum - 1) * maxSliceSize;
    }
    
    uint64_t tilingKey = RMSPOSTNORMQUANT_TILING_KEY_BASE;
    tilingKey += launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? RMSPOSTNORMQUANT_TILING_KEY_DTYPE : 0;
    tilingKey += tilingDataPtr->numCol > tilingDataPtr->sliceSize ? RMSPOSTNORMQUANT_TILING_KEY_WITH_SLICE : 0;
    kernelInfo.SetTilingId(tilingKey);

    MKI_LOG(INFO) << "RmsPostNormQuant TilingData:"
                  << " numCore " << tilingDataPtr->numCore
                  << " numCol " << tilingDataPtr->numCol
                  << " numRow " << tilingDataPtr->numRow
                  << " avgFactor " << *reinterpret_cast<float *>(&tilingDataPtr->avgFactor)
                  << " epsilon " << *reinterpret_cast<float *>(&tilingDataPtr->epsilon)
                  << " quantMin " << tilingDataPtr->quantMin
                  << " slicesize " << tilingDataPtr->sliceSize
                  << " slicenum " << tilingDataPtr->sliceNum
                  << " tail " << tilingDataPtr->tailSliceSize
                  << " tilingKey " << kernelInfo.GetTilingId();
    return Status::OkStatus();
}
} // namespace AsdOps