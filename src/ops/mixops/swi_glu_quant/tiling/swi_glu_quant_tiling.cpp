/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include <iostream>
#include "atbops/params/params.h"
#include "tiling_data.h"
#include "swi_glu_quant_tiling_utils.h"
#include "swi_glu_quant_tiling.h"

static constexpr uint32_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
static constexpr uint32_t BLOCK_SIZE = 32;
static constexpr uint32_t L2_CACHE_LINE_SIZE = 512; // pack unit in cache 512B
static constexpr uint32_t SIZE_OF_FLOAT16 = 2;

static constexpr uint32_t SINGLE_UB_SIZE = 25;

static constexpr int TILING_KEY_BF16_QUANT_MODE = 206; // Tiling key for BF16 quantization mode
static constexpr int TILING_KEY_FP16_QUANT_MODE = 106; // Tiling key for FP16 quantization mode
static constexpr int TILING_KEY_FP32_QUANT_MODE = 306; // Tiling key for FP32 quantization mode

namespace AtbOps  {
using namespace Mki;

void PrintSwiQuantTiling(SwiGluQuantTilingData *tilingData)
{
    MKI_LOG(INFO) << "SwiGlu Tiling Data:"
                  << " groupLen " << tilingData->groupLen
                  << " rowLen " << tilingData->rowLen
                  << " colLen " << tilingData->colLen
                  << " rowLenPerHeadCore " << tilingData->rowLenPerHeadCore
                  << " rowLenPerTailCore " << tilingData->rowLenPerTailCore
                  << " basicRowLenHeadCore " << tilingData->basicRowLenHeadCore
                  << " basicRowLenTailCore " << tilingData->basicRowLenTailCore
                  << " basicColLen  " << tilingData->basicColLen
                  << " headCoreNum " << tilingData->headCoreNum
                  << " realCoreNum  " << tilingData->realCoreNum;
}

Status SwiGluQuantTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    MKI_LOG(INFO) << "----- [ Enter SwiGluForwardTiling ] -----";
    uint32_t totalCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    uint32_t ubSize = PlatformInfo::Instance().GetUbSize();
    SwiGluQuantTilingData *tilingData = reinterpret_cast<SwiGluQuantTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingData != nullptr, "tilingData should not be empty", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingData->totalCore = totalCore;
    MKI_LOG(INFO) << "swigluquant compileInfo->totalCore is : " << tilingData->totalCore;
    MKI_LOG(INFO) << "swigluquant totalCore is : " << totalCore;
    tilingData->ubSize = ubSize;
    uint64_t tilingKey = 0;
    auto xDtype = launchParam.GetInTensor(0).desc.dtype;
    if ((xDtype == TENSOR_DTYPE_FLOAT16)) {
        tilingKey = TILING_KEY_FP16_QUANT_MODE;
    }
    if ((xDtype == TENSOR_DTYPE_BF16)) {
        tilingKey = TILING_KEY_BF16_QUANT_MODE;
    }
    tilingData->groupLen = tilingData->groupLength;
    tilingData->inputDataByte = TENSOR_DTYPE_FLOAT16;
    tilingData->dataNumSingleUb = tilingData->ubSize / SINGLE_UB_SIZE;
    tilingData->blockNum = BLOCK_SIZE / SIZE_OF_FLOAT16;
    tilingData->cacheLineLen = L2_CACHE_LINE_SIZE / SIZE_OF_FLOAT16;
    const Mki::SVector<int64_t> &xShape = launchParam.GetInTensor(0).desc.dims;
    if (!SetTotalShape(xShape, tilingData)) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    CalTilingData(tilingData);
    SetTilingData(tilingData);
    kernelInfo.SetBlockDim(tilingData->coreNumUsed);
    MKI_LOG(INFO) << "swigluquant tilingKey is : " << tilingKey;
    kernelInfo.SetTilingId(tilingKey);
    PrintSwiQuantTiling(tilingData);
    return Status::OkStatus();
}
}