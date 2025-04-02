/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cast_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/elewise.h"
#include "mixops/elewise/cast/tiling/tiling_data.h"

namespace AtbOps {
using namespace Mki;
constexpr uint32_t CAST_MAX_AVAILABLE_UB = 32760;
constexpr uint32_t CAST_MAX_AVAILABLE_UB_910B = 24568;
constexpr uint32_t CAST_MAX_AVAILABLE_UB_DB = 16376;
constexpr uint32_t CAST_MAX_AVAILABLE_UB_DB_910B = 12280;
constexpr uint32_t CAST_MAX_SHAPE_SIZE = 12288 * 2048 * 32;

const std::unordered_map<uint32_t, uint32_t> SPLIT_FACTORS{
    {1, 32767},
    {2, 32767},
    {4, 16383},
    {8, 8191},
};

void DoBlockTiling(uint32_t outShapeSize, uint32_t &blockFactor, uint32_t &blockNum, uint32_t &blockDim)
{
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    uint32_t alignedBlockFactor = 128; // CAST_UBFACTORALIGN

    // 每个核处理的数据数量
    blockFactor = Utils::CeilDiv(outShapeSize, coreNum);
    MKI_LOG(INFO) << "blockFactor1: " << blockFactor;
    // 进行128对齐
    blockFactor = ((blockFactor + alignedBlockFactor - 1) / alignedBlockFactor) * alignedBlockFactor;
    MKI_LOG(INFO) << "blockFactor2 align: " << blockFactor;
    // 每个核的数量与一个每个核处理能力比较，取大值得到每个核处理的数据数量
    blockFactor = std::min(std::max(static_cast<const long int>(blockFactor), static_cast<const long int>(32)),
                           static_cast<const long int>(outShapeSize)); // 32 表示CAST_MAXOUT_DTYPENUM， int8 256B

    blockNum = blockFactor;
    MKI_LOG(INFO) << "blockFactor3: " << blockFactor;
    // 最终需要的工作数量
    blockDim = Utils::CeilDiv(outShapeSize, blockFactor);
}

void DoUbTiling(uint32_t factor, uint32_t blockFactor, uint32_t &ubFactor, bool needDoubleBuffer)
{
    ubFactor = blockFactor;

    PlatformType platformType = PlatformInfo::Instance().GetPlatformType();
    uint32_t maxAvailableUb =
        platformType == PlatformType::ASCEND_910B ? CAST_MAX_AVAILABLE_UB_910B : CAST_MAX_AVAILABLE_UB_DB_910B;
    uint32_t maxAvailableUbDb =
        platformType == PlatformType::ASCEND_910B ? CAST_MAX_AVAILABLE_UB : CAST_MAX_AVAILABLE_UB_DB;

    uint32_t limit = std::min(maxAvailableUb, factor); // 一轮ub上限
    if (needDoubleBuffer) {
        limit = std::min(maxAvailableUbDb, factor);
    }
    if (limit < ubFactor) {
        uint32_t alignedUbFactor = 128;
        uint32_t ubForNum = (ubFactor + limit - 1) / limit;           // ub要算几轮
        uint32_t adjustFactor = (ubFactor + ubForNum - 1) / ubForNum; // 每轮ub计算数量
        uint32_t alignFactor = (adjustFactor + alignedUbFactor - 1) / alignedUbFactor;
        ubFactor = alignFactor * alignedUbFactor;                          // 每轮ub计算数量向上对齐128
        if (ubFactor > limit) {                                            // 若对齐后超过了上限
            ubFactor = (adjustFactor / alignedUbFactor) * alignedUbFactor; // 每轮ub计算数量向下对齐128
        }
    }
}

bool GetTransKey(const LaunchParam &launchParam, uint32_t &dataTransKey)
{
    TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
    TensorDType outDtype = launchParam.GetOutTensor(0).desc.dtype;
    if (inDtype == TENSOR_DTYPE_FLOAT16 && outDtype == TENSOR_DTYPE_FLOAT) {
        dataTransKey = TransKey::HALF_TO_FLOAT;
    } else if (inDtype == TENSOR_DTYPE_FLOAT && outDtype == TENSOR_DTYPE_FLOAT16) {
        dataTransKey = TransKey::FLOAT_TO_HALF;
    } else if (inDtype == TENSOR_DTYPE_FLOAT && outDtype == TENSOR_DTYPE_INT32) {
        dataTransKey = TransKey::FLOAT_TO_INT32;
    } else if (inDtype == TENSOR_DTYPE_FLOAT16 && outDtype == TENSOR_DTYPE_INT32) {
        dataTransKey = TransKey::HALF_TO_INT32;
    } else if (inDtype == TENSOR_DTYPE_INT64 && outDtype == TENSOR_DTYPE_INT32) {
        dataTransKey = TransKey::INT64_TO_INT32;
    } else if (inDtype == TENSOR_DTYPE_INT32 && outDtype == TENSOR_DTYPE_INT64) {
        dataTransKey = TransKey::INT32_TO_INT64;
    } else if (inDtype == TENSOR_DTYPE_INT32 && outDtype == TENSOR_DTYPE_FLOAT16) {
        dataTransKey = TransKey::INT32_TO_HALF;
    } else if (inDtype == TENSOR_DTYPE_FLOAT && outDtype == TENSOR_DTYPE_BF16) {
        dataTransKey = TransKey::FLOAT_TO_BF16;
    } else if (inDtype == TENSOR_DTYPE_BF16 && outDtype == TENSOR_DTYPE_FLOAT) {
        dataTransKey = TransKey::BF16_TO_FLOAT;
    } else {
        return false;
    }
    return true;
}

Status CastCommonTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t maxShapeSize = static_cast<uint32_t>(launchParam.GetOutTensor(0).Numel());

    const uint32_t multiCoreThreshold = 32 * 2; // 32 表示 CAST_MAXOUT_DTYPENUM, 2 表示 DOUBLE_BUFFER_SIZE
    MKI_LOG(INFO) << "multiCoreThreshold: " << multiCoreThreshold;
    bool needMultiCore = maxShapeSize < multiCoreThreshold ? false : true;
    auto tilingdata = reinterpret_cast<AtbOps::CastTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingdata != nullptr, "tilingdata should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingdata should not be empty"));
    tilingdata->numTotal = maxShapeSize;
    tilingdata->blockNum = 0;
    tilingdata->blockTail = 0;
    tilingdata->ubFactor = 0;
    bool ret = GetTransKey(launchParam, tilingdata->dataTransKey);
    MKI_CHECK(ret, "Unsupported combination of inDtype and outDtype!",
        return Status::FailStatus(ERROR_INVALID_VALUE, "Unsupported combination of inDtype and outDtype!"));

    uint32_t blockDims = 0;

    PlatformType platformType = PlatformInfo::Instance().GetPlatformType();
    uint32_t maxAvailableUb =
        platformType == PlatformType::ASCEND_910B ? CAST_MAX_AVAILABLE_UB_910B : CAST_MAX_AVAILABLE_UB_DB_910B;

    if (needMultiCore) {
        uint32_t blockFactor = 0; // 每个核处理的数据数量
        bool needDoubleBuffer = false;
        DoBlockTiling(maxShapeSize, blockFactor, tilingdata->blockNum, blockDims);
        uint32_t factor = SPLIT_FACTORS.at(8); // CAST_MAXDTYPE, int64 hast  8B
        needDoubleBuffer = blockFactor > std::min(maxAvailableUb, factor) ? true : false;
        DoUbTiling(factor, blockFactor, tilingdata->ubFactor, needDoubleBuffer);
    } else {
        blockDims = 1;
        tilingdata->ubFactor = maxShapeSize;
        tilingdata->blockNum = maxShapeSize;
    }
    kernelInfo.SetBlockDim(blockDims);
    MKI_CHECK(tilingdata->ubFactor != 0, "tilingdata ubFactor is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingdata ubFactor is invalid"));
    MKI_LOG(INFO) << "launchBufferSize = " << kernelInfo.GetTilingSize();
    MKI_LOG(INFO) << "blockNum, blockTail, ubFactor= " << tilingdata->blockNum << " " << tilingdata->blockTail << " "
                  << tilingdata->ubFactor;
    MKI_LOG(INFO) << "dataTransKey = " << tilingdata->dataTransKey;
    MKI_LOG(INFO) << "tilingKey is " << tilingdata->dataTransKey;
    kernelInfo.SetTilingId(tilingdata->dataTransKey);
    return Status::OkStatus();
}

} // namespace AtbOps