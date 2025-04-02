/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <climits>
#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "reshape_and_cache_tiling_dependency.h"

namespace AtbOps {
using namespace Mki;
bool CommonReshapeAndCacheTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto &kShape = launchParam.GetInTensor(DIM_0).desc.dims;
    uint32_t numTokens = static_cast<uint32_t>(kShape.at(DIM_0));
    uint32_t numHeads = static_cast<uint32_t>(kShape.at(DIM_1));
    uint32_t headSizeK = static_cast<uint32_t>(kShape.at(DIM_2));

    MKI_CHECK(numTokens > 0 && numTokens <= INT_MAX, "numTokens is invalid", return false);
    MKI_CHECK(numHeads > 0 && numHeads <= INT_MAX, "numHeads is invalid", return false);
    MKI_CHECK(headSizeK > 0 && headSizeK <= INT_MAX, "headSizeK is invalid", return false);

    ReshapeAndCacheTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::ReshapeAndCacheTilingData *>(kernelInfo.GetTilingHostAddr());
    
    tilingDataPtr->numTokens = numTokens;
    tilingDataPtr->numHeads = numHeads;
    tilingDataPtr->headSizeK = static_cast<uint32_t>(headSizeK);

    TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
    uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSize(inDtype));
    tilingDataPtr->typeByte = typeByte;

    return true;
}

Status ReshapeAndCacheTilingNd(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    ReshapeAndCacheTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::ReshapeAndCacheTilingData *>(kernelInfo.GetTilingHostAddr());

    auto &vShape = launchParam.GetInTensor(DIM_1).desc.dims;
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(DIM_2));
    MKI_CHECK(headSizeV > 0 && headSizeV <= INT_MAX, "headSizeV is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->headSizeV = static_cast<uint32_t>(headSizeV);

    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    blockDim = tilingDataPtr->numTokens < blockDim ? tilingDataPtr->numTokens : blockDim;
    bool isAlign = ((tilingDataPtr->numHeads * tilingDataPtr->headSizeK * tilingDataPtr->typeByte) % ALIGN == 0
        && (tilingDataPtr->numHeads * tilingDataPtr->headSizeV * tilingDataPtr->typeByte) % ALIGN == 0);

    uint32_t tilingKey = 0;
    if (tilingDataPtr->numTokens < SMALL_SHAPE && isAlign) {
        if (tilingDataPtr->headSizeK == headSizeV) {
            tilingKey = TILING_ID_DTYPE * tilingDataPtr->typeByte
                        + TILING_ID_MODE * static_cast<uint32_t>(OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND);
        } else {
            tilingKey = TILING_ID_DTYPE * tilingDataPtr->typeByte
                        + TILING_ID_MODE * static_cast<uint32_t>(OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND)
                        + TILING_ID_MLA;
        }
    } else {
        tilingKey = TILING_ID_DTYPE * tilingDataPtr->typeByte
                    + TILING_ID_MODE * static_cast<uint32_t>(OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND)
                    + TILING_ID_MLA_FULL;
    }

    MKI_LOG(INFO) << "numTokens: " << tilingDataPtr->numTokens << ", numHeads: " << tilingDataPtr->numHeads <<
        ", headSizeK: " << tilingDataPtr->headSizeK << ", headSizeV: " << headSizeV <<
        ", typeByte: " << tilingDataPtr->typeByte <<", tilingKey: " << tilingKey;
    kernelInfo.SetBlockDim(blockDim);
    kernelInfo.SetTilingId(tilingKey);

    return Status::OkStatus();
}

Status ReshapeAndCacheTilingNdSiso(KernelInfo &kernelInfo)
{
    ReshapeAndCacheTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::ReshapeAndCacheTilingData *>(kernelInfo.GetTilingHostAddr());
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    blockDim = tilingDataPtr->numTokens < blockDim ? tilingDataPtr->numTokens : blockDim;

    uint32_t tilingKey = 0;
    if (tilingDataPtr->numTokens < SMALL_SHAPE) {
        tilingKey = TILING_ID_DTYPE * tilingDataPtr->typeByte
                    + TILING_ID_MODE * static_cast<uint32_t>(OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO);
    } else {
        tilingKey = TILING_ID_DTYPE * tilingDataPtr->typeByte
                    + TILING_ID_MODE * static_cast<uint32_t>(OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO)
                    + TILING_ID_MLA_FULL;
    }

    MKI_LOG(INFO) << "numTokens: " << tilingDataPtr->numTokens << ", numHeads: " << tilingDataPtr->numHeads <<
        ", headSizeK: " << tilingDataPtr->headSizeK << ", headSizeV: " << tilingDataPtr->headSizeV <<
        ", typeByte: " << tilingDataPtr->typeByte << ", tilingKey: " << tilingKey;
    kernelInfo.SetBlockDim(blockDim);
    kernelInfo.SetTilingId(tilingKey);

    return Status::OkStatus();
}

Status ReshapeAndCacheTilingNz(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    ReshapeAndCacheTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::ReshapeAndCacheTilingData *>(kernelInfo.GetTilingHostAddr());

    auto &vShape = launchParam.GetInTensor(DIM_1).desc.dims;
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(DIM_2));
    MKI_CHECK(headSizeV > 0 && headSizeV <= INT_MAX, "headSizeV is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->headSizeV = static_cast<uint32_t>(headSizeV);

    auto kvCacheShape = launchParam.GetInTensor(DIM_2).desc.dims;
    uint32_t blockSize = static_cast<uint32_t>(kvCacheShape.at(DIM_2));
    MKI_CHECK(blockSize > 0 && blockSize <= INT_MAX, "blockSize is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->blockSize = blockSize;

    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    blockDim = tilingDataPtr->numTokens < blockDim ? tilingDataPtr->numTokens : blockDim;

    MKI_LOG(INFO) << "numTokens: " << tilingDataPtr->numTokens << ", numHeads: " << tilingDataPtr->numHeads <<
        ", headSizeK: " << tilingDataPtr->headSizeK << ", headSizeV: " << headSizeV <<
        ", blockSize: " << blockSize <<  ", typeByte: " << tilingDataPtr->typeByte;
    kernelInfo.SetBlockDim(blockDim);

    return Status::OkStatus();
}

Status ReshapeAndCacheTilingWinsAndRope(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    ReshapeAndCacheTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::ReshapeAndCacheTilingData *>(kernelInfo.GetTilingHostAddr());
    auto param = AnyCast<OpParam::ReshapeAndCache>(launchParam.GetParam());
    auto tensorSeqLen = launchParam.GetInTensor(DIM_6); // seqLen.shape = [batch]
    MKI_CHECK(tensorSeqLen.data != nullptr, "seqLen should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    int64_t numBatchs = tensorSeqLen.desc.dims.at(DIM_0);
    MKI_CHECK(numBatchs > 0 && numBatchs <= UINT32_MAX, "numBatchs is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t numTokens = tilingDataPtr->numHeads * static_cast<uint32_t>(numBatchs);

    MKI_CHECK(numTokens > 0 && numTokens <= INT_MAX, "numTokens is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    tilingDataPtr->numTokens = numTokens;
    tilingDataPtr->numBatchs = static_cast<uint32_t>(numBatchs);

    auto &vShape = launchParam.GetInTensor(DIM_1).desc.dims;
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(DIM_2));
    MKI_CHECK(headSizeV > 0 && headSizeV <= INT_MAX, "headSizeV is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->headSizeV = static_cast<uint32_t>(headSizeV);

    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE) {
        blockDim = numTokens * TASK_MULTIPLE <= blockDim ? numTokens * TASK_MULTIPLE : blockDim;
    } else {
        blockDim = numTokens < blockDim ? numTokens : blockDim;
    }

    uint32_t tilingKey = 0;
    if (param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS) {
        tilingKey = TILING_ID_DTYPE * tilingDataPtr->typeByte +
            TILING_ID_MODE * static_cast<uint32_t>(OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS);
    } else if (param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE) {
        tilingKey = TILING_ID_DTYPE * tilingDataPtr->typeByte +
            TILING_ID_MODE * static_cast<uint32_t>(OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE);
        TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
        if (inDtype == TENSOR_DTYPE_BF16) {
            tilingKey += TILING_ID_DTYPE;
        }
    }

    kernelInfo.SetBlockDim(blockDim);
    kernelInfo.SetTilingId(tilingKey);

    MKI_LOG(INFO) << "numTokens: " << tilingDataPtr->numTokens << ", numHeads: " << tilingDataPtr->numHeads <<
        ", headSizeK: " << tilingDataPtr->headSizeK << ", headSizeV: " << tilingDataPtr->headSizeV <<
        ", typeByte: " << tilingDataPtr->typeByte << ", numBatchs: " << tilingDataPtr->numBatchs <<
        ", tilingKey: " << tilingKey;

    return Status::OkStatus();
}

Status ReshapeAndCacheTilingOmniCompress(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    ReshapeAndCacheTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::ReshapeAndCacheTilingData *>(kernelInfo.GetTilingHostAddr());

    auto &vShape = launchParam.GetInTensor(DIM_1).desc.dims;
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(DIM_2));
    MKI_CHECK(headSizeV > 0 && headSizeV <= INT_MAX, "headSizeV is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->headSizeV = static_cast<uint32_t>(headSizeV);

    auto tensorSeqLen = launchParam.GetInTensor(DIM_6); // seqLen.shape = [batch]
    MKI_CHECK(tensorSeqLen.data != nullptr, "seqLen should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    int64_t numBatchs = tensorSeqLen.desc.dims.at(DIM_0);
    MKI_CHECK(numBatchs > 0 && numBatchs <= UINT32_MAX, "numBatchs is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    uint32_t numTokens = tilingDataPtr->numHeads * static_cast<uint32_t>(numBatchs);
    MKI_CHECK(numTokens > 0 && numTokens <= INT_MAX, "numTokens is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->numTokens = numTokens;
    tilingDataPtr->numBatchs = static_cast<uint32_t>(numBatchs);

    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    blockDim = numTokens * TASK_MULTIPLE <= blockDim ? numTokens * TASK_MULTIPLE : blockDim;

    uint32_t tilingKey = 0;
    tilingKey = TILING_ID_DTYPE * tilingDataPtr->typeByte +
            TILING_ID_MODE * static_cast<uint32_t>(OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_OMNI_COMPRESS);
    TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
    if (inDtype == TENSOR_DTYPE_BF16) {
            tilingKey += TILING_ID_DTYPE;
        }

    kernelInfo.SetBlockDim(blockDim);
    kernelInfo.SetTilingId(tilingKey);

    MKI_LOG(INFO) << "numTokens: " << tilingDataPtr->numTokens << ", numHeads: " << tilingDataPtr->numHeads <<
        ", headSizeK: " << tilingDataPtr->headSizeK << ", headSizeV: " << tilingDataPtr->headSizeV <<
        ", typeByte: " << tilingDataPtr->typeByte << ", numBatchs: " << tilingDataPtr->numBatchs <<
        ", tilingKey: " << tilingKey;
    
    return Status::OkStatus();
}

Status ReshapeAndCacheTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    ReshapeAndCacheTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::ReshapeAndCacheTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingHost should not be empty"));
    
    MKI_CHECK(CommonReshapeAndCacheTiling(launchParam, kernelInfo), "value is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    auto param = AnyCast<OpParam::ReshapeAndCache>(launchParam.GetParam());
    switch (param.type) {
        case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND:
            return ReshapeAndCacheTilingNd(launchParam, kernelInfo);
        case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO:
            return ReshapeAndCacheTilingNdSiso(kernelInfo);
        case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ:
            return ReshapeAndCacheTilingNz(launchParam, kernelInfo);
        case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS:
        case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE:
            return ReshapeAndCacheTilingWinsAndRope(launchParam, kernelInfo);
        case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_OMNI_COMPRESS:
            return ReshapeAndCacheTilingOmniCompress(launchParam, kernelInfo);
        default:
            return Status::FailStatus(ERROR_ATTR_INVALID_TYPE,
                "Failed to check reshape param, type of specificParam is invalid");
    }

    return Status::OkStatus();
}
} // namespace AtbOps
