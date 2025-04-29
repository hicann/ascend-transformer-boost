/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "blockcopy_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling_data.h"

static constexpr uint64_t BLOCKCOPY_WKSP_TENSOR_IDX = 7;
static constexpr uint64_t BLOCKCOPY_SYNC_WORKSPACE_TENSOR_IDX = 8;
static constexpr int32_t BYTES_32 = 32;
namespace AtbOps {
using namespace Mki;
Status BlockCopyTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto &kShape = launchParam.GetInTensor(DIM_0).desc.dims;
    auto &vShape = launchParam.GetInTensor(DIM_1).desc.dims;
    auto &srcShape = launchParam.GetInTensor(DIM_2).desc.dims;
    auto &dstShape = launchParam.GetInTensor(DIM_3).desc.dims;
    auto &cumShape = launchParam.GetInTensor(DIM_4).desc.dims;
    uint32_t blockCount = static_cast<uint32_t>(kShape.at(DIM_0));
    uint32_t blockSize = static_cast<uint32_t>(kShape.at(DIM_1));
    uint32_t numHead = static_cast<uint32_t>(kShape.at(DIM_2));
    uint32_t headSizeK = static_cast<uint32_t>(kShape.at(DIM_3));
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(DIM_3));
    uint32_t sourceCount = static_cast<uint32_t>(srcShape.at(DIM_0));
    uint32_t destinationCount = static_cast<uint32_t>(dstShape.at(DIM_0));
    uint32_t cumSumCount = static_cast<uint32_t>(cumShape.at(DIM_0));

    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::BlockCopy), "OpParam is invalid",
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

    MKI_CHECK(sourceCount > 0, "sourceCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(destinationCount > 0, "destinationCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(blockCount > 0, "blockCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(blockSize > 0, "blockSize is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(numHead > 0, "numHead is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(headSizeK > 0, "headSizeK is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(headSizeV > 0, "headSizeV is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(sourceCount == cumSumCount, "src&cum's shape must be same ",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    BlockCopyTilingData *tilingDataPtr = reinterpret_cast<AtbOps::BlockCopyTilingData*>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingHost should not be empty"));

    tilingDataPtr->blockCount = blockCount;
    tilingDataPtr->blockSize = blockSize;
    tilingDataPtr->numHead = numHead;
    tilingDataPtr->headSizeK = headSizeK;
    tilingDataPtr->headSizeV = headSizeV;
    tilingDataPtr->sourceCount = sourceCount;
    tilingDataPtr->destinationCount = destinationCount;

    TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
    uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSize(inDtype));
    tilingDataPtr->typeByte = typeByte;

    uint32_t tilingKey = TILING_DTYPE_IDX * typeByte;

    uint64_t maxCore = static_cast<uint64_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    uint32_t actualCore = destinationCount < maxCore ? destinationCount : maxCore;

    tilingDataPtr->blockDim = actualCore;

    tilingDataPtr->perCoreCopyCount = destinationCount / actualCore;
    tilingDataPtr->tailCoreCopyCount = destinationCount % actualCore;

    MKI_LOG(INFO) << "blockCount: " << blockCount << ", sourceCount: " << sourceCount
                  << ", destinationCount: " << destinationCount;
    kernelInfo.SetBlockDim(actualCore);
    kernelInfo.SetTilingId(tilingKey);
    return Status::OkStatus();
}
} // namespace AtbOps
