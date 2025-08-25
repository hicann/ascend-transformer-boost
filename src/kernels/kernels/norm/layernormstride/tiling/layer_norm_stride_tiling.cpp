/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "layer_norm_stride_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "asdops/params/norm.h"
#include "tiling_data.h"
#include "kernels/norm/common/common_tiling.h"

namespace AsdOps {
constexpr uint32_t LAYER_NORM_TILING_KEY_BASE = 2000000000;
constexpr uint32_t LAYER_NORM_TILING_KEY_DTYPE = 100000000; // 0: fp16; 1: bf16
constexpr uint32_t LAYER_NORM_TILING_KEY_FAST = 10000000;   // 0: fast; 1:slice

constexpr int64_t FP16_DATA_USED = 5;
constexpr int64_t FP16_OTHER_USED = 6;
constexpr uint32_t SCALAR_USED = 50;
constexpr uint32_t NUM_TEMP_BUF = 32;
constexpr uint32_t MEAN_AND_VAR_SIZE = 64;

void LayerNormStridePrintLog(LayerNormStrideTilingData &tilingData)
{
    MKI_LOG(INFO) << " tilingDataPtr->numCore = " << tilingData.numCore
                  << " tilingDataPtr->numFirstDim = " << tilingData.numFirstDim
                  << " tilingDataPtr->lFirstdimPerCore = " << tilingData.lFirstdimPerCore
                  << " tilingDataPtr->nlFirstdimPerCore = " << tilingData.nlFirstdimPerCore
                  << " tilingDataPtr->epsStr = " << *reinterpret_cast<float *>(&tilingData.epsStr)
                  << " tilingDataPtr->aveStr = " << *reinterpret_cast<float *>(&tilingData.aveStr)
                  << " tilingDataPtr->numLastDim = " << tilingData.numLastDim
                  << " tilingDataPtr->firstDimPerTimes = " << tilingData.firstDimPerTimes
                  << " tilingDataPtr->normMode = " << tilingData.normMode
                  << " tilingDataPtr->zoomScale = " << tilingData.zoomScale
                  << " tilingDataPtr->sliceNum = " << tilingData.sliceNum
                  << " tilingDataPtr->sliceSize = " << tilingData.sliceSize
                  << " tilingDataPtr->tailSliceSize = " << tilingData.tailSliceSize;
}

Status GetTilingSliceInfo(LayerNormStrideTilingData &tilingData, uint32_t maxUbSize, uint32_t numCol,
                          KernelBufferInfoLayerNormStride kernelBufferInfo)
{
    uint32_t singleRowSizePerElem =
        kernelBufferInfo.fp32BufNum * sizeof(uint32_t) + kernelBufferInfo.fp16BufNum * sizeof(uint16_t);
    uint32_t multiRowSizePerElem = kernelBufferInfo.fp16BufNumForMulRow * sizeof(uint16_t);
    MKI_CHECK(numCol <= (UINT_MAX / (singleRowSizePerElem + multiRowSizePerElem)), "RowBufferSize invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "RowBufferSize invalid!"));
    uint32_t singleRowBufferSize = singleRowSizePerElem * numCol;
    uint32_t multiRowBufferSize = multiRowSizePerElem * numCol;

    if ((maxUbSize - MEAN_AND_VAR_SIZE) < (singleRowBufferSize + multiRowBufferSize)) {
        uint32_t oneRepeatElemCount = 256U / sizeof(uint16_t);
        uint32_t elemSize = Utils::RoundDown(
            (maxUbSize - MEAN_AND_VAR_SIZE) / (singleRowSizePerElem + multiRowSizePerElem), oneRepeatElemCount);
        tilingData.sliceNum = Utils::CeilDiv(numCol, elemSize);
        tilingData.sliceSize = elemSize;
        tilingData.tailSliceSize = numCol - (tilingData.sliceNum - 1) * elemSize;
    } else {
        tilingData.sliceNum = 1;
        tilingData.sliceSize = numCol;
        tilingData.tailSliceSize = numCol;
    }

    return Status::OkStatus();
}

void GetTilingBasicInfo(NormTilingDataPtrCon layerNormPtrCon, LayerNormStrideTilingData &tilingData, uint32_t numCol)
{
    tilingData.aveStr =  float(1.0 / numCol);
    tilingData.numLastDim = numCol;
    uint32_t numCore = layerNormPtrCon.numCore;
    tilingData.numCore = numCore;
    uint32_t numRow = layerNormPtrCon.numRow;
    tilingData.numFirstDim = numRow;
    uint32_t nlFirstdimPerCoreNum = layerNormPtrCon.nlFirstdimPerCoreNum;
    tilingData.nlFirstdimPerCore = nlFirstdimPerCoreNum;
    tilingData.lFirstdimPerCore = (numRow - nlFirstdimPerCoreNum * (numCore - 1));
}

Status LayerNormStrideTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const KernelBufferInfoLayerNormStride &kernelBufferInfo)
{
    LayerNormStrideTilingData *tilingDataPtr =
        reinterpret_cast<LayerNormStrideTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    auto attrs = AnyCast<OpParam::Norm>(launchParam.GetParam());
    tilingDataPtr->normMode = attrs.opsMode;
    tilingDataPtr->zoomScale = attrs.zoomScaleValue;
    tilingDataPtr->epsStr = *reinterpret_cast<uint32_t *>(&attrs.epsilon);
    NormTilingDataPtrCon layerNormPtrCon;
    Status ret =
        PostLayerNormPtrFunc<LayerNormStrideTilingData>(tilingDataPtr, layerNormPtrCon, launchParam, kernelInfo);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    uint32_t maxUbSize = layerNormPtrCon.maxUbSize; // maxUb
    uint32_t numCol = layerNormPtrCon.numCol;
    GetTilingBasicInfo(layerNormPtrCon, *tilingDataPtr, numCol);
    GetTilingSliceInfo(*tilingDataPtr, maxUbSize, numCol, kernelBufferInfo);
    if (tilingDataPtr->sliceNum == 1) {
        MKI_CHECK(layerNormPtrCon.numCol <= (UINT_MAX / (FP16_DATA_USED * layerNormPtrCon.nlFirstdimPerCoreNum)),
                  "totalMemNeed is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE));
        uint32_t totalMemNeed = static_cast<uint32_t>(FP16_DATA_USED) * layerNormPtrCon.nlFirstdimPerCoreNum *
                                layerNormPtrCon.numCol;
        MKI_CHECK(layerNormPtrCon.numCol <= (UINT_MAX / FP16_OTHER_USED),
                  "sumData is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE, "sumData is invalid!"));
        int32_t sumData = layerNormPtrCon.maxEleFp16 - NUM_TEMP_BUF -
                           static_cast<uint32_t>(FP16_OTHER_USED) * layerNormPtrCon.numCol - SCALAR_USED;
        ret = CheckSplit(tilingDataPtr, totalMemNeed, sumData, layerNormPtrCon);
        OP_TILING_CHECK_STATUS_RETURN(ret);
    } else { tilingDataPtr->firstDimPerTimes = 1; }
    uint64_t tilingKey = LAYER_NORM_TILING_KEY_BASE;
    tilingKey += launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? LAYER_NORM_TILING_KEY_DTYPE : 0;
    tilingKey += tilingDataPtr->sliceNum == 1 ? LAYER_NORM_TILING_KEY_FAST : 0;
    kernelInfo.SetTilingId(tilingKey); // 2000000000 + 100000000 + 10000000
    const auto& xStrides = launchParam.GetInTensor(0).desc.strides;
    const auto& xShapes = launchParam.GetInTensor(0).desc.dims;
    uint32_t dimNum = xStrides.size();
    bool isNCT = (xStrides.empty() || xStrides[dimNum - 2] == xShapes[dimNum - 1]) ? false : true;
    if (!isNCT) {
        tilingDataPtr->xDimNum = 0;
    } else {
        tilingDataPtr->xDimNum = dimNum;
        for (size_t i = 0; i < dimNum; ++ i) {
            tilingDataPtr->xStrides[i] = xStrides[i];
        }
        tilingDataPtr->xOffset = launchParam.GetInTensor(0).desc.offset;
    }
    LayerNormStridePrintLog(*tilingDataPtr);
    return Status::OkStatus();
}
} // namespace AsdOps