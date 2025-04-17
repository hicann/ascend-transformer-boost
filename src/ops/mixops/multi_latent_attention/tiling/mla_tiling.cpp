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
#include <algorithm>
#include "mki/utils/assert/assert.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/platform/platform_info.h"
#include "mla_tiling.h"

namespace AtbOps {

const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM16 = 16;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const float SPLITKV_RATION = 0.8;

Status GetMLANdInfo(const LaunchParam &launchParam, MLAInfo &mmInfo,
                    OpParam::MLA &param)
{
    auto kcacheShape = launchParam.GetInTensor(DIM_2).desc.dims;
    auto KDims = kcacheShape.size();
    auto tableShape = launchParam.GetInTensor(DIM_4).desc.dims;
    mmInfo.kNz = (kcacheShape.at(KDims - 1) == NUM16 || kcacheShape.at(KDims - 1) == NUM32) ? 1 : 0;
    if (mmInfo.kNz) {
        mmInfo.embeddingSize = static_cast<int32_t>(kcacheShape.at(DIM_3)) *
                            static_cast<int32_t>(kcacheShape.at(DIM_1));
        mmInfo.blockSize = static_cast<int32_t>(kcacheShape.at(DIM_2));
    } else {
        mmInfo.embeddingSize = static_cast<int32_t>(kcacheShape.at(DIM_3));
        mmInfo.blockSize = static_cast<int32_t>(kcacheShape.at(DIM_1));
    }
    mmInfo.numTokens = static_cast<int32_t>(param.kvSeqLen.size());
    mmInfo.numBlocks = static_cast<int32_t>(kcacheShape.at(DIM_0));
    mmInfo.maxNumBlocksPerQuery = static_cast<int32_t>(tableShape.at(DIM_1));
    mmInfo.tor = param.tor;
    mmInfo.kvSeqLen = param.kvSeqLen.data();
    mmInfo.qSeqLen = param.qSeqLen.data();
    param.kvHead = param.kvHead <= 0 ? param.headSize : param.kvHead;
    mmInfo.batch = static_cast<int32_t>(param.kvSeqLen.size());
    mmInfo.kvHeads = param.kvHead;
    mmInfo.numHeads = static_cast<int32_t>(param.headSize);
    mmInfo.maskType = static_cast<int32_t>(param.maskType);
    mmInfo.mtpTp1Flag = (mmInfo.numHeads == M_LIMIT); // quant not support
    if (mmInfo.mtpTp1Flag) {
        mmInfo.maskType = 0;
    }
    if (static_cast<int32_t>(mmInfo.type) >= NUM2) {
        mmInfo.maskType = 0;
    }
    if (mmInfo.qSeqLen != nullptr) {
        mmInfo.totalTaskNum = std::accumulate(mmInfo.qSeqLen, mmInfo.qSeqLen + mmInfo.batch, static_cast<int32_t>(0));
    } else {
        mmInfo.totalTaskNum = mmInfo.batch;
    }
    return Status::OkStatus();
}

Status GetMLAInfo(const LaunchParam &launchParam, MLAInfo &mmInfo, OpParam::MLA &param)
{
    OP_TILING_CHECK_STATUS_RETURN(GetMLANdInfo(launchParam, mmInfo, param));
    return Status::OkStatus();
}

Status GetTilingKeyTypeBase(MLAInfo &mmInfo, const Tensor &qTensor, const Tensor &qRopeTensor)
{
    if (qTensor.desc.dtype == TENSOR_DTYPE_BF16) {
        mmInfo.type = TilingKeyType::TILING_BF16_DATA;
    } else if (qTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        mmInfo.type = TilingKeyType::TILING_HALF_DATA;
    } else if (qRopeTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        mmInfo.type = TilingKeyType::TILING_INT8_HALF_DATA;
    } else {
        mmInfo.type = TilingKeyType::TILING_INT8_BF16_DATA;
    }
    return Status::OkStatus();
}

Status GenTilingKey(MLAInfo &mmInfo, KernelInfo &kernelInfo, OpParam::MLA &param)
{
    uint32_t dataType = static_cast<int32_t>(mmInfo.type);
    uint32_t tilingKey = dataType + (mmInfo.kNz << NUM4) + (mmInfo.mtpTp1Flag << NUM2) + (param.isRing << NUM5);
    kernelInfo.SetTilingId(tilingKey);
    MKI_LOG(INFO) << "TILING KEY IS = " << tilingKey;
    return Status::OkStatus();
}

Status MLATiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
    auto qTensor = launchParam.GetInTensor(DIM_0);
    auto qRopeTensor = launchParam.GetInTensor(DIM_1);
    
    MLAInfo mmInfo = {0};
    GetTilingKeyTypeBase(mmInfo, qTensor, qRopeTensor);
    Status ret1  = GetMLAInfo(launchParam, mmInfo, param);
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    uint32_t *tilingParam = reinterpret_cast<uint32_t *>(kernelInfo.GetTilingHostAddr());
    uint64_t tilingSize = kernelInfo.GetTilingSize();
    Status ret = GetMLATilingParam(launchParam, mmInfo, blockDim, tilingParam, tilingSize);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    uint32_t dataLenHalf = sizeof(uint16_t);
    uint32_t dataLenFloat = sizeof(float);
    uint32_t dataLenInt = sizeof(int32_t);
    uint64_t basicWorkSpaceHalf = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenHalf;
    uint64_t basicWorkSpaceFloat = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenFloat;
    uint64_t basicWorkSpaceInt8 = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenInt;
    bool isQuant = (static_cast<int32_t>(mmInfo.type) < NUM2) ? 0 : 1;
    if (isQuant) {
        uint64_t sWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        uint64_t pWorkSpaceSize = basicWorkSpaceInt8;
        uint64_t oTempWorkSpcaceSize = basicWorkSpaceInt8 * 2;
        kernelInfo.GetScratchSizes() = {sWorkSpaceSize, sWorkSpaceSize, pWorkSpaceSize,
                                        oTempWorkSpcaceSize, basicWorkSpaceFloat};
    } else {
        uint64_t sWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        uint64_t pWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceHalf * 4 : basicWorkSpaceHalf * 2;
        uint64_t oTempWorkSpcaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        uint64_t goWorkSpcaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        kernelInfo.GetScratchSizes() = {sWorkSpaceSize, NUM512, pWorkSpaceSize, oTempWorkSpcaceSize,
                                        goWorkSpcaceSize};
    }
    Status ret2 = GenTilingKey(mmInfo, kernelInfo, param);
    OP_TILING_CHECK_STATUS_RETURN(ret2);
    kernelInfo.SetBlockDim(blockDim);
    MKI_LOG(INFO) << "launchBufferSize = " << tilingSize << " block dim = " << blockDim;
    return Status::OkStatus();
}

} // namespace AtbOps
