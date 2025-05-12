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
const int32_t NUM6 = 6;
const int32_t NUM8 = 8;
const int32_t NUM16 = 16;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const float SPLITKV_SEQLEN = 2048;

int32_t CalcSplitNum(MLAInfo &mmInfo, int32_t blockDim, int32_t minKVSeqlen, int32_t maxKVSeqlen, int32_t blockSize)
{
    if (blockDim - mmInfo.flashDecodingTaskNum <= NUM4 || mmInfo.quantFlag) {
        return NUM1;
    }
    if (blockSize == 0 || blockDim == 0) {
        return NUM1;
    }
    int32_t maxKVBlocks = (maxKVSeqlen + blockSize - 1) / blockSize;
    for (int32_t splitNum = 2; splitNum <= NUM6; splitNum++) {
        if ((mmInfo.flashDecodingTaskNum * splitNum) % blockDim != 0) {
            continue;
        }
        int32_t repeatTimesPerBlock = mmInfo.flashDecodingTaskNum * splitNum / blockDim;
        if (minKVSeqlen / splitNum >= blockSize &&
            repeatTimesPerBlock + NUM6 <= maxKVBlocks - (maxKVBlocks + splitNum - 1) / splitNum * repeatTimesPerBlock) {
            return splitNum;
        } else {
            return NUM1;
        }
    }
    return NUM1;
}

Status GetFlashDecodingInfo(MLAInfo &mmInfo, OpParam::MLA &param, uint32_t blockDim)
{
    if (blockDim <= 0) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    mmInfo.tailBatch = mmInfo.batch % blockDim;
    mmInfo.tailTaskNum = mmInfo.totalTaskNum % blockDim;
    mmInfo.flashDecodingTaskNum = mmInfo.quantFlag ? mmInfo.tailTaskNum : mmInfo.tailBatch;
    auto minKVSeqlen = std::min_element(param.kvSeqLen.begin(), param.kvSeqLen.end());
    auto maxKVSeqlen = std::max_element(param.kvSeqLen.begin(), param.kvSeqLen.end());
    auto minQSeqlen = mmInfo.qSeqLen != nullptr ? *std::min_element(param.qSeqLen.begin(), param.qSeqLen.end()) : 1;
    auto maxQSeqlen = mmInfo.qSeqLen != nullptr ? *std::max_element(param.qSeqLen.begin(), param.qSeqLen.end()) : 1;
    mmInfo.flashDecoding = mmInfo.flashDecodingTaskNum != 0 && param.isRing == 0 &&
                           *minKVSeqlen >= SPLITKV_SEQLEN &&
                           ((minQSeqlen == NUM2 && maxQSeqlen == NUM2) ||
                           (minQSeqlen == 1 && maxQSeqlen == 1));
    if (!mmInfo.flashDecoding) {
        return Status::OkStatus();
    }
    mmInfo.splitKVNum = blockDim / mmInfo.flashDecodingTaskNum > 1 ?  blockDim / mmInfo.flashDecodingTaskNum :
                        CalcSplitNum(mmInfo, blockDim, *minKVSeqlen, *maxKVSeqlen, mmInfo.blockSize);
    mmInfo.flashDecoding = mmInfo.splitKVNum == 1 ? false : true;
    if (mmInfo.flashDecoding) {
        for (int32_t batchIdx = 0; batchIdx < mmInfo.batch; batchIdx++) {
            mmInfo.batchList.push_back(BatchNode(batchIdx, *(mmInfo.kvSeqLen + batchIdx)));
        }
        std::sort(mmInfo.batchList.begin(), mmInfo.batchList.end());
        int32_t taskNum = mmInfo.quantFlag ? mmInfo.totalTaskNum : mmInfo.batch;
        mmInfo.normalTaskNum = taskNum / blockDim * blockDim;
    }
    MKI_LOG(INFO) << "mmInfo.flashDecoding is = " << mmInfo.flashDecoding;
    return Status::OkStatus();
}

Status GetMLANdInfo(const LaunchParam &launchParam, MLAInfo &mmInfo,
                    OpParam::MLA &param, uint32_t blockDim)
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
    mmInfo.quantFlag = (static_cast<int32_t>(mmInfo.type) < NUM2) ? 0 : 1;
    mmInfo.mtpTp1Flag = (mmInfo.numHeads == M_LIMIT);
    if (mmInfo.mtpTp1Flag || static_cast<int32_t>(mmInfo.type) >= NUM2) {
        mmInfo.maskType = 0;
    }
    mmInfo.totalTaskNum = mmInfo.qSeqLen != nullptr ?
                          std::accumulate(mmInfo.qSeqLen, mmInfo.qSeqLen + mmInfo.batch, static_cast<int32_t>(0)) :
                          mmInfo.batch;
    if (mmInfo.mtpTp1Flag) {
        OP_TILING_CHECK_STATUS_RETURN(GetFlashDecodingInfo(mmInfo, param, blockDim));
    }
    return Status::OkStatus();
}

Status GetMLAInfo(const LaunchParam &launchParam, MLAInfo &mmInfo, OpParam::MLA &param, uint32_t blockDim)
{
    OP_TILING_CHECK_STATUS_RETURN(GetMLANdInfo(launchParam, mmInfo, param, blockDim));
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
    uint32_t tilingKey = dataType + (mmInfo.kNz << NUM4) + (mmInfo.mtpTp1Flag << NUM2) +
                         (param.isRing << NUM5) + (mmInfo.flashDecoding << NUM6);
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
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    Status ret1 = GetMLAInfo(launchParam, mmInfo, param, blockDim);
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
    uint64_t oCoreWorkSpaceSize = mmInfo.flashDecoding && mmInfo.mtpTp1Flag ?
        WORKSPACE_BLOCK_SIZE_DB * mmInfo.flashDecodingTaskNum * mmInfo.splitKVNum * dataLenFloat * 2 : 0;
    uint64_t lWorkSpaceSize = mmInfo.flashDecoding && mmInfo.mtpTp1Flag ?
        mmInfo.numHeads * mmInfo.flashDecodingTaskNum * mmInfo.splitKVNum * dataLenFloat * 2 * 8 : 0;
    if (mmInfo.quantFlag) {
        uint64_t sWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        uint64_t pWorkSpaceSize = basicWorkSpaceInt8;
        uint64_t oTempWorkSpaceSize = basicWorkSpaceInt8 * 2;
        kernelInfo.GetScratchSizes() = {sWorkSpaceSize, sWorkSpaceSize, pWorkSpaceSize,
                                        oTempWorkSpaceSize, basicWorkSpaceFloat, oCoreWorkSpaceSize, lWorkSpaceSize};
    } else {
        uint64_t sWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        uint64_t pWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceHalf * 4 : basicWorkSpaceHalf * 2;
        uint64_t oTempWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        uint64_t goWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        kernelInfo.GetScratchSizes() = {sWorkSpaceSize, NUM512, pWorkSpaceSize, oTempWorkSpaceSize,
                                        goWorkSpaceSize, oCoreWorkSpaceSize, lWorkSpaceSize};
    }
    Status ret2 = GenTilingKey(mmInfo, kernelInfo, param);
    OP_TILING_CHECK_STATUS_RETURN(ret2);
    kernelInfo.SetBlockDim(blockDim);
    MKI_LOG(INFO) << "launchBufferSize = " << tilingSize << " block dim = " << blockDim;
    return Status::OkStatus();
}

} // namespace AtbOps
