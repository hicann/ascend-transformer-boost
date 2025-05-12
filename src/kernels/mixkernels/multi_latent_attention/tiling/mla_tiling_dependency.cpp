/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <securec.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "mla_tiling.h"

namespace AtbOps {
const int32_t TILING_BATCH = 0;
const int32_t TILING_NUMHEADS = 1;
const int32_t TILING_HEADDIM = 2;
const int32_t TILING_NUMBLOKS = 3;
const int32_t TILING_BLOCKSIZE = 4;
const int32_t TILING_MAXBLOCKS = 5;
const int32_t TILING_TOR = 6;
const int32_t TILING_KVHEADS = 7;
const int32_t TILING_HEADSIZE = 8;
const int32_t TILING_PARASIZE = 9;
const int32_t TILING_MTP_HEAD_SPLIT_SIZE = 10;
const int32_t TILING_TOTAL_BLOCK_NUM = 11;
const int32_t TILING_MASK_TYPE_ND = 12;
const int32_t TILING_TASK_NUM = 13;
const int32_t TILING_MAX_KV_SEQ_LEN = 14;
const int32_t TILING_KV_SPLIT_NUM = 15;
const int32_t TILING_SPLIT_TASKNUM = 16;
const int32_t TILING_TAIL_BATCH = 17;

const int32_t NUM0 = 0;
const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM6 = 6;
const int32_t NUM7 = 7;
const int32_t NUM8 = 8;
const int32_t NUM9 = 9;
const int32_t NUM10 = 10;
const int32_t NUM11 = 11;
const int32_t NUM12 = 12;
const int32_t NUM13 = 13;
const int32_t NUM14 = 14;
const int32_t NUM15 = 15;
const int32_t NUM16 = 16;
const int32_t NUM17 = 17;
const int32_t NUM18 = 18;
const int32_t NUM19 = 19;
const int32_t NUM20 = 20;
const int32_t NUM21 = 21;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM128 = 128;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const int32_t INDEX125 = 125;
const int32_t INDEX126 = 126;
const int32_t INDEX127 = 127;
const int32_t INDEX190 = 190;
const int32_t INDEX191 = 191;
const int32_t SPECIALNUM_TOKENS = 16;
const int32_t SPECIALNUM_HEADS = 32;
const int32_t EMBEDDING_LIMIT = 128;
const int32_t HIGH_32BIT = 32;
const uint32_t BATCH_MLA = 32;
const uint32_t BLOCK_DIM_MLA = 20;
const int32_t PP_MM_NUM = 8;
const int32_t PP_INDEX = 16;

constexpr std::array<int32_t, PP_MM_NUM> PP_MM = { 16, 32, 48, 64, 80, 96, 112, 128 };
constexpr std::array<int32_t, NUM6> QN_TILE_LIST = { 128, 64, 32, 16, 8, 1 };

using IndexArr = std::array<int32_t, NUM4>;

inline uint32_t GetHigh32Bit(uint64_t v) { return static_cast<uint32_t>(v >> HIGH_32BIT); }
inline uint32_t GetLoww32Bit(uint64_t v) { return static_cast<uint32_t>(v); }

inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound) // 16, 7
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}

void GetAddrOffsetMLA(uint32_t *tilingParam, const AddrOffsets addrOffsets, const int32_t tilingOffset)
{
    tilingParam[tilingOffset + NUM2] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[tilingOffset + NUM3] = GetLoww32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[tilingOffset + NUM4] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
    tilingParam[tilingOffset + NUM5] = GetLoww32Bit(addrOffsets.addrOSeqOffset);

    // mask offset
    tilingParam[tilingOffset + NUM6] = GetHigh32Bit(addrOffsets.addrMaskOffset);
    tilingParam[tilingOffset + NUM7] = GetLoww32Bit(addrOffsets.addrMaskOffset);
}

int32_t GetQNBlockTile(const MLAInfo &mmInfo, int32_t qSeqLen)
{
    int32_t tileListIdx = static_cast<int32_t>(std::ceil(std::log2(qSeqLen)));
    tileListIdx = (tileListIdx > NUM5) ? NUM5 : tileListIdx;
    int32_t qNBlockTile = QN_TILE_LIST[tileListIdx];
    int32_t group = mmInfo.numHeads / mmInfo.kvHeads;
    qNBlockTile = (qNBlockTile > group) ? group : qNBlockTile;

    return qNBlockTile;
}

int32_t GetMaxQseqlen(const OpParam::MLA &param)
{
    auto qSeqLen = param.qSeqLen;
    auto maxQSeqlenIter = std::max_element(qSeqLen.begin(), qSeqLen.end());
    auto maxQseqlen = maxQSeqlenIter != qSeqLen.end() ? *maxQSeqlenIter : 1;
    MKI_LOG(INFO) << "maxQseqlen = " << maxQseqlen;
    return maxQseqlen;
}

int32_t GetMaxKVseqlen(const OpParam::MLA &param)
{
    auto kvSeqLen = param.kvSeqLen;
    auto maxKVSeqlenIter = std::max_element(kvSeqLen.begin(), kvSeqLen.end());
    auto maxKVseqlen = maxKVSeqlenIter != kvSeqLen.end() ? *maxKVSeqlenIter : 1;
    MKI_LOG(INFO) << "maxKVseqlen = " << maxKVseqlen;
    return maxKVseqlen;
}

Status GetNdMLATiling(const MLAInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam,
                      const OpParam::MLA &param)
{
    AddrOffsets addrOffsets {};
    auto qSeqLen = param.qSeqLen;
    int32_t maxQseqlen =  GetMaxQseqlen(param);
    MKI_CHECK(maxQseqlen > 0, "qSeqlen max value invalid, please check",
        return AtbOps::Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        
    int32_t maxKVseqlen =  GetMaxKVseqlen(param);
    MKI_CHECK(maxKVseqlen > 0, "kvSeqlen max value invalid, please check",
        return AtbOps::Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));

    int32_t curQNBlockTile = GetQNBlockTile(mmInfo, maxQseqlen);

    uint32_t emptySeq = (mmInfo.qSeqLen == nullptr) ? 1 : 0;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = 1;
        qSeqLen = (emptySeq == 1) ? 1 : *(mmInfo.qSeqLen + seqIdx);
        qSeqLen = (*(mmInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqLen;
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);

        int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE * seqIdx;
        tilingParam[tilingOffset] = static_cast<uint32_t>(qSeqLen);
        tilingParam[tilingOffset + 1] = static_cast<uint32_t>(kvSeqlen);

        GetAddrOffsetMLA(tilingParam, addrOffsets, tilingOffset);
        uint64_t addressQffset = static_cast<uint64_t>(mmInfo.numHeads * qSeqLen);
        uint64_t addressOffset = static_cast<uint64_t>(mmInfo.numHeads * mmInfo.embeddingSize * qSeqLen);
        uint64_t addressMaskOffset = static_cast<uint64_t>(qSeqLen * maxKVseqlen);
        addrOffsets.addrQSeqOffset += addressQffset;
        addrOffsets.addrOSeqOffset += addressOffset;
        addrOffsets.addrMaskOffset += addressMaskOffset;
    }

    tilingParam[TILING_MTP_HEAD_SPLIT_SIZE] = static_cast<uint32_t>(curQNBlockTile);
    tilingParam[TILING_MAX_KV_SEQ_LEN] = static_cast<uint32_t>(maxKVseqlen);

    return AtbOps::Status::OkStatus();
}

void GetDecodingNormalTaskTiling(MLAInfo &mmInfo, uint32_t blockDim, uint32_t *tilingParam, int32_t &curTask,
                                 int32_t &prevTaskNum)
{
    int32_t maxQPerJob = mmInfo.quantFlag ? NUM1 : NUM2;
    int32_t qRowIdx = 0;
    for (int32_t taskIdx = 0; taskIdx < mmInfo.batch; taskIdx++) {
        int32_t batchIdx = mmInfo.batchList[taskIdx].batchIdx;
        int32_t qSeqLen = mmInfo.qSeqLen == nullptr ? 1 : *(mmInfo.qSeqLen + batchIdx);
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + batchIdx);
        int32_t curKvSeq = kvSeqlen - qSeqLen;
        if (prevTaskNum >= mmInfo.normalTaskNum) {
            curTask = taskIdx;
            break;
        }
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq += maxQPerJob) {
            int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE_TP1 * prevTaskNum;
            int32_t curQLen = ((qSeqLen - qSeq) > maxQPerJob) ? maxQPerJob : (qSeqLen - qSeq);
            curKvSeq += curQLen;
            tilingParam[tilingOffset] = batchIdx;
            tilingParam[tilingOffset + NUM1] = batchIdx * qSeqLen + qSeq;
            tilingParam[tilingOffset + NUM2] = curKvSeq;
            tilingParam[tilingOffset + NUM3] = curQLen;
            prevTaskNum++;
            qRowIdx += curQLen;
        }
    }
}

void GetDecodingTailBatchTiling(MLAInfo &mmInfo, uint32_t *tilingParam, int32_t &curTask, int32_t &prevTaskNum,
                                std::unordered_map<int, int> &batchTilingMap)
{
    int32_t batchTilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE_TP1 * prevTaskNum;
    for (int32_t taskIdx = curTask; taskIdx < mmInfo.batch; taskIdx++) {
        int32_t batchIdx = mmInfo.batchList[taskIdx].batchIdx;
        int32_t qSeqLen = mmInfo.qSeqLen == nullptr ? 1 : *(mmInfo.qSeqLen + batchIdx);
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            tilingParam[batchTilingOffset +
                        ((taskIdx - curTask) * qSeqLen + qSeq) * NUM2] = mmInfo.quantFlag ?
                                                                         batchTilingMap[batchIdx * qSeqLen + qSeq] :
                                                                         batchTilingMap[batchIdx * qSeqLen];
            tilingParam[batchTilingOffset +
                        ((taskIdx - curTask) * qSeqLen + qSeq) * NUM2 + 1] = mmInfo.quantFlag ? 0 : qSeq;
        }
    }
}

void GetNdMLADecodingMtpTilingTP1(MLAInfo &mmInfo, uint32_t blockDim, uint32_t tilingParam[])
{
    int32_t curTask = 0;
    int32_t prevTaskNum = 0;
    GetDecodingNormalTaskTiling(mmInfo, blockDim, tilingParam, curTask, prevTaskNum);
    tilingParam[TILING_TAIL_BATCH] = static_cast<uint32_t>(mmInfo.batch - curTask);
    std::unordered_map<int, int> batchTilingMap;
    for (int32_t taskIdx = curTask; taskIdx < mmInfo.batch; taskIdx++) {
        int32_t batchIdx = mmInfo.batchList[taskIdx].batchIdx;
        int32_t qSeqLen = mmInfo.qSeqLen == nullptr ? 1 : *(mmInfo.qSeqLen + batchIdx);
        int32_t qLoop = mmInfo.quantFlag ? qSeqLen : 1;
        int32_t splitNum = mmInfo.splitKVNum;
        for (int32_t qSeq = 0; qSeq < qLoop; qSeq++) {
            int32_t nowKVSeqlen = mmInfo.batchList[taskIdx].kvSeqlen - qLoop + qSeq + 1;
            int32_t kvBlocks = (nowKVSeqlen + mmInfo.blockSize - 1) / mmInfo.blockSize;
            if (kvBlocks < splitNum) {
                splitNum = kvBlocks;
            }
            int32_t prevKVSeqlen = 0;
            int32_t splitBlocks = (kvBlocks + splitNum - 1) / splitNum;
            if (qSeqLen == NUM2 && splitBlocks == 1 && nowKVSeqlen % mmInfo.blockSize == 1) {
                splitNum--;
                splitBlocks = (kvBlocks + splitNum - 1) / splitNum;
            }
            int32_t resBlocks = (splitNum - kvBlocks % splitNum) % splitNum;
            int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE_TP1 * prevTaskNum;
            batchTilingMap[batchIdx * qSeqLen + qSeq] = tilingOffset;
            for (int kvBlockIdx = 0; kvBlockIdx < splitNum; kvBlockIdx++) {
                int32_t nowBlocks = kvBlockIdx < resBlocks ? splitBlocks - 1 : splitBlocks;
                tilingParam[tilingOffset] = batchIdx;
                tilingParam[tilingOffset + 1] = qSeqLen;
                tilingParam[tilingOffset + NUM2] = kvBlockIdx == splitNum - 1 ?
                nowKVSeqlen - prevKVSeqlen : nowBlocks * mmInfo.blockSize;
                tilingParam[tilingOffset + NUM3] = prevKVSeqlen;
                tilingParam[tilingOffset + NUM4] = mmInfo.quantFlag ? batchIdx * qSeqLen + qSeq : batchIdx * qSeqLen;
                tilingParam[tilingOffset + NUM5] = splitNum;
                tilingParam[tilingOffset + NUM6] = mmInfo.prevSplitNumSum;
                prevKVSeqlen += tilingParam[tilingOffset + NUM2];
                tilingOffset += TILING_PARA_SIZE_TP1;
                prevTaskNum++;
            }
            mmInfo.prevSplitNumSum += splitNum;
        }
    }
    GetDecodingTailBatchTiling(mmInfo, tilingParam, curTask, prevTaskNum, batchTilingMap);
}

void GetNdMLAMtpTilingTP1(const MLAInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam,
                          const OpParam::MLA &param)
{
    bool isFP16 = static_cast<int32_t>(mmInfo.type) < NUM2;
    int32_t maxQPerJob = isFP16 ? NUM2 : NUM1;
    int32_t prevTaskNum = 0;
    int32_t qRowIdx = 0;
    int32_t totalTaskNum = mmInfo.totalTaskNum;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = mmInfo.qSeqLen == nullptr ? 1 : *(mmInfo.qSeqLen + seqIdx);
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);
        int32_t curKvSeq = kvSeqlen - qSeqLen;
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq += maxQPerJob) {
            if (totalTaskNum <= static_cast<int32_t>(blockDim) && (prevTaskNum % static_cast<int32_t>(blockDim) == 0)) {
                maxQPerJob = NUM1;
            }
            int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE_TP1 * prevTaskNum;
            int32_t curQLen = ((qSeqLen - qSeq) > maxQPerJob) ? maxQPerJob : (qSeqLen - qSeq);
            curKvSeq += curQLen;
            tilingParam[tilingOffset] = seqIdx;
            tilingParam[tilingOffset + NUM1] = qRowIdx;
            tilingParam[tilingOffset + NUM2] = curKvSeq;
            tilingParam[tilingOffset + NUM3] = curQLen;
            prevTaskNum++;
            qRowIdx += curQLen;
            totalTaskNum -= curQLen;
            MKI_LOG(INFO) << "seqIdx = " << tilingParam[tilingOffset];
            MKI_LOG(INFO) << "qRowIdx = " << tilingParam[tilingOffset + NUM1];
            MKI_LOG(INFO) << "kvSeqlen = " << tilingParam[tilingOffset + NUM2];
            MKI_LOG(INFO) << "curQLen = " << tilingParam[tilingOffset + NUM3];
            MKI_LOG(INFO) << "TILING_BATCH = " << tilingParam[TILING_BATCH];
        }
    }
}

void GetTilingHead(const MLAInfo &mmInfo, const OpParam::MLA &param, uint32_t *tilingParam,
                   const uint32_t *torPtr, uint32_t blockDim)
{
    tilingParam[TILING_BATCH] = static_cast<uint32_t>(mmInfo.batch);
    tilingParam[TILING_HEADSIZE] = static_cast<uint32_t>(TILING_HEAD_SIZE);
    tilingParam[TILING_PARASIZE] = mmInfo.mtpTp1Flag ? static_cast<uint32_t>(TILING_PARA_SIZE_TP1) :
                                    static_cast<uint32_t>(TILING_PARA_SIZE);
    tilingParam[TILING_NUMHEADS] = static_cast<uint32_t>(mmInfo.numHeads);
    tilingParam[TILING_HEADDIM] = static_cast<uint32_t>(mmInfo.embeddingSize);
    tilingParam[TILING_NUMBLOKS] = static_cast<uint32_t>(mmInfo.numBlocks);
    tilingParam[TILING_BLOCKSIZE] = static_cast<uint32_t>(mmInfo.blockSize);
    tilingParam[TILING_MAXBLOCKS] = static_cast<uint32_t>(mmInfo.maxNumBlocksPerQuery);
    tilingParam[TILING_TOR] = *torPtr;
    tilingParam[TILING_KVHEADS] = (mmInfo.kvHeads == 0) ? mmInfo.numHeads : mmInfo.kvHeads;

    tilingParam[TILING_MASK_TYPE_ND] = static_cast<uint32_t>(mmInfo.maskType);
    if (mmInfo.flashDecoding) {
        tilingParam[TILING_TASK_NUM] = static_cast<uint32_t>(mmInfo.normalTaskNum);
        tilingParam[TILING_KV_SPLIT_NUM] = static_cast<uint32_t>(mmInfo.splitKVNum);
        tilingParam[TILING_SPLIT_TASKNUM] = static_cast<uint32_t>(mmInfo.prevSplitNumSum);
    } else {
        tilingParam[TILING_TASK_NUM] = static_cast<uint32_t>(mmInfo.totalTaskNum);
    }

    MKI_LOG(INFO) << "TILING_BATCH = " << tilingParam[TILING_BATCH];
    MKI_LOG(INFO) << "TILING_HEADSIZE = " << tilingParam[TILING_HEADSIZE];
    MKI_LOG(INFO) << "TILING_PARASIZE = " << tilingParam[TILING_PARASIZE];
    MKI_LOG(INFO) << "TILING_NUMHEADS = " << tilingParam[TILING_NUMHEADS];
    MKI_LOG(INFO) << "TILING_HEADDIM = " << tilingParam[TILING_HEADDIM];
    MKI_LOG(INFO) << "TILING_NUMBLOKS = " << tilingParam[TILING_NUMBLOKS];
    MKI_LOG(INFO) << "TILING_BLOCKSIZE = " << tilingParam[TILING_BLOCKSIZE];
    MKI_LOG(INFO) << "TILING_MAXBLOCKS = " << tilingParam[TILING_MAXBLOCKS];
    MKI_LOG(INFO) << "TILING_TOR = " << tilingParam[TILING_TOR];
    MKI_LOG(INFO) << "TILING_KVHEADS = " << tilingParam[TILING_KVHEADS];
    MKI_LOG(INFO) << "TILING_TASK_NUM = " << tilingParam[TILING_TASK_NUM];
    MKI_LOG(INFO) << "TILING_KV_SPLIT_NUM = " << tilingParam[TILING_KV_SPLIT_NUM];
    MKI_LOG(INFO) << "TILING_SPLIT_TASKNUM = " << tilingParam[TILING_SPLIT_TASKNUM];
}

Status GetMLATilingParam(const LaunchParam &launchParam, MLAInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize)
{
    auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
    float tor = mmInfo.tor;
    uint32_t *torPtr = reinterpret_cast<uint32_t *>(&tor);
    uint64_t curTilingParamSize = mmInfo.mtpTp1Flag ?
                                  (TILING_HEAD_SIZE + TILING_PARA_SIZE_TP1 * mmInfo.totalTaskNum) * sizeof(uint32_t) :
                                  (TILING_HEAD_SIZE + TILING_PARA_SIZE * mmInfo.batch) * sizeof(uint32_t);
    MKI_CHECK(memset_s(tilingParam, tilingParamSize, 0, curTilingParamSize) == EOK, "init tiling failed",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    if (mmInfo.mtpTp1Flag) {
        if (mmInfo.flashDecoding) {
            GetNdMLADecodingMtpTilingTP1(mmInfo, blockDim, tilingParam);
        } else {
            GetNdMLAMtpTilingTP1(mmInfo, blockDim, tilingParam, param);
        }
    } else {
        GetNdMLATiling(mmInfo, blockDim, tilingParam, param);
        blockDim = mmInfo.batch == BATCH_MLA ? BLOCK_DIM_MLA : blockDim;
    }
    GetTilingHead(mmInfo, param, tilingParam, torPtr, blockDim);
    return AtbOps::Status::OkStatus();
}
} // end paged_attention namespace
