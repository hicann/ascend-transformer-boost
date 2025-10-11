/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "kernel_operator.h"

namespace PagedCacheLoad {

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t UB_LOCATE_INFO_SIZE = 40 * 1024;
constexpr int32_t BUF_NUM_LOCATE_INFO = 2;
constexpr int32_t UB_TMP_BUF_SIZE = 148 * 1024;
constexpr uint16_t DATA_COPY_MAX_BLKCNT = 4095;

template <typename T>
class PagedCacheLoadNz {
public:
    __aicore__ inline PagedCacheLoadNz(){};
    __aicore__ inline void Init(__gm__ uint8_t * __restrict__ keyCacheInGm,
                                __gm__ uint8_t * __restrict__ valueCacheInGm,
                                __gm__ uint8_t * __restrict__ blockTablesInGm,
                                __gm__ uint8_t * __restrict__ contextLensInGm,
                                __gm__ uint8_t * __restrict__ seqStartsInGm,
                                __gm__ uint8_t * __restrict__ keyOutGm,
                                __gm__ uint8_t * __restrict__ valueOutGm,
                                __gm__ uint8_t * __restrict__ tilingParaGm);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(__gm__ uint8_t * __restrict__ tilingGm);
    __aicore__ inline void CopyLocateInfoIn(int32_t tokenId);
    __aicore__ inline void RestoreDataFromCache(int32_t blkTabValue, int32_t blockOffset,
                                    int32_t numRsltProc, int32_t ctxtId, bool isKeyValue);

private:
    AscendC::TPipe pipe;

    // input

    AscendC::GlobalTensor<T> inKeyCacheGM;
    AscendC::GlobalTensor<T> inValueCacheGM;
    AscendC::GlobalTensor<int32_t> inBlockTableGM;
    AscendC::GlobalTensor<int32_t> inContextLenGM;
    AscendC::GlobalTensor<int32_t> inSeqStartsGM;

    // output
    AscendC::GlobalTensor<T> outKeyGM;
    AscendC::GlobalTensor<T> outValueGM;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> locateInfoBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    AscendC::LocalTensor<int32_t> contextLenTensor;
    AscendC::LocalTensor<int32_t> blockTableTensor;
    AscendC::LocalTensor<T> tmpTensor;

    // tiling param
    int32_t blockSize = 0;
    int32_t numTokens = 0;
    int32_t numblkTabCol = 0;
    int32_t tokenSizeK = 0;
    int32_t tokenSizeV = 0;
    int32_t typeByte = 0;
    int32_t hasSeqStarts = 0;
    int32_t cuSeqLens = 0;

    uint32_t maxNumTokens = 0;
    uint32_t eleNumPerBlk = 0;
};

template <typename T>
__aicore__ inline void PagedCacheLoadNz<T>::Init(
    __gm__ uint8_t * __restrict__ keyCacheInGm,
    __gm__ uint8_t * __restrict__ valueCacheInGm,
    __gm__ uint8_t * __restrict__ blockTablesInGm,
    __gm__ uint8_t * __restrict__ contextLensInGm,
    __gm__ uint8_t * __restrict__ seqStartsInGm,
    __gm__ uint8_t * __restrict__ keyOutGm,
    __gm__ uint8_t * __restrict__ valueOutGm,
    __gm__ uint8_t * __restrict__ tilingParaGm)
{
    ParseTilingData(tilingParaGm);

    // input
    this->inKeyCacheGM.SetGlobalBuffer((__gm__ T *)keyCacheInGm);
    this->inValueCacheGM.SetGlobalBuffer((__gm__ T *)valueCacheInGm);
    this->inBlockTableGM.SetGlobalBuffer((__gm__ int32_t *)blockTablesInGm);
    this->inContextLenGM.SetGlobalBuffer((__gm__ int32_t *)contextLensInGm);
    this->inSeqStartsGM.SetGlobalBuffer((__gm__ int32_t *)seqStartsInGm);
    // output
    this->outKeyGM.SetGlobalBuffer((__gm__ T *)keyOutGm);
    this->outValueGM.SetGlobalBuffer((__gm__ T *)valueOutGm);

    this->pipe.InitBuffer(this->locateInfoBuf, UB_LOCATE_INFO_SIZE); // 40kb, save context len and block table
    this->pipe.InitBuffer(this->tmpBuf, UB_TMP_BUF_SIZE); // ensure the size of one token less than 148kb

    // max number of context len per copy
    this->maxNumTokens = (UB_LOCATE_INFO_SIZE - BUF_NUM_LOCATE_INFO * BLOCK_SIZE)
                                    / sizeof(int32_t) / (this->numblkTabCol + 1);

    int32_t maxBlkNumTokensRnd = (this->maxNumTokens * sizeof(int32_t) - 1) / BLOCK_SIZE + 1;
    int32_t maxNumTokensRnd = maxBlkNumTokensRnd * BLOCK_SIZE / sizeof(int32_t);

    AscendC::LocalTensor<int32_t> locateInfoTensor = locateInfoBuf.Get<int32_t>();
    this->contextLenTensor = locateInfoTensor[0];
    this->blockTableTensor = locateInfoTensor[maxNumTokensRnd];
    this->tmpTensor = tmpBuf.Get<T>();
    this->eleNumPerBlk = BLOCK_SIZE / this->typeByte; // fp16, bf16, int8
}


template <typename T>
__aicore__ inline void PagedCacheLoadNz<T>::ParseTilingData(__gm__ uint8_t * __restrict__ tilingGm)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    int64_t locId = 0;
    this->blockSize = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    this->numTokens = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    this->numblkTabCol = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    this->tokenSizeK = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    this->tokenSizeV = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    this->typeByte = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    this->hasSeqStarts = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    this->cuSeqLens = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
}


template <typename T>
__aicore__ inline void PagedCacheLoadNz<T>::CopyLocateInfoIn(int32_t tokenId)
{
    int64_t copyNum = this->numTokens - tokenId;
    if (copyNum > this->maxNumTokens) {
        copyNum = this->maxNumTokens;
    }

    uint16_t ctxtLenBlkNum = (copyNum * sizeof(int32_t) - 1) / BLOCK_SIZE + 1; // block number
    uint16_t blkTabBlkNum = (copyNum * this->numblkTabCol * sizeof(int32_t) - 1) / BLOCK_SIZE + 1; // block number

    AscendC::DataCopyParams ctxtLenCopyParams = {1, ctxtLenBlkNum, 0, 0};
    AscendC::DataCopyParams blkTabCopyParams = {1, blkTabBlkNum, 0, 0};

    DataCopy(this->contextLenTensor, this->inContextLenGM[static_cast<uint64_t>(tokenId)], ctxtLenCopyParams);
    DataCopy(this->blockTableTensor, this->inBlockTableGM[static_cast<uint64_t>(tokenId
                                            * this->numblkTabCol)], blkTabCopyParams);
    AscendC::PipeBarrier<PIPE_ALL>();
}


template <typename T>
__aicore__ inline void PagedCacheLoadNz<T>::RestoreDataFromCache(
    int32_t blkTabValue,
    int32_t blockOffset,
    int32_t numRsltProc,
    int32_t ctxtId,
    bool isKeyValue)
{
    // default is key
    AscendC::GlobalTensor<T> inCacheGM = this->inKeyCacheGM;
    AscendC::GlobalTensor<T> outGM = this->outKeyGM;
    uint64_t tokenSize = this->tokenSizeK; // improve precision to avoid out of gm addr limit
    if (!isKeyValue) {
        inCacheGM = this->inValueCacheGM;
        outGM = this->outValueGM;
        tokenSize = this->tokenSizeV;
    }

    uint64_t start = (numRsltProc + ctxtId) * tokenSize * this->typeByte / sizeof(int8_t); // use int8 as T commonly
    uint64_t cacheStart = (blkTabValue * this->blockSize * tokenSize + blockOffset
                        * this->eleNumPerBlk) * this->typeByte / sizeof(int8_t); // use int8 as T commonly

    uint16_t inBlkCnt = tokenSize / this->eleNumPerBlk; // max number is 4095
    uint16_t inBlkLen = 1;
    uint16_t inSrcStride = this->blockSize - 1;
    uint16_t inDstStride = 0;

    AscendC::DataCopyParams copyInParams = {inBlkCnt, inBlkLen, inSrcStride, inDstStride};

    if (inBlkCnt > DATA_COPY_MAX_BLKCNT) {
        // use int8 as T commonly
        uint32_t localOffset = DATA_COPY_MAX_BLKCNT * BLOCK_SIZE / sizeof(int8_t);
        uint64_t globalOffset = DATA_COPY_MAX_BLKCNT * this->blockSize * BLOCK_SIZE / sizeof(int8_t);

        DataCopy(this->tmpTensor, inCacheGM[cacheStart],
                {DATA_COPY_MAX_BLKCNT, inBlkLen, inSrcStride, inDstStride});

        uint16_t leftBlkCnt = inBlkCnt - DATA_COPY_MAX_BLKCNT;
        DataCopy(this->tmpTensor[localOffset], inCacheGM[cacheStart + globalOffset],
                {leftBlkCnt, inBlkLen, inSrcStride, inDstStride});
    } else {
        DataCopy(this->tmpTensor, inCacheGM[cacheStart], copyInParams);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    uint16_t outBlkLen = tokenSize * this->typeByte / BLOCK_SIZE; // fp16, bf16, int8
    AscendC::DataCopyParams copyOutParams = {1, outBlkLen, 0, 0};

    DataCopy(outGM[start], this->tmpTensor, copyOutParams);
    AscendC::PipeBarrier<PIPE_ALL>();
}


template <typename T>
__aicore__ inline void PagedCacheLoadNz<T>::Process()
{
    int32_t vecIdx = AscendC::GetBlockIdx(); // current vector core id
    int32_t coreNum = AscendC::GetBlockNum(); // total vector core number

    int32_t numRsltProc = 0;
    int32_t ctxtStart = 0;
    if (this->cuSeqLens) ctxtStart = this->contextLenTensor.GetValue(0);

    for (int32_t tokenId = 0; tokenId < this->numTokens; tokenId++) {
        int32_t localtokenId = tokenId % this->maxNumTokens;

        // copy partial block table in ub when local data is used up
        if (localtokenId == 0) {
            CopyLocateInfoIn(tokenId);
        }

        // get the context len of current token
        int32_t ctxtLenValue;
        if (this->cuSeqLens) {
            int32_t ctxtEnd = this->contextLenTensor.GetValue(localtokenId + 1);
            ctxtLenValue = ctxtEnd - ctxtStart;
            ctxtStart = ctxtEnd;
        } else {
            ctxtLenValue = this->contextLenTensor.GetValue(localtokenId);
        }

        uint32_t localBlkTabId = 0;
        int32_t blkTabValue = 0;
        for (int32_t ctxtId = 0; ctxtId < ctxtLenValue; ctxtId++) {

            // need to change block id when block offset is equal to zero
            int32_t blkOffset = ctxtId % this->blockSize;
            if (blkOffset == 0) {
                localBlkTabId = localtokenId * this->numblkTabCol + ctxtId / this->blockSize;
                if (this->hasSeqStarts) localBlkTabId += this->inSeqStartsGM.GetValue(localtokenId);
                blkTabValue = this->blockTableTensor.GetValue(localBlkTabId);
            }

            // process one matrix every time for each core
            if (ctxtId % coreNum != vecIdx) {
                continue;
            }

            // process key and value
            RestoreDataFromCache(blkTabValue, blkOffset, numRsltProc, ctxtId, true);
            RestoreDataFromCache(blkTabValue, blkOffset, numRsltProc, ctxtId, false);
        }
        numRsltProc += ctxtLenValue;
    }
}
}


extern "C" __global__ __aicore__ void paged_cache_load_nz(
    __gm__ uint8_t * __restrict__ keyCacheInGm,
    __gm__ uint8_t * __restrict__ valueCacheInGm,
    __gm__ uint8_t * __restrict__ blockTablesInGm,
    __gm__ uint8_t * __restrict__ contextLensInGm,
    __gm__ uint8_t * __restrict__ keyInGm,
    __gm__ uint8_t * __restrict__ valueInGm,
    __gm__ uint8_t * __restrict__ seqStartsInGm,
    __gm__ uint8_t * __restrict__ keyOutGm,
    __gm__ uint8_t * __restrict__ valueOutGm,
    __gm__ uint8_t * __restrict__ tilingParaGm)
{
    PagedCacheLoad::PagedCacheLoadNz<int8_t> op;
    op.Init(keyCacheInGm, valueCacheInGm, blockTablesInGm,
             contextLensInGm, seqStartsInGm, keyOutGm, valueOutGm, tilingParaGm);
    op.Process();
}