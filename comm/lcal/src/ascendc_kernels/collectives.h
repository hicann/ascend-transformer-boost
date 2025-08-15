/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_COLLECTIVES_H
#define LCCL_COLLECTIVES_H

#include <climits>

#include "datacopy_gm2gm.h"
#include "datacopy_gm2gm_delay.h"
#include "sync_collectives.h"
using namespace AscendC;
using namespace Lcal;

#define KERNELS_ARGS_FUN() \
GM_ADDR input, GM_ADDR output, GM_ADDR commArgs, int64_t len, int64_t magic, int op, int root, int cycleCount, \
GM_ADDR scale, int64_t scaleCount, GM_ADDR offset

#define KERNELS_ARGS_CALL() \
input, output, commArgs, len, magic, op, root, cycleCount, scale, scaleCount, offset

#define KERNELS_GATHER_TABLE_ARGS_FUN() \
GM_ADDR embTable, GM_ADDR lookup, GM_ADDR revData, int64_t lookupLen, int64_t embTableLen, int64_t embTableDim

#define KERNELS_GATHER_TABLE_ARGS_CALL() \
embTable, lookup, revData, lookupLen, embTableLen, embTableDim

enum DfxPos : int {
    MAGIC,
    LEN,
    RUN_STATUS
};

class Collectives {
    constexpr static int32_t UB_HEAD_OFFSET = 96;
    constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + ALIGN_SIZE;
public:
    FORCE_INLINE_AICORE Collectives(int rank, int rankSize, uint32_t extraFlag) : rank(rank), rankSize(rankSize), 
        extraFlag(extraFlag) {}
    
    FORCE_INLINE_AICORE ~Collectives()
    {
        const int64_t notRunning = 0xdead;
        dfx.SetValue(RUN_STATUS, notRunning);
    }

    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        dumpAddr_ = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->dumpAddr;
        GlobalTensor<GM_ADDR> peerMemsAddrGm;
        peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ CommArgs *>(commArgs))->peerMems[0],
                                        LCAL_MAX_RANK_SIZE);
        for (int i = 0; i < rankSize; ++i) {
            shareAddrs[i] = peerMemsAddrGm.GetValue(i) + 
                            (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
        }
        dfx.SetGlobalBuffer((reinterpret_cast<__gm__ CommArgs *>(commArgs))->dfx,
            DFX_COUNT);
        this->root = root;
        this->len = len;
        this->magic = magic;
        this->localRank = reinterpret_cast<__gm__ CommArgs *>(commArgs)->localRank;
        this->localRankSize = reinterpret_cast<__gm__ CommArgs *>(commArgs)->localRankSize;
        this->xRankSize = localRankSize;
        this->yRankSize = rankSize / localRankSize;
        this->xRankIdx = rank % localRankSize;
        this->yRankIdx = rank / localRankSize;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum() * LCAL_BLOCK_NUM_MULTI;
        
        sync.Init(rank, rankSize, shareAddrs);
        dfx.SetValue(MAGIC, magic);
        dfx.SetValue(LEN, len);
        const int64_t running = 0xbeef;
        dfx.SetValue(RUN_STATUS, running);
    }

    template <typename T>
    FORCE_INLINE_AICORE void DataCopyWrapPingPong(const GlobalTensor<T>& inputGT, const GlobalTensor<T>& outputGT, 
        int64_t dataSizeRemain, int op, TBuf<QuePosition::VECCALC> tbuf)
    {
        if (dataSizeRemain <= 0) {
            return;
        }
        LocalTensor<T> localUB[2];
        localUB[0] = tbuf.GetWithOffset<T>(UB_SINGLE_PING_PONG_ADD_SIZE_MAX, 0);
        localUB[1] = tbuf.GetWithOffset<T>(UB_SINGLE_PING_PONG_ADD_SIZE_MAX, UB_SINGLE_PING_PONG_ADD_SIZE_MAX);

        int inputOffsetNum = 0;
        int outputOffsetNum = 0;

        PipeBarrier<PIPE_ALL>();
        if (op != COPYONLY) {
            SetAscendCAtomic<T>(op);
        }
        PipeBarrier<PIPE_ALL>();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > UB_SINGLE_PING_PONG_ADD_SIZE_MAX ?
                UB_SINGLE_PING_PONG_ADD_SIZE_MAX : dataSizeRemain;
            TEventID eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            DataCopyWrap(localUB[(i & 1) ? 0 : 1], inputGT[inputOffsetNum], size);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            DataCopyWrap(outputGT[outputOffsetNum], localUB[(i & 1) ? 0 : 1], size);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);

        if (op != COPYONLY) {
            SetAtomicNone();
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T, typename U = T>
    FORCE_INLINE_AICORE void CpGM2GM(const GlobalTensor<T>& outputGT, const GlobalTensor<U>& inputGT,
        const uint32_t calCount, int op)
    {
        DataCopyGM2GM<T, U> cpKernel;
        cpKernel.Init(outputGT, inputGT, calCount, op);
        cpKernel.Process();
    }

    template <typename V, typename T, typename U = T>
    FORCE_INLINE_AICORE void CpGM2GMDelay(GlobalTensor<V>& outputGT, GlobalTensor<U> (&inputGT)[8],
        GlobalTensor<U> (&inputScaleGT)[8], const uint32_t calCount, int rankCount, GlobalTensor<U>& outScaleGT, 
        TBuf<QuePosition::VECCALC> tbuf)
    {
        DataCopyGM2GMDelay<V, T, U> cpKernel;
        cpKernel.Init(outputGT, inputGT, inputScaleGT, calCount, rankCount, outScaleGT, tbuf);
        cpKernel.Process();
    }

    template <typename T1, typename T2>
    FORCE_INLINE_AICORE T1 CeilDiv(T1 a, T2 b)
    {
      if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    FORCE_INLINE_AICORE void VecAddCce(int64_t curDealSize, __ubuf__ T *ubuf0, __ubuf__ T *ubuf1)
    {
        if (curDealSize > MAX_VADD_SIZE) {
            vadd(ubuf0, ubuf1, ubuf0, VADD_MAX_REPEAT, 1, 1, 1
                VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO);
            vadd((__ubuf__ T*)((__ubuf__ int8_t*)ubuf0 + VADD_MAX_REPEAT * VADD_UNIT_BYTE),
                (__ubuf__ T*)((__ubuf__ int8_t*)ubuf0 + VADD_MAX_REPEAT * VADD_UNIT_BYTE),
                (__ubuf__ T*)((__ubuf__ int8_t*)ubuf1 + VADD_MAX_REPEAT * VADD_UNIT_BYTE),
                VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO);
        } else {
            Avadd(ubuf0, ubuf1, ubuf0, VADD_MAX_REPEAT, 1, 1, 1,
                VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO); 
        }
    }

    template <typename T>
    FORCE_INLINE_AICORE void LoopVaddCceProcess(__ubuf__ T* localUB[2], const int64_t remainSize,
        int64_t (&targetRankArr)[8], const int64_t targetRankArrValidSize, const int64_t srcIpcOffsetNum,
        __gm__ T *srcGmMem, __gm__ T *dstGmMem, int64_t alreadyDealNum)
    {
        for 
    }
};