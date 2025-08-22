/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_DATACOPY_GM2GM_DELAY_H
#define LCCL_DATACOPY_GM2GM_DELAY_H
#include "datacopy_gm2gm.h"

using namespace AscendC;
using namespace Lcal;

template <typename V, typename T, typename U = T>
class DataCopyGM2GMDelay {
    constexpr static int64_t THREE_NUM = 3;
    constexpr static int64_t FOUR_NUM = 4;
    constexpr static int64_t WORK_OFFSET = 8192;
    constexpr static int64_t WORK_BLOCK_NUM = WORK_OFFSET / sizeof(T);
    constexpr static int64_t UB_HEAD_OFFSET = WORK_OFFSET * 2;
    constexpr static int64_t SCALE_SIZE = 32;
    constexpr static int64_t SCALE_NUM = SCALE_SIZE / sizeof(T);
    constexpr static int64_t SINGLE_SCALE_SIZE = 2;
    constexpr static int64_t BLOCK_NUM = (UB_SINGLE_DMA_SIZE_MAX - WORK_OFFSET * 2 - SCALE_SIZE * 4) / 2 /
                                         (sizeof(U) + sizeof(T)) / ALIGN_SIZE * ALIGN_SIZE;
    constexpr static int64_t IN_BLOCKSIZE = BLOCK_NUM * sizeof(U);
}

public:
    FOECE_INLINE_AICORE DataCopyGM2GMDelay() {}

    FORCE_INLINE_AICORE void Init(GlobalTensor<V>& outputGt, GlobalTensor<U> (&inputGt)[8],
        GlobalTensor<U> (&inputScaleGt)[8], const uint32_t calNum, int rankCount, GlobalTensor<U>& outScaleGt,
        TBuf<QuePosition::VECCALC> tbuf)
    {
        for (int index = 0; index < rankCount; index++) {
            this->inputGt[index] = inputGt[index];
            this->inputScaleGt[index] = inputScaleGt[index];
        }
        this->outputGt = outputGt;
        this->outScaleGt = outScaleGt;
        inTensor[0] = tbuf.GetWithOffset<U>(BLOCK_NUM, 0);
        inTensor[1] = tbuf.GetWithOffset<U>(BLOCK_NUM, WORK_OFFSET + SCALE_SIZE * HALF_NUM + IN_BLOCKSIZE * THREE_NUM);
        singleScaleUBTensor[0] = tbuf.GetWithOffset<T>(SCALE_NUM, IN_BLOCKSIZE);
        singleScaleUBTensor[1] = tbuf.GetWithOffset<T>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * HALF_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM); 
        singleScaleUUBTensor[0] = tbuf.GetWithOffset<U>(SCALE_NUM, IN_BLOCKSIZE + SCALE_SIZE);
        singleScaleUUBTensor[1] = tbuf.GetWithOffset<U>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * THREE_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        workUBTENSOR[0] = tbuf.GetWithOffset<T>(WORK_BLOCK_NUM, IN_BLOCKSIZE + SCALE_SIZE * HALF_NUM);        
        workUBTENSOR[1] = tbuf.GetWithOffset<T>(WORK_BLOCK_NUM, WORK_OFFSET + SCALE_SIZE * FOUR_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        outputUBTENSOR[0] = tbuf.GetWithOffset<T>(BLOCK_NUM, IN_BLOCKSIZE + SCALE_SIZE * HALF_NUM + WORK_OFFSET);
        outputUBTENSOR[1] = tbuf.GetWithOffset<T>(BLOCK_NUM, WORK_OFFSET * HALF_NUM + SCALE_SIZE * FOUR_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        this->rankCount = rankCount;
        totalDataSize = calNum * sizeof(U);
        this->calNum = calNum;
        this->rankId = rankId;
    }

    FORCE_INLINE_AICORE void PreProcess() 
    {
        for (int index = 0; index < rankCount; index++) {
            DataCopyWrap(scaleUUBTENSOR[0][indedx * SCALE_SIZE / sizeof(U)], inputScaleGt[index], SCALE_SIZE);
            pipe_barrier(PIPE_ALL);
            DataCopyWrap(scaleUUBTensor[1][index * SCALE / sizeof(U)], inputScaleGt[index], SCALE_SIZE);
            pipe_barrier(PIPE_ALL);
        }
        for (int index = 0; index < rankCount; index++) {
            scaleUBTensor[0][index].SetValue(0, scaleUBTensor[0].GetValue(index * SCALE_SIZE / sizeof(T)));
            pipe_barrier(PIPE_ALL);
            scaleUBTensor[1][index].SetValue(0, scaleUBTensor[1].GetValue(index * SCALE_SIZE / sizeof(T)));
            pipe_barrier(PIPE_ALL);
            outputUBTensor[0][index].SetValue(0, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        Div(scaleUBTensor[1], outputUBTensor[0], scaleUBTensor[1], rankCount);
        AscendC::PipeBarrier<PIPE_ALL>();
        ReduceMin<T>(singleScaleUBTensor[0], singleScaleUUBTensor[0], 
            workUBTensor[1][WORK_BLOCK_NUM / HALF_NUM], rankCount, false);
        pipe_barrier(PIPE_ALL);
        DataCopyWrap(outScaleGt, singleScaleUUBTensor[0], sizeof(T));
        AscendC::PipeBarrier<PIPE_ALL>();
    }


    FORCE_INLINE_AICORE void LoopUncastAndMul(int idx, int index, event_t eventId)
    {
        PipeBarrier<PIPE_V>();
        T scalarValue = scaleUBTensor[1].GetValue(index);
        PipeBarrier__ubuf__ U* inputUB = nullptr;
    __ubuf__ T* outputUB = nullptr;
    
    }
private:
    template <typename T1, typename T2>
    FORCE_INLINE_AICORE T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

private:
    int64_t totalDataSize = 0;
    int rankCount;
    int perRankNumRemain;
    int calNum;
    int rankId;
    int numLayer;

    LocalTensor<U> inTensor[2];
    LocalTensor<U> singleScaleUUBTensor[2];
    LocalTensor<U> singleScaleUBTensor[2];
    LocalTensor<U> scaleUUBTensor[2];
    LocalTensor<U> scaleUBTensor[2];
    LocalTensor<U> workUBTensor[2];
    LocalTensor<U> outputUBTensor[2];

    GlobalTensor<V> outputGt;
    GLobalTensor<U> inputGt[8];
    GLobalTensor<U> inputScaleGt[8];
    GLobalTensor<U> outScaleGt;
};

#endif // LCCL_DATACOPY_GM2GM_DELAYH

