/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "ops/elewise/cast/tiling/tiling_data.h"
#include "ops/utils/kernel/kernel_utils.h"

namespace {
static constexpr uint32_t MAX_PROCESS_NUM = 12288 * 8;
static constexpr uint32_t BLOCK_SIZE = 16;
static constexpr uint32_t BUFFER_NUM = 1;
static constexpr uint32_t MODE_NONE = 0;
static constexpr uint32_t MODE_RINT = 1;
static constexpr uint32_t MODE_FLOOR = 2;
} // namespace

template <typename InType = half, typename OutType = float> class CastWideNd {
public:
    __aicore__ explicit CastWideNd(){};

    __aicore__ inline uint32_t ROUND_UP(uint32_t x) { return (((x) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE; }

    __aicore__ inline void setArgs(__gm__ uint8_t *x, __gm__ uint8_t *y, uint32_t numTotalIn, uint32_t blockNumIn,
                                   uint32_t blockTailIn, uint32_t ubFactorIn, uint32_t attrIn)
    {
        gm_x = reinterpret_cast<__gm__ InType *>(x);
        gm_y = reinterpret_cast<__gm__ OutType *>(y);

        numTotal = numTotalIn;
        blockNum = blockNumIn;
        blockTail = blockTailIn;
        ubFactor = ubFactorIn;
        blockLength = blockNum;
        transMode = static_cast<AscendC::RoundMode>(attrIn);
    }

    __aicore__ inline void Process()
    {
        Init();
        uint32_t loopCnt = (blockLength + ubFactor - 1) / ubFactor;
        for (uint64_t loop = 0; loop < loopCnt; loop++) {
            CopyIn(loop);
            Compute();
            CopyOut(loop);
        }
    }
    __aicore__ inline void ProcessInt32toHalf()
    {
        Init();
        uint64_t loopCnt = (blockLength + ubFactor - 1) / ubFactor;
        for (uint64_t loop = 0; loop < loopCnt; loop++) {
            CopyIn(loop);
            ComputeInt32ToHalf();
            CopyOut(loop);
        }
    }

private:
    __aicore__ inline void Init()
    {
        xGm.SetGlobalBuffer(gm_x + AscendC::GetBlockIdx() * blockLength);
        yGm.SetGlobalBuffer(gm_y + AscendC::GetBlockIdx() * blockLength);
        pipe.InitBuffer(xQue, BUFFER_NUM, ROUND_UP(ubFactor) * sizeof(InType));
        pipe.InitBuffer(yQue, BUFFER_NUM, ROUND_UP(ubFactor) * sizeof(OutType));
        pipe.InitBuffer(middleValue, ROUND_UP(ubFactor) * sizeof(float));
    }
    __aicore__ inline void CopyIn(uint64_t loopIdx)
    {
        AscendC::LocalTensor<InType> inX = xQue.AllocTensor<InType>();
        DataCopy(inX, xGm[loopIdx * ubFactor], ROUND_UP(ubFactor));
        xQue.EnQue(inX);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<InType> localx = xQue.DeQue<InType>();
        AscendC::LocalTensor<OutType> localy = yQue.AllocTensor<OutType>();
        Cast(localy, localx, transMode, ROUND_UP(ubFactor));
        xQue.FreeTensor(localx);
        yQue.EnQue(localy);
    }
    __aicore__ inline void ComputeInt32ToHalf()
    {
        AscendC::LocalTensor<InType> localx = xQue.DeQue<InType>();
        AscendC::LocalTensor<OutType> localy = yQue.AllocTensor<OutType>();
        AscendC::LocalTensor<float> xmiddleValue = middleValue.Get<float>();
        Cast(xmiddleValue, localx, AscendC::RoundMode::CAST_NONE, ROUND_UP(ubFactor));
        AscendC::PipeBarrier<PIPE_ALL>();
        Cast(localy, xmiddleValue, AscendC::RoundMode::CAST_NONE, ROUND_UP(ubFactor));
        xQue.FreeTensor(localx);
        yQue.EnQue(localy);
    }
    __aicore__ inline void CopyOut(uint64_t loopIdx)
    {
        AscendC::LocalTensor<OutType> yLocal = yQue.DeQue<OutType>();
        DataCopy(yGm[loopIdx * ubFactor], yLocal, ROUND_UP(ubFactor));
        yQue.FreeTensor(yLocal);
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> middleValue;
    AscendC::GlobalTensor<InType> xGm;
    AscendC::GlobalTensor<OutType> yGm;
    uint32_t numCore{static_cast<uint32_t>(AscendC::GetBlockNum())};
    uint32_t numTotal{0};
    uint32_t blockNum{0};
    uint32_t blockTail{0};
    uint32_t blockLength{0};
    uint32_t ubFactor{0};
    uint32_t dataTransKey{0};
    AscendC::RoundMode transMode{0};
    __gm__ InType *__restrict__ gm_x{nullptr};
    __gm__ OutType *__restrict__ gm_y{nullptr};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::CastTilingData *tilingData)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingData->numTotal = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingData->blockNum = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingData->blockTail = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingData->ubFactor = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingData->dataTransKey = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingData->transMode = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::CastTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingData->numTotal = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingData->blockNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingData->blockTail = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingData->ubFactor = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingData->dataTransKey = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingData->transMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                       \
    AsdOps::CastTilingData tiling_data;                                                                                \
    InitTilingData((tiling_arg), &(tiling_data))

extern "C" __global__ __aicore__ void cast_wide(GM_ADDR x, GM_ADDR y, GM_ADDR tiling_para_gm)
{
    GET_TILING_DATA(tilingData, tiling_para_gm);
    if (TILING_KEY_IS(17)) {
        CastWideNd<half, float> cast_half_to_float;
        cast_half_to_float.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                   tilingData.ubFactor, MODE_NONE);
        cast_half_to_float.Process();
    } else if (TILING_KEY_IS(21)) {
        CastWideNd<half, int32_t> cast_half_to_int32;
        cast_half_to_int32.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                   tilingData.ubFactor, MODE_FLOOR);
        cast_half_to_int32.Process();
    }  else if (TILING_KEY_IS(97)) {
        CastWideNd<int32_t, half> cast_int32_to_half;
        cast_int32_to_half.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                   tilingData.ubFactor, MODE_NONE);
        cast_int32_to_half.ProcessInt32toHalf();
    } else if (TILING_KEY_IS(33)) {
        CastWideNd<float, half> cast_float_to_half;
        cast_float_to_half.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                   tilingData.ubFactor, MODE_NONE);
        cast_float_to_half.Process();
    } else if (TILING_KEY_IS(36)) {
        CastWideNd<float, int32_t> cast_float_to_int32;
        cast_float_to_int32.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                    tilingData.ubFactor, MODE_FLOOR);
        cast_float_to_int32.Process();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    }else if (TILING_KEY_IS(99)) {
        CastWideNd<int32_t, int64_t> cast_int32_to_int64;
        cast_int32_to_int64.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                    tilingData.ubFactor, MODE_NONE);
        cast_int32_to_int64.Process();
    } else if (TILING_KEY_IS(114)) {
        CastWideNd<int64_t, int32_t> cast_int64_to_int32;
        cast_int64_to_int32.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                    tilingData.ubFactor, MODE_NONE);
        cast_int64_to_int32.Process();
    } else if (TILING_KEY_IS(38)) {
        CastWideNd<float, bfloat16_t> cast_float_to_bf16;
        cast_float_to_bf16.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                    tilingData.ubFactor, MODE_RINT);
        cast_float_to_bf16.Process();
    } else if (TILING_KEY_IS(130)) {
        CastWideNd<bfloat16_t, float> cast_bf16_to_float;
        cast_bf16_to_float.setArgs(x, y, tilingData.numTotal, tilingData.blockNum, tilingData.blockTail,
                                    tilingData.ubFactor, MODE_NONE);
        cast_bf16_to_float.Process();
#endif
    }
}