/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __COC_DEQUANTER__
#define __COC_DEQUANTER__

#ifdef __DAV_C220_VEC__

#include <type_traits>
#include "coc_internal.cce"

template <QuantGranularity GRANULARITY>
class LoopDequanter {
};

template <>
class LoopDequanter<QuantGranularity::PER_TENSOR> {
public:
    static constexpr int32_t max_len = 9792;
    inline __aicore__ LoopDequanter() = default;
    inline __aicore__ void SetForLoop()
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    inline __aicore__ void WaitForLoop()
    {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    inline __aicore__ void Loop(__gm__ bfloat16_t *dst, __gm__ int32_t *src, float32_t scale, int32_t offset,
            int32_t n_rows_this_loop, int32_t n_cols_this_loop, int32_t src_stride, int32_t dst_stride)
    {
        is_ping = !is_ping;
        auto ub_in = is_ping ? ub_in0 : ub_in1;
        auto ub_out = is_ping ? ub_out0 : ub_out1;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;

        int32_t n_blocks = Block32B<bfloat16_t>::Count(n_cols_this_loop) * (sizeof(int32_t) / sizeof(bfloat16_t));
        int32_t ubuf_gap = n_blocks - Block32B<int32_t>::Count(n_cols_this_loop);
        WaitFlag<HardEvent::V_MTE2>(event_id);
        CopyGmToUbufAlign(ub_in, src, n_rows_this_loop, n_cols_this_loop, src_stride - n_cols_this_loop, ubuf_gap);
        SetFlag<HardEvent::MTE2_V>(event_id);
        WaitFlag<HardEvent::MTE2_V>(event_id);
        Vadds(ub_adds, ub_in, offset, repeat, 1, 1, 8, 8);
        SetFlag<HardEvent::V_MTE2>(event_id);

        PipeBarrier<PIPE_V>();
        Vconv(ub_adds_f32, ub_adds, repeat, 1, 1, 8, 8);
        SetFlag<HardEvent::V_MTE2>(event_id);

        WaitFlag<HardEvent::V_MTE3>(event_id);
        CopyUbufToGmAlign(dst, ub_out, n_rows_this_loop, n_cols_this_loop, dst_stride - n_cols_this_loop);
        SetFlag<HardEvent::MTE3_V>(event_id);
    }

private:
    static constexpr uint8_t repeat = 153;
    __ubuf__ bfloat16_t *ub_out0 = reinterpret_cast<__ubuf__ bfloat16_t *>((uintptr_t)0);
    __ubuf__ bfloat16_t *ub_out1 = reinterpret_cast<__ubuf__ bfloat16_t *>((uintptr_t)19584);
    __ubuf__ float32_t *ub_adds_f32 = reinterpret_cast<__ubuf__ float32_t *>((uintptr_t)39936);
    __ubuf__ int32_t *ub_in0 = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)79104);
    __ubuf__ int32_t *ub_in1 = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)118272);
    __ubuf__ int32_t *ub_adds = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)157440);
    __ubuf__ float32_t *ub_muls = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)157440);

    bool is_ping = false;
};

template <>
class LoopDequanter<QuantGranularity::PER_CHANNEL> {
public:
    static constexpr int32_t max_len = 8192;
    inline __aicore__ LoopDequanter() = default;
    inline __aicore__ void SetForLoop()
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    inline __aicore__ void WaitForLoop()
    {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    inline __aicore__ void Loop(__gm__ bfloat16_t *dst, __gm__ int32_t *src, float32_t scale,
            int32_t n_rows_this_loop, int32_t n_cols_this_loop, int32_t src_stride, int32_t dst_stride)
    {
        is_ping = !is_ping;
        auto ub_in = is_ping ? ub_in0 : ub_in1;
        auto ub_out = is_ping ? ub_out0 : ub_out1;
        int32_t n_blocks = Block32B<bfloat16_t>::Count(n_cols_this_loop) * (sizeof(int32_t) / sizeof(bfloat16_t));
        int32_t ubuf_gap = n_blocks - Block32B<int32_t>::Count(n_cols_this_loop);

        WaitFlag<HardEvent::V_MTE2>(event_id);
        CopyGmToUbufAlign(ub_in, src, n_rows_this_loop, n_cols_this_loop, src_stride - n_cols_this_loop, ubuf_gap);
        SetFlag<HardEvent::MTE2_V>(event_id);
        WaitFlag<HardEvent::MTE2_V>(event_id);
        Vconv(ub_in_f32, ub_in, repeat, 1, 1, 8, 8);
        SetFlag<HardEvent::V_MTE2>(event_id);

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        if (scale_rows == 0 || scale_source != scale) {
            scale_rows = 1;
            scale_source = scale;
            CopyGmToUbufAlign(ub_scale, scale, 1, n_cols_this_loop, 0);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        for (; scale_rows < n_rows_this_loop; ++scale_rows) {
            CopyUB2UB(ub_scale + scale_rows * n_blocks * Block32B<float32_t>::size, ub_scale,
                0, 1, n_blocks, 0, 0);
        }
        PipeBarrier<PIPE_V>();
        Vmul(ub_mul, ub_in_f32, ub_scale, repeat, 1, 1, 1, 8, 8, 8);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        Vconv(ub_out, ub_mul, repeat, 1, 1, 4, 8, RoundMode::CAST_RINT);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyUbufToGmAlign(dst, ub_out, n_rows_this_loop, n_cols_this_loop, dst_stride - n_cols_this_loop);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

private:
    static constexpr uint8_t repeat = 128;
    __ubuf__ int32_t *ub_in0 = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)0);
    __ubuf__ float32_t *ub_mul = reinterpret_cast<__ubuf__ float32_t *>((uintptr_t)32768);
    __ubuf__ float32_t *ub_in_f32 = reinterpret_cast<__ubuf__ float32_t *>((uintptr_t)65536);
    __ubuf__ float32_t *ub_scale = reinterpret_cast<__ubuf__ float32_t *>((uintptr_t)98650);
    __ubuf__ bfloat16_t *ub_out = reinterpret_cast<__ubuf__ bfloat16_t *>((uintptr_t)131328);
    __ubuf__ int32_t *ub_in1 = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)163840);

    __gm__ float32_t *scale_source = nullptr;
    int32_t scale_rows = 0;
    bool is_ping = false;
};

template <typename T = half>
class LoopPerTokenDequanter {
public:
    static constexpr int32_t max_len = 8 * 32 / 4 * 128;

    inline __aicore__ LoopPerTokenDequanter(int32_t n0)
    {
        n_round = (n0 + 127) / 128 * 128;
        ub_in0 = reinterpret_cast<__ubuf__ T *>((uintptr_t)0);
        ub_in1 = reinterpret_cast<__ubuf__ T *>(ub_in0 + max_len);
        ub_out = reinterpret_cast<__ubuf__ T *>(ub_in1 + max_len);
        ub_scales = reinterpret_cast<__ubuf__ float32_t *>(ub_out + max_len);
        ub_in_f32 = reinterpret_cast<__ubuf__ float32_t *>(ub_scales + max_len);
        ub_out_f32 = reinterpret_cast<__ubuf__ float32_t *>(ub_in_f32 + max_len);
    }

    inline __aicore__ void SetForLoop()
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    inline __aicore__ void WaitForLoop()
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    inline __aicore__ void Loop(__gm__ T *buff, __gm__ float32_t *scale,
            int32_t n_rows_this_loop, int32_t n_cols_this_loop, int32_t stride)
    {
        is_ping = !is_ping;
        auto ub_in = is_ping ? ub_in0 : ub_in1;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;
        int32_t ubufGap = Block32B<T>::Count(n_round) - Block32B<T>::Count(n_cols_this_loop);
        WaitFlag<HardEvent::V_MTE2>(event_id);
        CopyGmToUbufAlign(ub_in, buff, n_rows_this_loop, n_cols_this_loop, stride - n_cols_this_loop, ubufGap);
        SetFlag<HardEvent::MTE2_V>(event_id);
        WaitFlag<HardEvent::MTE2_V>(event_id);
        WaitFlag<HardEvent::MTE2_V>(event_id);
        Vconv(ub_in_f32, ub_in, repeat, 1, 1, 8, 4);
        SetFlag<HardEvent::V_MTE2>(event_id);

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
        if (scale_source != scale) {
            scale_source = scale;
            CopyGmToUbufAlign(ub_scales, scale, 1, n_cols_this_loop, 0);
        }
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag
    }

}
