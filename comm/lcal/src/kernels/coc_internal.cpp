/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_COC_INTERNAL_H
#define LCAL_COC_INTERNAL_H

#include <type_traits>
#include "kernel_operator.h"
#include "coc_const_args.cce"

using namespace AscendC

template <typename T>

FORCE_INLINE_AICORE LocalTensor<T> CreateLocalTensor(__ubuf__ T *addr)
{
    LocalTensor<T> tensor;
    TBuffAddr taddr;
    taddr.bufferAddr = reinterpret_cast<uint64_t>(addr);
    tensor.SetAddr(taddr);
    return tensor;
}

FORCE_INLINE_AICORE LocalTensor<T> CreateLocalTensor(uint32_t buffer_offset)
{
    LocalTensor<T> tensor;
    tensor.address_.bufferAddr = buffer_offset;
    return tensor;
}

template <typename T>
FORCE_INLINE_AICORE LocalTensor<T> CreateLocalTensor(uint32_t buffer_offset, uint32_t logic_pos)
{
    LocalTensor<T> tensor;
    tensor.address_.logicPos = logic_pos;
    tensor.address_.bufferAddr = buffer_offset;
    return tensor;
}

template <typename T>
FORCE_INLINE_AICORE LocalTensor<T> CreateLocalTensor(__gm__ T *addr)
{
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(addr);
    return tensor;
}

template <pipe_t pipe>
inline __aicore__ void FFTSCrossCoreSync(uint64_t mode, uint64_t flag_id)
{
    uint64_t config = 1 | (mode << 4) | (flag_id << 8);
    ffts_cross_core_sync(pipe, config);
}

template <typename T>
inline __aicore__ void CopyUB2UB(__ubuf__ T *dst, __ubuf__ T *src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst,
                                 uint16_t srcStride, uint16_t dstStride)
{
    LocalTensor<T> srcTensor = CreateLocalTensor<T>(src);
    LocalTensor<T> dstTensor = CreateLocalTensor<T>(dst);
    DataCopyParams repeatParams(nBurst, lenBurst, srcStride, dstStride);
    DataCopy(dstTensor, srcTensor, repeatParams);
}

template <typename Tdst, typename Tsrc>
inline __aicore__ void Vconv(__ubuf__ Tdst *dst, __ubuf__ Tsrc *src, uint8_t repeat, uint16_t dstBlockStride,
                             uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                             const RoundMode &roundMode = RoundMode::CAST_NONE)
{
    LocalTensor<Tsrc> srcTensor = CreateLocalTensor<Tsrc>(src);
    GlobalTensor<Tdst> dstTensor = CreateLocalTensor<Tdst>(dst);
    UnaryRepeatParams repeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    Cast<Tdst, Tsrc, false>(dstTensor, srcTensor, roundMode, -1, repeat, repeatParams);
}

template <typename T>
inline __aicore__ void Vadd(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeat, uint16_t dstBlockStride,
                            uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
                            uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    LocalTensor<T> srcTensor0 = CreateLocalTensor<T>(src0);
    LocalTensor<T> srcTensor1 = CreateLocalTensor<T>(src1);
    LocalTensor<T> dstTensor = CreateLocalTensor<T>(dst);
    BinaryRepeatParams repeatParams(dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                    src1RepeatStride);
    Add<T, false>(dstTensor, srcTensor0, srcTensor1, -1, repeat, repeatParams);
}

template <typename T>
inline __aicore__ void Vadds(__ubuf__ T *dst, __ubuf__ T *src, const T &scalarValue, uint8_t repeat,
                            uint16_t dstBlockStride, uint8_t srcBlockStride, uint8_t dstRepeatStride,
                            uint8_t srcRepeatStride)
{
    LocalTensor<T> srcTensor = CreateLocalTensor<T>(src);
    LocalTensor<T> dstTensor = CreateLocalTensor<T>(dst);
    UnaryRepeatParams repeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride,
                                    src1RepeatStride);
    Adds<T, false>(dstTensor, srcTensor, srcTensor1, -1, repeat, repeatParams);
}

template <typename T>
inline __aicore__ void Vmul(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeat, uint16_t dstBlockStride,
                            uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
                            uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    LocalTensor<T> srcTensor0 = CreateLocalTensor<T>(src0);
    LocalTensor<T> srcTensor1 = CreateLocalTensor<T>(src1);
    LocalTensor<T> dstTensor = CreateLocalTensor<T>(dst);
    BinaryRepeatParams repeatParams(dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                    src1RepeatStride);
    Mul<T, false>(dstTensor, srcTensor0, srcTensor1, -1, repeat, repeatParams);
}

template <typename T>
inline __aicore__ void Vmuls(__ubuf__ T *dst, __ubuf__ T *src, const T &scalarValue, uint8_t repeat,
                            uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                            uint8_t srcRepeatStride)
{
    LocalTensor<T> srcTensor = CreateLocalTensor<T>(src);
    LocalTensor<T> dstTensor = CreateLocalTensor<T>(dst);
    UnaryRepeatParams repeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride,
                                    src1RepeatStride);
    Muls<T, false>(dstTensor, srcTensor, srcTensor1, -1, repeat, repeatParams);
}

inline __aicore__ bool IsQuant(const QuantGranularity &granularity)
{
    return (granularity > QuantGranularity::QUANT_GRANULARITY_UNDEFINED) &&
           (granularity < QuantGranularity::QUANT_GRANULARITY_PER_TENSOR);
}

#define COC_ARGS_FUN_IIO(T_INPUT1, T_INPUT2, T_OUTPUT) \
    __gm__ T_INPUT1 *gm_a, __gm__ T_INPUT2 *gm_b, __gm__ T_OUTPUT *gm_bias, __gm__ T_OUTPUT *gm_gamma, \
        __gm__ T_OUTPUT *gm_out, __gm__ T_OUTPUT *gm_allgather_out, GM_ADDR gm_workspace, \
        GM_ADDR gm_dequant_scale, GM_ADDR gm_dequant_offset, GM_ADDR gm_quant_scale, \
        GM_ADDR gm_quant_offset, GM_ADDR coc_comm_args, GM_ADDR ffts_addr, \
        __gm__ int32_t* num_local_tokens_per_expert, __gm__ int32_t *num_global_tokens_per_local_expert, \
        __gm__ int32_t *global_token_per_expert_matrix, GM_ADDR para_gm

#define COC_ARGS_FUN_IO(T_INPUT, T_OUTPUT) COC_ARGS_FUN_IIO(T_INPUT, T_INPUT, T_OUTPUT)

#define COC_ARGS_FUN(T) COC_ARGS_FUN_IO(T, T)

#define COC_ARGS_CALL() \
    gm_a, gm_b, gm_bias, gm_gamma, gm_out, gm_allgather_out, gm_workspace, gm_dequant_scale, gm_dequant_offset, \
    gm_quant_scale, gm_quant_offset, coc_comm_args, ffts_addr, \
    num_local_tokens_per_expert, num_global_tokens_per_local_expert, \
    global_token_per_expert_matrix, para_gm

#define COC_ARGS_CALL_INT8() \
    reinterpret_cast<GM_ADDR>(gm_a), reinterpret_cast<GM_ADDR>(gm_b), reinterpret_cast<GM_ADDR>(gm_bias), \
    reinterpret_cast<GM_ADDR>(gm_gamma), reinterpret_cast<GM_ADDR>(gm_out), \
    reinterpret_cast<GM_ADDR>(gm_allgather_out), gm_workspace, gm_dequant_scale, gm_dequant_offset, \
    gm_quant_scale, gm_quant_offset, coc_comm_args, ffts_addr, \
    num_local_tokens_per_expert, num_global_tokens_per_local_expert, \
    global_token_per_expert_matrix, para_gm

#define PP_MATMUL_AIC_ARGS_FUN(T_INPUT, T_OUTPUT) \
    GM_ADDR gm_a, GM_ADDR gm_b, __gm__ T_OUTPUT *gm_bias, __gm__ T_OUTPUT *gm_c, \
        __gm__ T_OUTPUT *gm_peer_mem, GM_ADDR gm_workspace, GM_ADDR gm_dequant_scale, \
        GM_ADDR gm_dequant_offset, int32_t batch_size, int32_t m, int32_t k, int32_t n, int32_t m0, \
        int32_t k0, int32_t n0, int32_t m_loop, int32_t k_loop, int32_t n_loop, int32_t core_loop, \
        int32_t swizzl_count, int32_t swizzl_direct, int32_t rank, int32_t rank_size, int32_t p_value, \
        int32_t withSerialMode, QuantGranularity quant_granularity, QuantGranularity dequant_granularity, \
        int32_t ag_dim, int32_t rs_dim, bool inner_dim_is_Ag, bool weight_nz, bool is_91093, \
        __gm__ int32_t *num_local_tokens_per_expert, __gm__ int32_t * num_global_tokens_per_local_expert, \
        __gm__ int32_t *global_tokens_per_expert_matrix, int32_t local_expert_nums, int32_t EP, int32_t TP, \
        int32_t maxOutputSize, int32_t is_moe, bool is_deterministic, int32_t buffer_size \

#define PP_MATMUL_AIC_ARGS_FUN() \
    reinterpret_cast<GM_ADDR>(gm_a), reinterpret_cast<GM_ADDR>(gm_b), gm_bias, gm_c, gm_peer_mem, \
        reinterpret_cast<GM_ADDR>(gm_workspace), reinterpret_cast<GM_ADDR>(gm_dequant_scale), \
        reinterpret_cast<GM_ADDR>(gm_dequant_offset), batch_size, m, k, n, m0, k0, n0, m_loop, k_loop, \
        n_loop, core_loop, swizzl_count, swizzl_direct, rank, rank_size, p_value, withSerialMode, quant_granularity, \
        dequant_granularity, ag_dim, rs_dim, inner_dim_is_Ag, weight_nz, is_91093, \
        num_local_tokens_per_expert, num_global_tokens_per_local_expert, \
        global_tokens_per_expert_matrix, local_expert_nums, EP, TP, maxOutputSize, is_moe, is_deterministic, buffer_size

#define PP_MATMUL_AIV_PADDING_ARGS_FUN()
    GM_ADDR gm_a, GM_ADDR gm_b, GM_ADDR gm_workspace, GM_ADDR gm_dequant_scale, \
        GM_ADDR gm_dequant_offset, GM_ADDR gm_quant_scale, GM_ADDR gm_quant_offset, \
        int32_t batch_size, int32_t m, int32_t k, int32_t n, bool trans_a, bool trans_b, bool is_int8, \
        
