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

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t UB_SLOT_MAPPING_SIZE = 128 * 1024;

extern "C" __global__ __aicore__ void reshape_and_cache(
    __gm__ uint8_t * __restrict__ key_input_gm,
    __gm__ uint8_t * __restrict__ value_input_gm,
    __gm__ uint8_t * __restrict__ key_cache_gm,
    __gm__ uint8_t * __restrict__ value_cache_gm,
    __gm__ uint8_t * __restrict__ slot_mapping_input_gm,
    __gm__ uint8_t * __restrict__ key_output_gm,
    __gm__ uint8_t * __restrict__ value_output_gm,
    __gm__ uint8_t * __restrict__ tiling_para_gm)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> slot_mapping_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubuf;
    pipe.InitBuffer(slot_mapping_buf, UB_SLOT_MAPPING_SIZE);

    // tiling_param从gm搬运到ub
    AscendC::GlobalTensor<int32_t> tiling_param_gt;
    AscendC::LocalTensor<int32_t> tiling_param_lt = slot_mapping_buf.Get<int32_t>();
    tiling_param_gt.SetGlobalBuffer((__gm__ int32_t *)tiling_para_gm);
    DataCopy(tiling_param_lt, tiling_param_gt, 32);
    AscendC::PipeBarrier<PIPE_ALL>();
    __ubuf__ int32_t *tiling_param_ub = (__ubuf__ int32_t *)tiling_param_lt.GetPhyAddr();
    // 取tiling参数
    int32_t num_tokens = (int32_t)(*((__ubuf__ int32_t *)tiling_param_ub)); // 取地址解引用，结构体地址偏移再取值
    int32_t num_heads = (int32_t)(*((__ubuf__ int32_t *)tiling_param_ub + 1));
    int32_t head_size = (int32_t)(*((__ubuf__ int32_t *)tiling_param_ub + 2));
    int32_t block_size = (int32_t)(*((__ubuf__ int32_t *)tiling_param_ub + 4));

    int32_t token_size = num_heads * head_size;
    int32_t burst_len = token_size * 2 / BLOCK_SIZE; // fp16

    int32_t core_num = AscendC::GetBlockNum(); // 核数block_dim
    int32_t base_task_num = num_tokens / core_num; // 用token_nums去计算基础任务量，一个基础任务量代表一次k和一次v的搬运
    int32_t tail_task_num = num_tokens % core_num;

    int32_t core_idx = AscendC::GetBlockIdx(); // 当前在第几个核
    int32_t start_task_id = core_idx * base_task_num; // 当前kv table存放起始点

    // 任务分配不均的时候，前面的核需要多处理一个任务
    if (core_idx < tail_task_num) {
        base_task_num++;
        start_task_id += core_idx;
    } else {
        start_task_id += tail_task_num;
    }

    AscendC::GlobalTensor<int32_t> slot_mapping_input_gt;
    AscendC::GlobalTensor<half> key_input_gt;
    AscendC::GlobalTensor<half> key_cache_gt;
    AscendC::GlobalTensor<half> value_input_gt;
    AscendC::GlobalTensor<half> value_cache_gt;

    slot_mapping_input_gt.SetGlobalBuffer((__gm__ int32_t *)slot_mapping_input_gm);
    key_input_gt.SetGlobalBuffer((__gm__ half *)key_input_gm);
    key_cache_gt.SetGlobalBuffer((__gm__ half *)key_cache_gm);
    value_input_gt.SetGlobalBuffer((__gm__ half *)value_input_gm);
    value_cache_gt.SetGlobalBuffer((__gm__ half *)value_cache_gm);

    pipe.InitBuffer(ubuf, token_size * 2);
    AscendC::LocalTensor<int32_t> slot_mapping_ub = slot_mapping_buf.Get<int32_t>();
    AscendC::LocalTensor<half> temp_ubuf = ubuf.Get<half>(); // 临时存放token

    uint32_t max_num_tokens = UB_SLOT_MAPPING_SIZE / 4; // 一次最多搬入max_num_tokens个slot_id
    for (uint32_t i = 0; i < base_task_num; i++) {
        int32_t slot_ub_offset = i % max_num_tokens;
        if (slot_ub_offset == 0) {
            // slot_mapping从gm搬运到ub
            uint64_t slot_gm_offset = static_cast<uint64_t>(start_task_id + i);
            int32_t slot_copy_num = base_task_num - i < max_num_tokens ? base_task_num - i : max_num_tokens;
            DataCopy(slot_mapping_ub, slot_mapping_input_gt[slot_gm_offset],
                     {1, static_cast<uint16_t>((slot_copy_num * sizeof(uint32_t) + 31) / 32), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        uint64_t start = (i + start_task_id) * token_size;
        int64_t slot_value = (int64_t)slot_mapping_ub.GetValue(slot_ub_offset);
        if (slot_value < 0) {
            continue;
        }
        uint64_t blocks_id = static_cast<uint64_t>(slot_value) / block_size;
        uint64_t blocks_offset = static_cast<uint64_t>(slot_value) % block_size;
        uint64_t cache_start = blocks_id * block_size * token_size + blocks_offset * 16;

        AscendC::DataCopyParams copy_in_params = {1, static_cast<uint16_t>(burst_len), 0, 0};
        AscendC::DataCopyParams copy_out_params = {static_cast<uint16_t>(token_size / 16), 1, 0,
                                                   static_cast<uint16_t>(block_size - 1)};
        DataCopy(temp_ubuf, key_input_gt[start], copy_in_params);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataCopy(key_cache_gt[cache_start], temp_ubuf, copy_out_params);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataCopy(temp_ubuf, value_input_gt[start], copy_in_params);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataCopy(value_cache_gt[cache_start], temp_ubuf, copy_out_params);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
