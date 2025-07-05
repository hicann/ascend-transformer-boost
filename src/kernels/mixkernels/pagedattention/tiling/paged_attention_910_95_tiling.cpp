/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "paged_attention_910_95_tiling.h"
#include <mki/utils/log/log.h>
#include <mki/launch_param.h>
#include "atbops/params/params.h"
#include "tbe_tiling_runner.h"

namespace AtbOps {
using namespace Mki;

Status PagedAttentionBaseAscend91095Tiling(const std::string &kernelName, const LaunchParam &launchParam,
                                           KernelInfo &kernelInfo, const Mki::BinHandle &binHandle)
{
    const TensorDesc &queryTensorDesc = launchParam.GetInTensor(0).desc;
    const TensorDesc &blockTables = launchParam.GetInTensor(13).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;

    const auto &param = Mki::AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
    std::vector<int64_t> kvSeqLen;
    for (const auto &s : param.kvSeqLen) {
        kvSeqLen.push_back(static_cast<int64_t>(s));
    }
    int32_t blockSize = launchParam.GetInTensor(1).desc.dims.at(1);

    auto runner =
        AsdOpsGeRt::TbeTilingRunner()
            .SetName("FusedInferAttentionScore")
            .SetKernelName(kernelName)
            .AddInput(queryTensorDesc.dtype, queryTensorDesc.format, queryTensorDesc.dims) // query
            .AddInTensorList(&launchParam.GetInTensor(1), launchParam.GetInputLen(1))      // key
            .AddInTensorList(&launchParam.GetInTensor(2), launchParam.GetInputLen(2))      // value
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {})      // pse_shift
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {})      // atten_mask
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {})      // actual_seq_lengths
            .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(kvSeqLen.size())},
                           kvSeqLen.data(), kvSeqLen.size() * sizeof(int64_t))        // actual_seq_lengths_kv
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // dequant_scale1
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // quant_scale1
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // dequant_scale2
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // quant_scale2
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // quant_offset2
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // antiquant_scale
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // antiquant_offset
            .AddInput(blockTables.dtype, blockTables.format, blockTables.dims)        // block_table
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // query_padding_size
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // kv_padding_size
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // key_antiquant_scale
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // key_antiquant_offset
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // value_antiquant_scale
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // value_antiquant_offset
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // key_shared_prefix
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // value_shared_prefix
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // actual_shared_prefix_len
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // query_rope
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // key_rope
            .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
            .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
            .AddAttrInt(param.headSize) // num_heads
            .AddAttrFloat(param.tor)    // scale
            .AddAttrInt(2147483647)     // pre_tokens
            .AddAttrInt(2147483647)     // next_tokens
            .AddAttrStr("BSND")         // input_layout
            .AddAttrInt(param.kvHead)   // num_key_value_heads
            .AddAttrInt(0)              // sparse_mode
            .AddAttrInt(1)              // inner_precise
            .AddAttrInt(blockSize)      // block_size
            .AddAttrInt(0)              // antiquant_mode
            .AddAttrBool(false)         // softmax_lse_flag
            .AddAttrInt(0)              // key_antiquant_mode
            .AddAttrInt(0);             // value_antiquant_mode

    return AsdOps::GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status PagedAttentionW8A16Ascend91095Tiling(const std::string &kernelName, const LaunchParam &launchParam,
                                            KernelInfo &kernelInfo, const Mki::BinHandle &binHandle)
{
    const TensorDesc &queryTensorDesc = launchParam.GetInTensor(0).desc;
    const TensorDesc &blockTables = launchParam.GetInTensor(13).desc;
    const TensorDesc &kDescale = launchParam.GetInTensor(16).desc;
    const TensorDesc &vDescale = launchParam.GetInTensor(18).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;

    const auto &param = Mki::AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
    std::vector<int64_t> kvSeqLen;
    for (const auto &s : param.kvSeqLen) {
        kvSeqLen.push_back(static_cast<int64_t>(s));
    }
    int32_t blockSize = launchParam.GetInTensor(1).desc.dims.at(1);

    auto runner =
        AsdOpsGeRt::TbeTilingRunner()
            .SetName("FusedInferAttentionScore")
            .SetKernelName(kernelName)
            .AddInput(queryTensorDesc.dtype, queryTensorDesc.format, queryTensorDesc.dims) // query
            .AddInTensorList(&launchParam.GetInTensor(1), launchParam.GetInputLen(1))      // key
            .AddInTensorList(&launchParam.GetInTensor(2), launchParam.GetInputLen(2))      // value
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {})      // pse_shift
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {})      // atten_mask
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {})      // actual_seq_lengths
            .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(kvSeqLen.size())},
                           kvSeqLen.data(), kvSeqLen.size() * sizeof(int64_t))        // actual_seq_lengths_kv
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // dequant_scale1
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // quant_scale1
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // dequant_scale2
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // quant_scale2
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // quant_offset2
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // antiquant_scale
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // antiquant_offset
            .AddInput(blockTables.dtype, blockTables.format, blockTables.dims)        // block_table
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // query_padding_size
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // kv_padding_size
            .AddInput(kDescale.dtype, kDescale.format, kDescale.dims)                 // key_antiquant_scale
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // key_antiquant_offset
            .AddInput(vDescale.dtype, vDescale.format, vDescale.dims)                 // value_antiquant_scale
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // value_antiquant_offset
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // key_shared_prefix
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // value_shared_prefix
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // actual_shared_prefix_len
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // query_rope
            .AddInputPlaceHolder(TENSOR_DTYPE_UNDEFINED, TENSOR_FORMAT_UNDEFINED, {}) // key_rope
            .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
            .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
            .AddAttrInt(param.headSize) // num_heads
            .AddAttrFloat(param.tor)    // scale
            .AddAttrInt(2147483647)     // pre_tokens
            .AddAttrInt(2147483647)     // next_tokens
            .AddAttrStr("BSND")         // input_layout
            .AddAttrInt(param.kvHead)   // num_key_value_heads
            .AddAttrInt(0)              // sparse_mode
            .AddAttrInt(1)              // inner_precise
            .AddAttrInt(blockSize)      // block_size
            .AddAttrInt(0)              // antiquant_mode
            .AddAttrBool(false)         // softmax_lse_flag
            .AddAttrInt(0)              // key_antiquant_mode
            .AddAttrInt(0);             // value_antiquant_mode

    return AsdOps::GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AtbOps