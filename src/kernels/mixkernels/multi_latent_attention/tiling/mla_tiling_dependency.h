/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MLA_TILING_DEPENDENCY_H
#define MLA_TILING_DEPENDENCY_H

#include <cstdint>
#include <vector>
#include <mki/tensor.h>
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include <mki/utils/status/status.h>

namespace AtbOps {
using namespace Mki;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int32_t TILING_PARA_SIZE = 8;
constexpr int32_t TILING_PARA_SIZE_PREFILL = 27;
constexpr int32_t TILING_HEAD_SIZE_PREFILL = 19;
constexpr int32_t MAX_EMBEDDING = 576;
constexpr int32_t TILING_PARA_SIZE_TP1 = 7;
constexpr int32_t TILING_HEAD_SIZE = 18;
constexpr int32_t M_LIMIT = 128;
constexpr int32_t ND_BATCH_LIMIT = INT32_MAX;
constexpr int32_t FLOAT_LIMIT = 64;
constexpr int32_t BLOCK_LIMIT = 128 * 128;
constexpr int32_t WORKSPACE_BLOCK_SIZE_DB = 65536; // 128 * 256 * 2
constexpr int32_t DOUBLE_PING_PONG_SIZE = 32768 * 8;
constexpr int32_t LONG_SEQ_LEN = 128;
constexpr int32_t NORM_CMP_MASK_LEN = 512;
constexpr int32_t LONG_SEQ_ALIBI_LEN = 256;

enum class TilingKeyType {
    TILING_HALF_DATA = 0,
    TILING_BF16_DATA = 1,
    TILING_INT8_HALF_DATA = 2,
    TILING_INT8_BF16_DATA = 3
};


using PrefillTensor = struct PrefillTensor {
    Tensor query;
    Tensor queryRope;
    Tensor kCache;
    Tensor kCacheRope;
    Tensor vCache;
    Tensor mask;
    Tensor alibiCoeff;
    Tensor blockTable;
};

using AddrOffsets = struct AddressOffsetInfo {
    uint64_t addrQSeqOffset = 0;
    uint64_t addrKSeqOffset = 0;
    uint64_t addrVSeqOffset = 0;
    uint64_t addrOSeqOffset = 0;
    uint64_t addrOFdSeqOffset = 0;
    uint64_t addrLSeqOffset = 0;
    uint64_t addrMaskOffset = 0;
    int32_t totalQBlkNum = 0;
    int32_t block = 0;
};

struct BatchNode {
    int32_t batchIdx;
    int32_t kvSeqlen;
    BatchNode() {}
    BatchNode(int32_t batchIdxIn, int32_t kvSeqlenIn) : batchIdx(batchIdxIn), kvSeqlen(kvSeqlenIn) {}
    bool operator < (const BatchNode &other) const
    {
        return other.kvSeqlen > this->kvSeqlen;
    }
};

using MLAInfo = struct MLATilingParams {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t maxNumBlocksPerQuery = 0;
    float tor = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int32_t *kvSeqLen{nullptr};
    int32_t *qSeqLen{nullptr};
    int32_t maskType = 0;
    int32_t totalTaskNum = 0;
    int32_t splitKVNum = 0;
    int32_t tailBatch = 0;
    int32_t tailTaskNum = 0;
    int32_t flashDecodingTaskNum = 0;
    int32_t prevSplitNumSum = 0;
    int32_t normalTaskNum = 0;
    int32_t maxKvSeqLen = 0;
    int32_t maxSeqLen = 0;
    int32_t maskStride = 0;
    int32_t headStride = 0;
    int32_t blockDim = 0;
    int32_t windowSize = 0;
    PrefillTensor tensors;
    std::vector<BatchNode> batchList = {};
    bool quantFlag = false;
    bool mtpTp1Flag = false;
    bool kNz = false;
    bool flashDecoding = false;
    TilingKeyType type = TilingKeyType::TILING_HALF_DATA;
};
}

#endif
// MLA_TILING_DEPENDENCY_H
