/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_ROPE_Q_CONCAT_TILING_DATA
#define ASCEND_ROPE_Q_CONCAT_TILING_DATA

#include <cstdint>

namespace AtbOps {
struct RopeQConcatTilingData {
    uint32_t hiddenSizeQ; // hidden_size_q
    uint32_t headNumQ; // head_num
    uint32_t headDim; // head_dim
    uint32_t concatSize; // concat_size
    uint32_t rotaryCoeff; // 2
    uint32_t ntokens;
    uint32_t realCore; // 运行核数
    uint32_t nlCoreRun; // 前核处理行数
    uint32_t lCoreRun; // 尾核处理行数
    uint32_t maxNPerLoopForUb; // ub一次能处理最大行数
    uint32_t preCoreLoopTime; // 前核处理轮数
    uint32_t preCoreLoopNLast; // 前核最后一轮处理行数
    uint32_t lastCoreLoopTime; // 尾核处理轮数
    uint32_t lastCoreLoopNLast; // 前核最后一轮处理行数
};

}
#endif