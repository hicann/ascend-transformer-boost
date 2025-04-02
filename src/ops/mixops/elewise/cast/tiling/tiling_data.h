/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATBOPS_CAST_WIDE_TILING_DATA
#define ATBOPS_CAST_WIDE_TILING_DATA

#include <cstdint>

namespace AtbOps {
struct CastTilingData {
    uint32_t numTotal{0};
    uint32_t blockNum{0};       // 每个核处理的数据数量
    uint32_t blockTail{0};      // 每个核处理的剩余数量
    uint32_t ubFactor{0};       // 每个核每次循环处理的数据数量
    uint32_t dataTransKey{0};   // 对应不同输入输出的数据类型
    uint32_t transMode{0};      // 转换的模式
};
}
#endif