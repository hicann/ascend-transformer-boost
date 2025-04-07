/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_INDEX_ADD_VALID_TILING_DATA
#define ASCEND_OPS_INDEX_ADD_VALID_TILING_DATA

#include <cstdint>

constexpr uint32_t MAX_VALUE_SIZE = 8 * 1024;
namespace AtbOps {
struct IndexAddValidTilingData {
    uint32_t indicesValid{1};
    uint32_t indicesTotal{1};
    uint32_t valueSize{1};
};
}
#endif