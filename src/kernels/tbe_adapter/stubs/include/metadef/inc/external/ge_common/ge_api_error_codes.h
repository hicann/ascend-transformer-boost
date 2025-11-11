/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_STUB_GE_API_ERROR_CODES_H
#define ASCEND_OPS_STUB_GE_API_ERROR_CODES_H

#include <cstdint>
#include "graph/ascend_string.h"

namespace ge {
constexpr uint32_t SUCCESS = 0;
constexpr uint32_t FAILED = 1;
constexpr uint32_t PARAM_INVALID = 2;
} // namespace ge

#endif