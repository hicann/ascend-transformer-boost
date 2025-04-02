/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_ELEWISE_CAST_TILING_H
#define ASCEND_OPS_ELEWISE_CAST_TILING_H

#include <unordered_map>
#include <mki/types.h>
#include <mki/kernel_info.h>
#include <mki/launch_param.h>
#include <mki/utils/status/status.h>

namespace AtbOps {
using namespace Mki;

enum RoundMode : int32_t {
    CAST_NONE = 0,
    CAST_RINT, // round
    CAST_FLOOR,
    CAST_CEIL,
    CAST_ROUND, // away-zero
    CAST_TRUNC, // to-zero
    CAST_ODD,   // Von Neumann rounding
};

enum TransKey : int32_t {
    HALF_TO_FLOAT = 17,
    HALF_TO_UINT8,
    HALF_TO_INT8,
    HALF_TO_INT16,
    HALF_TO_INT32,

    FLOAT_TO_HALF = 33,
    FLOAT_TO_FLOAT,
    FLOAT_TO_INT16,
    FLOAT_TO_INT32,
    FLOAT_TO_INT64,
    FLOAT_TO_BF16,

    UINT8_TO_HALF = 49,

    INT8_TO_HALF = 65,
    INT32_TO_HALF = 97,
    INT32_TO_INT64 = 99,
    INT64_TO_INT32 = 114,

    BF16_TO_FLOAT = 130
};

const std::unordered_map<TensorDType, int64_t> DATA_SIZE_MAP{
    {TENSOR_DTYPE_FLOAT16, 2}, {TENSOR_DTYPE_FLOAT, 4}, {TENSOR_DTYPE_UINT8, 1}, {TENSOR_DTYPE_INT8, 1},
    {TENSOR_DTYPE_INT16, 2},   {TENSOR_DTYPE_INT32, 4}, {TENSOR_DTYPE_INT64, 8}, {TENSOR_DTYPE_BF16, 2},
};

Status CastCommonTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AtbOps

#endif