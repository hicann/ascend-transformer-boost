/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_LCOC_ARGS_H
#define LCAL_LCOC_ARGS_H

#include <map>
#include <lcoc_base.h>
#include <lcal_types.h>

constexpr int64_t WORKSPACE_REDUCE_SIZE = 4000000;
#pragma once
namespace Lcal {
    const constexpr int32_t INT8_ELE_SIZE = 1;
    const constexpr int32_t FP_BF_16_ELE_SIZE = 2;
    constexpr uint32_t ALIGN_BYTES = 512;
    constexpr int32_t PARAM_CHECK_MAX_VALUE = -1;
    constexpr int32_t PARAM_CHECK_MIN_VALUE_ZERO = 0;
    constexpr int32_t PARAM_CHECK_MIN_VALUE_ONE = 1;
    constexpr int32_t INPUT_PARAM_DEFAULT_VALUE = -1;
    constexpr int32_t MAX_M_VALUE = 10000000;
    constexpr int32_t MAX_K_VALUE = 100000;
    constexpr int32_t MAX_N_VALUE = 100000;

    enum CoCDataTypeDesc : int {
        COC_DATA_TYPE_UNDEFINED = -1,
        FP16FP16_FP32_FP16 = 0,
        BF16BF16_FP32_BF16 = 1,
        INT8INT8_INT32_FP16 = 2,
        INT8INT8_INT32_BF16 = 3,
        FP16INT8_INT32_FP16 = 4,
        BF16INT8_INT32_BF16 = 5,
        FP16INT8_FP32_FP16 = 6,
        BF16INT8_FP32_BF16 = 7,
        FP16INT4_FP32_FP16 = 8,
        BF16INT4_FP32_BF16 = 9,
        COC_DATA_TYPE_DESC_MAX = 10,
    };

    const std::map<CoCDataTypeDesc, int32_t> COC_TYPE2ELE_SIZE = {
        { FP16FP16_FP32_FP16, FP_BF_16_ELE_SIZE }, { BF16BF16_FP32_BF16, FP_BF_16_ELE_SIZE },
        { INT8INT8_INT32_FP16, INT8_ELE_SIZE }, { INT8INT8_INT32_BF16, INT8_ELE_SIZE },
        { FP16INT8_INT32_FP16, INT8_ELE_SIZE }, { BF16INT8_INT32_BF16, INT8_ELE_SIZE },
        { FP16INT8_FP32_FP16, FP_BF_16_ELE_SIZE }, { BF16INT8_FP32_BF16, FP_BF_16_ELE_SIZE },
        { FP16INT4_FP32_FP16, FP_BF_16_ELE_SIZE }, { BF16INT4_FP32_BF16, FP_BF_16_ELE_SIZE }
    };

    const std::map<CoCDataTypeDesc, HcclDataType> COC_TYPE2HCCL_TYPE = {
        { FP16FP16_FP32_FP16, HCCL_DATA_TYPE_FP16 }, { BF16BF16_FP32_BF16, HCCL_DATA_TYPE_BFP16 },
        { INT8INT8_INT32_FP16, HCCL_DATA_TYPE_FP16 }, { INT8INT8_INT32_BF16, HCCL_DATA_TYPE_BFP16 },
        { FP16INT8_INT32_FP16, HCCL_DATA_TYPE_FP16 }, { BF16INT8_INT32_BF16, HCCL_DATA_TYPE_BFP16 },
        { FP16INT8_FP32_FP16, HCCL_DATA_TYPE_FP16 }, { BF16INT8_FP32_BF16, HCCL_DATA_TYPE_BFP16 },
        { FP16INT4_FP32_FP16, HCCL_DATA_TYPE_FP16 }, { BF16INT4_FP32_BF16, HCCL_DATA_TYPE_BFP16 }
    };

    struct CoCParamDesc {
        CoCDataTypeDesc dataTypeDesc = FP16FP16_FP32_FP16;
        MatMulInfo mmInfo = {};
        QuantInfo quantInfo = {};
        PostInfo postInfo = {};
        HcclReduceOp op = HCCL_REDUCE_SUM;
        TwoDimTPInfo twoDimTPInfo = {};
    };

    struct CoCInputPkg {
        void *matrixA = nullptr;
        void *matrixB = nullptr;
        void *bias = nullptr;
        void *gamma = nullptr;
        void *dequantScale = nullptr;
        void *dequantOffset = nullptr;

        void *quantScale = nullptr;
        void *quantOffset = nullptr;
    };

    struct CoCOutputPkg {
        void *output = nullptr;
        void *midOutput = nullptr;
    };

    struct TaskParam {
        int32_t rank = -1;
        int32_t rankSize = -1;
        int32_t blockDim = -1;
        int32_t bufferSize = -1;
        ChipName chipName = ChipName::CHIP_910B3;
        CoCParamDesc cocParamDesc = {};
        LcalType lcalType = LcalType::ALL_REDUCE;
    };
}
#endif