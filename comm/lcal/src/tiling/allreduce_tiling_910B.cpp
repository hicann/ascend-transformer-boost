/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include "tiling_91093.h"
#include "tiling_func.h"

namespace Lcal {
    const int32_t ALLREDUCE_SERIAL_MODE_K_SIZE = 8192;
    const int32_t ALLREDUCE_SERIAL_MODE_MN_SIZE = 256 * 256 *12;

    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_UBMOVENUM_DEFAULT = 30;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_UBMOVENUM_DEFAULT = 30;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 30;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_UBMOVENUM_DEFAULT = 30;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_UBMOVENUM_DEFAULT = 30;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_TWO_RANK_INT8_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_TWO_RANK_INT8_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_TWO_RANK_INT8_UBMOVENUM_DEFAULT = 30;
    constexpr int32_t ALLREDUCE_TWO_RANK_INT8_M0_DEFAULT = 128;

    static std::vector<double> g_allreduceUbmovenumCoef = {
    };

    static std::vector<double> g_allreducePvalueCoef = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankInT8M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankInT8DatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankInT8PvalueMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankInT8UbmovenumMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16DatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16PvalueMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16UbmovenumMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16DatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16PvalueMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16UbmovenumMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8DatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8PvalueMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8UbmovenumMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16CommdatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16UbmovenumMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16SwizzldirectMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16SwizzlcountMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16PvalueMap = {
    };


}