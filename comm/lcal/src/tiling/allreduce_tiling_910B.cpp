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
#include "tiling_910B.h"
#include "tiling_func.h"
#include "lcal_types.h"

namespace Lcal {
    const int32_t ALLREDUCE_SERIAL_MODE_K_SIZE = 8192;
    const int64_t ALLREDUCE_SERIAL_MODE_MN_SIZE = 256 * 256 * 12;

    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_UBMOVENUM_DEFAULT = 30;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_UBMOVENUM_DEFAULT = 40;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_PVALUE_DEFAULT = 14;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 100;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_DATASPLIT_DEFAULT = 16;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_UBMOVENUM_DEFAULT = 100;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_PVALUE_DEFAULT = 14;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_DATASPLIT_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_PVALUE_DEFAULT = 6;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_SWIZZLCOUNT_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_SWIZZLDIRECT_DEFAULT = 0;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_UBMOVENUM_DEFAULT = 6;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT = 16;

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

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16UbmovenumMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16PvalueMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16DatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16DatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16UbmovenumMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16PvalueMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8DatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8PvalueMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8UbmovenumMap = {
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

    int32_t AllReduceUbMoveNum(int m, int k, int n)
    {
        double commPredict = 1.0 * (m / ONE_K) * (n / ONE_K) * (SECOND_TO_MS / ONE_K) / 40;
        double cubePredict = DOUBLE * m * k / B1_FLOP_PER_MS * n;
        double mknGB = (m / ONE_K) * (k / ONE_K) * (n / ONE_K);
        double mteTimePredict1 = GetMTETime(mknGB, DEFAULT_ROW, DEFAULT_COL);
        double mteTimePredict2 = GetMTETime(mknGB, DEFAULT_COL, DEFAULT_ROW);
        double mteTimePredict = std::min(mteTimePredict1, mteTimePredict2);
        double matmulPredict = std::max(cubePredict, mteTimePredict);
        double c0 = matmulPredict / commPredict;
        double c1 = 1.0 * m * n / k;
        double c2 = sqrt(c1);
        double c3 = sqrt(1.0 * m * n) / k;
        double c4 = c3 * c3;
        double c5 = matmulPredict;
        double c6 = commPredict;
        double c7 = 1.0 * n / m;
        double c8 = 1.0 * m * n / sqrt(k);
        double c9 = 1.0 * m * n * sqrt(k);
        double c10 = sqrt(1.0 * m * n) * k;
        double c11 = sqrt(1.0 * m * n * k);
        double c12 = sqrt(1.0 * m * n);
        double c13 = 1.0 * k * k / sqrt(1.0 * m * n);
        double c14 = 1.0 * k * k * sqrt(1.0 * m * n);
        double ubMoveNumDouble = 0;
        std::vector<double> featsUpdate = { c0, c1, c2, c3, c4, c5, c6, c7, 1.0 / c0, 1.0 / c1, 1.0 / c2, 1.0 / c3,
                                            1.0 / c4, c8, c9, c10, c11, c12, c13, 1.0 / c13, c14, 1 };
        for (uint32_t i = 0; i < featsUpdate.size(); i++) {
            ubMoveNumDouble += featsUpdate[i] * g_allreduceUbmovenumCoef[i];
        }

        return std::min(std::max(static_cast<int32_t>(ubMoveNumDouble) * HALF_KBYTE, MIN_UB_MOVE_NUM), MAX_UB_NUM);
    }

    int32_t AllReducePValue(int m, int k, int n)
    {
        double commPredict = 1.0 * (m / ONE_K) * (n / ONE_K) * (SECOND_TO_MS / ONE_K) / 40;
        double cubePredict = DOUBLE * m * k / B1_FLOP_PER_MS * n;
        double mknGB = (m / ONE_K) * (k / ONE_K) * (n / ONE_K);
        double mteTimePredict1 = GetMTETime(mknGB, DEFAULT_ROW, DEFAULT_COL);
        double mteTimePredict2 = GetMTETime(mknGB, DEFAULT_COL, DEFAULT_ROW);
        double mteTimePredict = std::min(mteTimePredict1, mteTimePredict2);
        double matmulPredict = std::max(cubePredict, mteTimePredict);
        double c0 = matmulPredict / commPredict;
        double c1 = 1.0 * m * n / k;
        double c2 = sqrt(c1);
        double c3 = sqrt(1.0 * m * n) / k;
        double c4 = c3 * c3;
        double c5 = matmulPredict;
        double c6 = commPredict;
        double c7 = 1.0 * n / m;
        double c8 = 1.0 * m * n / sqrt(k);
        double c9 = 1.0 * m * n * sqrt(k);
        double c10 = sqrt(1.0 * m * n) * k;
        double c11 = sqrt(1.0 * m * n * k);
        double c12 = sqrt(1.0 * m * n);
        double c13 = 1.0 * k * k / sqrt(1.0 * m * n);
        double c14 = 1.0 * k * k * sqrt(1.0 * m * n);
        double pValueDouble = 0;
        std::vector<double> featsUpdate = { c0, c1, c2, c3, c4, c5, c6, c7, 1.0 / c0, 1.0 / c1, 1.0 / c2, 1.0 / c3,
                                            1.0 / c4, c8, c9, c10, c11, c12, c13, 1.0 / c13, c14, 1 };
        for (uint32_t i = 0; i < featsUpdate.size(); i++) {
            pValueDouble += featsUpdate[i] * g_allreducePvalueCoef[i];
        }

        return std::min(std::max(static_cast<int32_t>(pValueDouble), 1), MAX_P_VALUE);
    }
}