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
    constexpr int32_t ALLREDUCE_91093_EIGHT_RANK_FP16_UBMOVENUM_DEFALUT = 160;
    constexpr int32_t ALLREDUCE_91093_EIGHT_RANK_FP16_M0_DEFALUT = 128;
    constexpr int32_t ALLREDUCE_91093_EIGHT_RANK_FP16_PVALUE_DEFALUT = 14;
    constexpr int32_t ALLREDUCE_91093_EIGHT_RANK_FP16_COMMDATASPLIT_DEFALUT = 16;
    constexpr int32_t ALLREDUCE_91093_SIXTEEN_RANK_FP16_PVALUE_DEFALUT = 14;
    constexpr int32_t ALLREDUCE_91093_SIXTEEN_RANK_FP16_UBMOVENUM_DEFALUT = 160;
    constexpr int32_t ALLREDUCE_91093_SIXTEEN_RANK_FP16_M0_DEFALUT = 128;
    constexpr int32_t ALLREDUCE_91093_SIXTEEN_RANK_FP16_COMMDATASPLIT_DEFALUT = 16;
    
    static std::map<int, std::vector<std::vector<int>>> g_allreduce91093EightRankFP16CommdatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduce91093EightRankFP16PvalueMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduce91093EightRankFP16M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduce91093EightRankFP16UbmovenumMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduce91093SixteenRankFP16CommdatasplitMap = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduce91093SixteenRankFP16M0Map = {
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduce91093SixteenRankFP16UbmovenumMap = {
    };
    
    static std::map<int, std::vector<std::vector<int>>> g_allreduce91093SixteenRankFP16PvalueMap = {
    };

    void AllReduceNPU91093EightRankFP16Tiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TillingValue> tillingParamMap = {
            {&cocTilingData.commDataSplit,
             {ALLREDUCE_91093_EIGHT_RANK_FP16_COMMDATASPLIT_DEFALUT,
             g_allreduce91093EightRankFP16CommdatasplitMap}},
            {&cocTilingData.pValue,
             {ALLREDUCE_91093_EIGHT_RANK_FP16_PVALUE_DEFALUT,
             g_allreduce91093EightRankFP16PvalueMap}},
            {&cocTilingData.m0,
             {ALLREDUCE_91093_EIGHT_RANK_FP16_M0_DEFALUT,
             g_allreduce91093EightRankFP16M0Map}},
            {&cocTilingData.ubMoveNum,
             {ALLREDUCE_91093_EIGHT_RANK_FP16_UBMOVENUM_DEFALUT,
             g_allreduce91093EightRankFP16UbmovenumMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {DEFAULT_SWIZZLE_COUNT}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}}
        };
        SetTillingParam(cocTilingData, tillingParamMap);

        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
        cocTilingData.commNpuSplit =
                cocTilingData.commDataSplit == COMMDATASPLIT_ONE ? cocTilingData.rankSize : COMMNPUSPLIT_ONE;
        SetSecondCoreSplitTling(cocTilingData);
    }

    void AllReduceNPU91093SixteenRankFP16Tiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TillingValue> tillingParamMap = {
            {&cocTilingData.commDataSplit,
             {ALLREDUCE_91093_SIXTEEN_RANK_FP16_COMMDATASPLIT_DEFALUT,
             g_allreduce91093SixteenRankFP16CommdatasplitMap}},
            {&cocTilingData.pValue,
             {ALLREDUCE_91093_SIXTEEN_RANK_FP16_PVALUE_DEFALUT,
             g_allreduce91093SixteenRankFP16PvalueMap}},
            {&cocTilingData.m0,
             {ALLREDUCE_91093_SIXTEEN_RANK_FP16_M0_DEFALUT,
             g_allreduce91093SixteenRankFP16M0Map}},
            {&cocTilingData.ubMoveNum,
             {ALLREDUCE_91093_SIXTEEN_RANK_FP16_UBMOVENUM_DEFALUT,
             g_allreduce91093SixteenRankFP16UbmovenumMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {DEFAULT_SWIZZLE_COUNT}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}}
        };
        SetTillingParam(cocTilingData, tillingParamMap);

        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
        cocTilingData.commNpuSplit =
                cocTilingData.commDataSplit == COMMDATASPLIT_ONE ? cocTilingData.rankSize : COMMNPUSPLIT_ONE;
        SetSecondCoreSplitTling(cocTilingData);
    }
}