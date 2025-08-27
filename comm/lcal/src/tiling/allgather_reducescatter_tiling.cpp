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
#include <iostream>
#include "tiling.h"
#include "tiling_func.h"
#include "lcoc_func.h"

#define TILING_MAP std::<map<int, std::vector<std::vector<int>>>
namespace Lcal {
constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_SWIZZLECOUNT_DEFAULT = 11;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16SwizzlecountMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_UBMOVENUM_DEFAULT = 40;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16UbmovenumMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_LENPERLOOPMULT_DEFAULT = 400;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16LenperloopmultMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_COMMNPUSPLIT_DEFAULT = 8;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16CommnpusplitMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_COMMDATASPLIT_DEFAULT = 1;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16CommdatasplitMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRAUBMOVENUM_DEFAULT = 12;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16ExtraubmovenumMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRALENPERLOOPMULT_DEFAULT = 4;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16ExtralenperloopmultMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRACOMMNPUSPLIT_DEFAULT = 1;

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRACOMMDATASPLIT_DEFAULT = 8;

// 821
constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_SWIZZLECOUNT_DEFAULT = 5;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16SwizzlecountMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_UBMOVENUM_DEFAULT = 60;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16UbmovenumMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_LENPERLOOPMULT_DEFAULT = 400;

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMNPUSPLIT_DEFAULT = 8;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16CommnpusplitMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMDATASPLIT_DEFAULT = 1;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16CommdatasplitMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMDIRECT_DEFAULT = 1;

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRAUBMOVENUM_DEFAULT = 20;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16ExtraubmovenumMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRALENPERLOOPMULT_DEFAULT = 2;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16ExtralenperloopmultMap = {};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRACOMMNPUSPLIT_DEFAULT = 1;

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRACOMMDATASPLIT_DEFAULT = 8;

// 281
constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_SWIZZLECOUNT_DEFAULT = 11;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16SwizzlecountMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_UBMOVENUM_DEFAULT = 10;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16UbmovenumMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_LENPERLOOPMULT_DEFAULT = 400;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16LenperloopmultMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMNPUSPLIT_DEFAULT = 1;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMDATASPLIT_DEFAULT = 8;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRAUBMOVENUM_DEFAULT = 20;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16ExtraubmovenumMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRALENPERLOOPMULT_DEFAULT = 2;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16ExtralenperloopmultMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRACOMMNPUSPLIT_DEFAULT = 8;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16ExtracommnpusplitMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRACOMMDATASPLIT_DEFAULT = 2;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16ExtracommdatasplitMap = {};

// 280
constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_SWIZZLECOUNT_DEFAULT = 9;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16SwizzlecountMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_UBMOVENUM_DEFAULT = 40;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16UbmovenumMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_LENPERLOOPMULT_DEFAULT = 400;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMNPUSPLIT_DEFAULT = 1;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMDATASPLIT_DEFAULT = 8;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMDIRECT_DEFAULT = 0;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16CommdirectMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRAUBMOVENUM_DEFAULT = 60;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16ExtraubmovenumMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRALENPERLOOPMULT_DEFAULT = 2;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16ExtralenperloopmultMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRACOMMNPUSPLIT_DEFAULT = 8;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16ExtracommnpusplitMap = {};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRACOMMDATASPLIT_DEFAULT = 1;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16ExtracommdatasplitMap = {};

const int PVALE_ONE = 1;
const int M0_DEFAULT = 128;
const int K0_DEFAULT = 256;
const int N0_DEFAULT = 256;
const int SWIZZLEDIRECT_ONE = 1;

void AG8RS2FalseFP16Tiling(CoCTilingData &cocTilingData)
{
    std::map<int * TilingValue> tilingParamMap = {
        {&cocTilingData.swizzlCount,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_SWIZZLECOUNT_DEFAULT,
            g_allgatherEightReducescatterTwoFalseFP16SwizzlecountMap}},
        {&cocTilingData.ubMoveNum,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_UBMOVENUM_DEFAULT,
            g_allgatherEightReducescatterTwoFalseFP16UbmovenumMap}},
        {&cocTilingData.lenPerLoop,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_LENPERLOOPMULT_DEFAULT,
            g_allgatherEightReducescatterTwoFalseFP16LenperloopmultMap}},
        {&cocTilingData.commNpuSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_COMMNPUSPLIT_DEFAULT,
            g_allgatherEightReducescatterTwoFalseFP16CommnpusplitMap}},
        {&cocTilingData.commDataSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_COMMDATASPLIT_DEFAULT,
            g_allgatherEightReducescatterTwoFalseFP16CommdatasplitMap}},
        {&cocTilingData.extraUbMoveNum,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRAUBMOVENUM_DEFAULT.
            g_allgatherEightReducescatterTwoFalseFP16ExtraubmovenumMap}},
        {&cocTilingData.extraLenPerLoop,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRALENPERLOOPMULT_DEFAULT,
            g_allgatherEightReducescatterTwoFalseFP16ExtralenperloopmultMap}}
        {&cocTilingData.extraCommNpuSplit, {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRACOMMNPUSPLIT_DEFAULT}},
        {&cocTilingData.extraCommDataSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRACOMMDATASPLIT_DEFAULT}}};
    SetTilingParam2D(cocTilingData, tilingParamMap);
    return;
}

void AG8RS2TrueFP16Tiling(CoCTilingData &cocTilingData)
{
    std::map<int * TilingValue> tilingParamMap = {
        {&cocTilingData.swizzlCount,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_SWIZZLECOUNT_DEFAULT,
            g_allgatherEightReducescatterTwoTrueFP16SwizzlecountMap}},
        {&cocTilingData.ubMoveNum,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_UBMOVENUM_DEFAULT,
            g_allgatherEightReducescatterTwoTrueFP16UbmovenumMap}},
        {&cocTilingData.lenPerLoop,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_LENPERLOOPMULT_DEFAULT}},
        {&cocTilingData.commNpuSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMNPUSPLIT_DEFAULT,
            g_allgatherEightReducescatterTwoTrueFP16CommnpusplitMap}},
        {&cocTilingData.commDataSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMDATASPLIT_DEFAULT,
            g_allgatherEightReducescatterTwoTrueFP16CommdatasplitMap}},
        {&cocTilingData.commDirect, {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMDIRECT_DEFAULT}},
        {&cocTilingData.extraUbMoveNum,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRAUBMOVENUM_DEFAULT.
            g_allgatherEightReducescatterTwoTrueFP16ExtraubmovenumMap}},
        {&cocTilingData.extraLenPerLoop,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRALENPERLOOPMULT_DEFAULT,
            g_allgatherEightReducescatterTwoTrueFP16ExtralenperloopmultMap}}
        {&cocTilingData.extraCommNpuSplit, {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRACOMMNPUSPLIT_DEFAULT}},
        {&cocTilingData.extraCommDataSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRACOMMDATASPLIT_DEFAULT}}};
    SetTilingParam2D(cocTilingData, tilingParamMap);
    return;
}

void AG2RS8TrueFP16Tiling(CoCTilingData &cocTilingData)
{
    std::map<int * TilingValue> tilingParamMap = {
        {&cocTilingData.swizzlCount,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_SWIZZLECOUNT_DEFAULT,
            g_allgatherTwoReducescatterEightTrueFP16SwizzlecountMap}},
        {&cocTilingData.ubMoveNum,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_UBMOVENUM_DEFAULT,
            g_allgatherTwoReducescatterEightTrueFP16UbmovenumMap}},
        {&cocTilingData.lenPerLoop,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_LENPERLOOPMULT_DEFAULT,
            g_allgatherTwoReducescatterEightTrueFP16LenperloopmultMap}},
        {&cocTilingData.commNpuSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMNPUSPLIT_DEFAULT}},
        {&cocTilingData.commDataSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMDATASPLIT_DEFAULT}},
        {&cocTilingData.commDirect, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMDIRECT_DEFAULT}},
        {&cocTilingData.extraUbMoveNum,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRAUBMOVENUM_DEFAULT.
            g_allgatherTwoReducescatterEightTrueFP16ExtraubmovenumMap}},
        {&cocTilingData.extraLenPerLoop,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRALENPERLOOPMULT_DEFAULT,
            g_allgatherTwoReducescatterEightTrueFP16ExtralenperloopmultMap}},
        {&cocTilingData.extraCommNpuSplit, 
            {DIM_EIGHT, g_allgatherTwoReducescatterEightTrueFP16ExtracommnpusplitMap}},
        {&cocTilingData.extraCommDataSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRACOMMDATASPLIT_DEFAULT,
            g_allgatherTwoReducescatterEightTrueFP16ExtracommdatasplitMap}}};
    SetTilingParam2D(cocTilingData, tilingParamMap);
    return;
}

void AG2RS8FalseFP16Tiling(CoCTilingData &cocTilingData)
{
    std::map<int * TilingValue> tilingParamMap = {
        {&cocTilingData.swizzlCount,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_SWIZZLECOUNT_DEFAULT,
            g_allgatherTwoReducescatterEightFalseFP16SwizzlecountMap}},
        {&cocTilingData.ubMoveNum,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_UBMOVENUM_DEFAULT,
            g_allgatherTwoReducescatterEightFalseFP16UbmovenumMap}},
        {&cocTilingData.lenPerLoop, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_LENPERLOOPMULT_DEFAULT}},
        {&cocTilingData.commNpuSplit, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMNPUSPLIT_DEFAULT}},
        {&cocTilingData.commDataSplit, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMDATASPLIT_DEFAULT}},
        {&cocTilingData.commDirect, 
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMDIRECT_DEFAULT,
            g_allgatherTwoReducescatterEightFalseFP16CommdirectMap}},
        {&cocTilingData.extraUbMoveNum,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRAUBMOVENUM_DEFAULT.
            g_allgatherTwoReducescatterEightFalseFP16ExtraubmovenumMap}},
        {&cocTilingData.extraLenPerLoop, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRALENPERLOOPMULT_DEFAULT}},
        {&cocTilingData.extraCommNpuSplit, 
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRACOMMNPUSPLIT_DEFAULT,
            g_allgatherTwoReducescatterEightFalseFP16ExtracommnpusplitMap}},
        {&cocTilingData.extraCommDataSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRACOMMDATASPLIT_DEFAULT,
            g_allgatherTwoReducescatterEightFalseFP16ExtracommdatasplitMap}}};
    SetTilingParam2D(cocTilingData, tilingParamMap);
    return;
}

void CoCAllgatherMatmulReduceScatterTilingFunc::GetDefaultTiling(const TaskParam &taskParam)
{
    CoCTilingFunc::GetDefaultTiling(taskParam);

    cocTilingData.swizzleDirect = SWIZZLEDIRECT_ONE;

    cocTilingData.m0 = M0_DEFAULT;
    cocTilingData.k0 = K0_DEFAULT;
    cocTilingData.n0 = N0_DEFAULT;

    cocTilingData.withSerialMode = 0;
    cocTilingData.is91093 = 0;
    cocTilingData.pValue = PVALE_ONE;
    cocTilingData.commDirect = 0;

    auto rsDim = taskParam.cocParamDesc.twoDimTPInfo.rsDim;
    auto agDim = taskParam.cocParamDesc.twoDimTPInfo.agDim;
    auto innerDimIsAg = taskParam.cocParamDesc.twoDimTPInfo.innerDimIsAg;
    if (agDim == DIM_EIGHT && rsDim == DIM_TWO && !innerDimIsAg) {
        AG8RS2FalseFP16Tiling(cocTilingData);
    } else if (agDim == DIM_EIGHT && rsDim == DIM_TWO && innerDimIsAg) {
        AG8RS2TrueFP16Tiling(cocTilingData);
    } else if (agDim == DIM_TWO && rsDim == DIM_EIGHT && innerDimIsAg) {
        AG2RS8TrueFP16Tiling(cocTilingData);
    } else {
        AG2RS8FalseFP16Tiling(cocTilingData);
    }
    cocTilingData.commNpuSplit = std::min(cocTilingData.commNpuSplit, agDim);
    cocTilingData.extraCommNpuSplit = std::min(cocTilingData.extraCommNpuSplit, rsDim);
}

bool CoCAllgatherMatmulReduceScatterTilingFunc::CheckTiling(const TaskParam &taskParam)
{
    if (!CoCTilingFunc::CheckTiling(taskParam)) {
        return false;
    }

    auto commNpuSplit = cocTilingData.commNpuSplit;
    auto commDataSplit = cocTilingData.commDataSplit;
    auto extraCommNpuSplit = cocTilingData.extraCommNpuSplit;
    auto extraCommDataSplit = cocTilingData.extraCommDataSplit;
    auto coreNum = cocTilingData.blockDim;
    auto useCoreCount = commNpuSplit * commDataSplit + extraCommNpuSplit * extraCommDataSplit;

    const int maxMValue = 200000;
    const int maxNValue = 32768;
    const int maxKValue = 32768;
    std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
        {"m", cocTilingData.m, PARAM_CHECK_MIN_VALUE_ONE, maxMValue},
        {"k", cocTilingData.k, PARAM_CHECK_MIN_VALUE_ONE, maxKValue},
        {"n", cocTilingData.n, PARAM_CHECK_MIN_VALUE_ONE, maxNValue},
        {"commNpuSplit * commDataSplit + extraCommNpuSplit * extraCommDataSplit",
        useCoreCount, PARAM_CHECK_MIN_VALUE_ONE, coreNum},
    };
    return CheckParamScopeList(paramCheckList);
}
}