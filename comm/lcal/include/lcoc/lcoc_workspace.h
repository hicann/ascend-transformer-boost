/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_LCOC_WORKSPACE_H
#define LCAL_LCOC_WORKSPACE_H

#if !defined(__DAV_C220_VEC__) && !defined(__DAV_M200_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV__C310__)
#define __aicore__
#define GM_ADDR int64_t
#endif

struct LcalWorkspaceInfo {
    GM_ADDR gm_reducebuf{ 0 };
    GM_ADDR gm_a_align{ 0 };
    GM_ADDR gm_b_align{ 0 };
    GM_ADDR gm_accum{ 0 };
    GM_ADDR gm_formate_dequant_scale{ 0 };
    GM_ADDR gm_dequant_param{ 0 };

    GM_ADDR workspaceSize {0};
};

inline __aicore__ int32_t AlignUp(int32_t len, int32_t size)
{
    return (len + size -1) & ~(size - 1);
}

#if !defined(__DAV_C220_VEC__) && !defined(__DAV_M200_VEC__) && !defined(__DAV_C220_CUBE__) && !defined(__DAV__C310__)
inline uint64_t GetDequantWorkSpaceSize(Lcal::LcalType lcalType, int32_t withSerialMode, int32_t m, int32_t n,
    int32_t m0, int32_t n0, int32_t pValue, int32_t nLoop, int32_t rankSize, int32_t blockDim,
    int32_t maxOutputSize = -1)
    {
    constexpr int32_t TWO = 2;
    uint64_t dequantWorkSpaceSize = 0;
    if (withSerialMode > 0) {
        dequantWorkSpaceSize = (maxOutputSize == -1 ? m : maxOutputSize) * n * sizeof(int32_t);
    } else {
        if (lcalType == Lcal::LcalType::MATMUL_ALL_REDUCE) {
            dequantWorkSpaceSize = pValue * blockDim * m0 * n0 * TWO * sizeof(int32_t);
        } else {
            dequantWorkSpaceSize = (maxOutputSize == -1 ? m : maxOutputSize) * n * sizeof(int32_t);
        }
    }
    return dequantWorkSpaceSize;
}
#endif

inline __aicore__ LcalWorkspaceInfo GetLcalWorkspaceInfo(GM_ADDR gmWorkSpace, int32_t batchSize, int32_t m,
    int32_t k, int32_t n, int32_t mAlign, int32_t kAlign, int32_t nAlign, bool transa, bool transb,
    int32_t mmadSize, bool hasAAlign, bool hasBAlign, int32_t accumRankSize, bool hasAccum = false,
    uint64_t dequantWorkSpaceSize = 0, bool hasDequantParam = false, bool hasFormatDequantScale = false,
    bool isDeterministic = false,
    int32_t isMoe = false, int32_t is_alltoallvc = false,
    int32_t EP = 1, int32_t expertPerRank = 1, int32_t outputSize = -1)
{
    if (outputSize == -1) {
        outputSize = m;
    }
    constexpr int32_t ALIGN8 = 8;
    LcalWorkspaceInfo lcalWorkspaceInfo;
    lcalWorkspaceInfo.gm_reducebuf = gmWorkSpace;
    GM_ADDR workspaceOffset = gmWorkSpace;
    if (isDeterministic) {
        workspaceOffset += WORKSPACE_REDUCE_SIZE;
    }

    if (hasAAlign) {
        lcalWorkspaceInfo.gm_a_align = workspaceOffset;
        workspaceOffset += static_cast<uint64_t>(batchSize) * (transa ? k * mAlign : m * kAlign) * mmadSize;
    }

    if (hasBAlign) {
        lcalWorkspaceInfo.gm_b_align = workspaceOffset;
        workspaceOffset += static_cast<uint64_t>(batchSize) * (transb ? n * kAlign : k * nAlign) * mmadSize *
            (expertPerRank <= 0 ? 1 : expertPerRank);            
    }

    if (!isMoe && hasDequantParam) {
        lcalWorkspaceInfo.gm_dequant_param = workspaceOffset;
        workspaceOffset += sizeof(int32_t) * AlignUp(n, ALIGN8);
    }

    if (hasFormatDequantScale) {
        lcalWorkspaceInfo.gm_formate_dequant_scale = workspaceOffset;
        workspaceOffset += sizeof(float) * AlignUp(n, ALIGN8);
    }

    if (hasAccum) {
        lcalWorkspaceInfo.gm_accum = workspaceOffset;
        workspaceOffset += dequantWorkSpaceSize;
    }
    lcalWorkspaceInfo.workspaceSize = workspaceOffset;
    return lcalWorkspaceInfo;
}

#endif