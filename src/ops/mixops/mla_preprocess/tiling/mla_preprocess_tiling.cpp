/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "mla_preprocess_tiling.h"
#include "atbops/params/params.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/math.h"
#include "mki/utils/platform/platform_info.h"

#include <cmath>

constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t BASE_BLOCK_STEP = 2;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t L1AB_PINGPONG_BUFFER_LEN = 262144;
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 131072;
constexpr uint32_t L0C_SIZE = 128 * 1024;
constexpr uint32_t CONCAT_SIZE = 512;
constexpr uint32_t HIDDEN_STRATE = 7168;
constexpr uint32_t HIDDEN_STRATE_ROPE = 192;
constexpr uint32_t HIDDEN_STRATE_MM = 2112;
constexpr uint32_t HIDDEN_STRATE_RMS = 1536;
constexpr uint32_t UB_SIZE = 196352;
constexpr uint32_t HEADDIM = 64;

const int32_t NUM0 = 0;
const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM6 = 6;
const int32_t NUM7 = 7;
const int32_t NUM8 = 8;
const int32_t NUM9 = 9;
const int32_t NUM10 = 10;
const int32_t NUM11 = 11;
const int32_t NUM12 = 12;
const int32_t NUM13 = 13;
const int32_t NUM14 = 14;
const int32_t NUM15 = 15;
const int32_t NUM16 = 16;
const int32_t NUM17 = 17;
const int32_t NUM18 = 18;
const int32_t NUM19 = 19;
const int32_t NUM20 = 20;
const int32_t NUM21 = 21;
const int32_t NUM22 = 22;
const int32_t NUM23 = 23;
const int32_t NUM24 = 24;
const int32_t NUM25 = 25;
const int32_t NUM26 = 26;
const int32_t NUM27 = 27;
const int32_t NUM28 = 28;
const int32_t NUM29 = 29;
const int32_t NUM30 = 30;
const int32_t NUM31 = 31;
const int32_t NUM32 = 32;
const int32_t NUM33 = 33;
const int32_t NUM34 = 34;
const int32_t NUM35 = 35;
const int32_t NUM36 = 36;
const int32_t NUM37 = 37;
const int32_t NUM38 = 38;
const int32_t NUM39 = 39;
const int32_t NUM40 = 40;
const int32_t NUM41 = 41;
const int32_t NUM42 = 42;
const int32_t NUM43 = 43;
const int32_t NUM44 = 44;
const int32_t NUM45 = 45;
const int32_t NUM46 = 46;
const int32_t NUM47 = 47;
const int32_t NUM48 = 48;
const int32_t NUM49 = 49;
const int32_t NUM50 = 50;
const int32_t NUM51 = 51;
const int32_t NUM52 = 52;
const int32_t NUM53 = 53;
const int32_t NUM54 = 54;
const int32_t NUM55 = 55;
const int32_t NUM56 = 56;
const int32_t NUM57 = 57;
const int32_t NUM58 = 58;
const int32_t NUM59 = 59;
const int32_t NUM60 = 60;
const int32_t NUM61 = 61;
const int32_t NUM62 = 62;
const int32_t NUM63 = 63;
const int32_t NUM64 = 64;

inline uint32_t CeilDiv(const uint32_t dividend, const uint32_t divisor)
{
    if (divisor == 0) {
        return UINT32_MAX;
    }
    return (dividend + divisor - 1) / divisor;
}

inline uint32_t RoundUp(const uint32_t val, const uint32_t align = 16)
{
    if (align == 0) {
        return 0;
    }
    return (val + align - 1) / align * align;
}

inline uint32_t RoundDown(const uint32_t val, const uint32_t align = 16)
{
    if (align == 0) {
        return 0;
    }
    return val / align * align;
}

template <typename T = uint32_t>
inline T Max(const T a, const T b)
{
    return a > b ? a : b;
}

template <typename T = uint32_t>
inline T Min(const T a, const T b)
{
    return a < b ? a : b;
}

void MmTilingData::SetBaseOp(const uint32_t numCore, const uint32_t newM0, const uint32_t newN0)
{
    m0 = newM0;
    n0 = newN0;
    mLoop = CeilDiv(mSize, m0);
    nLoop = CeilDiv(nSize, n0);
    coreLoop = bSize * mLoop * nLoop;
    if (mLoop == 1 && transB && coreLoop % numCore < numCore / NUM4 * NUM3) {
        uint32_t tmpM0 = RoundUp(mSize, CONST_16);
        uint32_t maxN0 = L0C_SIZE / (tmpM0 * sizeof(float));
        if (isInt8) {
            maxN0 = maxN0 < CONST_256 ? maxN0 : CONST_256;
        }
        uint32_t x = CeilDiv(nSize, numCore);
        uint32_t y = CeilDiv(x, maxN0);
        uint32_t tmpN0 = RoundUp(CeilDiv(x, y), CONST_16);
        uint32_t rqdL0cSize = tmpM0 * tmpN0 * sizeof(float);
        if (rqdL0cSize < L0C_SIZE && (tmpM0 + tmpN0) * CONST_256 * sizeof(uint16_t) < L1AB_PINGPONG_BUFFER_LEN) {
            m0 = tmpM0;
            n0 = tmpN0;
            nLoop = CeilDiv(nSize, n0);
            coreLoop = bSize * nLoop;
        }
    }
    blockDim = Min(coreLoop, numCore);
}

void MmTilingData::Swizzle()
{
    float minCost = mSize * kSize + kSize * nSize;
    for (uint32_t i = 1; i <= blockDim; ++i) {
        int c = static_cast<int32_t>((blockDim + i - 1) / i);
        float cost;
        // B0 + A < A0 + B
        if (i * n0 + mSize < m0 * c + nSize) {
            swizzleDirect = 1; // Nz
            cost = n0 * i + m0 * c;
            if (cost <= minCost) {
                minCost = cost;
                swizzleCount = i;
            }
        } else {
            swizzleDirect = 0; // Zn
            cost = m0 * i + n0 * c;
            if (cost < minCost) {
                minCost = cost;
                swizzleCount = i;
            }
        }
    }
}

namespace AtbOps {

using namespace Mki;

float GetCost(const MmTilingData* tilingData,
              const HardwareInfo& hwInfo,
              const uint32_t newM0,
              const uint32_t newN0,
              const bool transA,
              const bool transB)
{
    float aCoef = 1;
    float bCoef = 1;
    float bwCoef = static_cast<float>(hwInfo.l2Bw) / static_cast<float>(hwInfo.hbmBw);
    uint32_t mLoop = CeilDiv(tilingData->mSize, newM0);
    uint32_t nLoop = CeilDiv(tilingData->nSize, newN0);
    if (mLoop == 0 || nLoop == 0) {
        return __FLT_MAX__;
    }
    uint32_t rqdNumCore = tilingData->bSize * mLoop * nLoop;
    uint32_t blockDim = Min(rqdNumCore, hwInfo.numCore);
    uint32_t mOnce = blockDim < nLoop ? newM0 : blockDim / nLoop * newM0;
    uint32_t nOnce = blockDim < nLoop ? hwInfo.numCore * newN0 : tilingData->nSize;
    if (mOnce * tilingData->kSize * sizeof(uint16_t) > hwInfo.l2Size) {
        aCoef = bwCoef;
    }
    if (nOnce * tilingData->kSize * sizeof(uint16_t) > hwInfo.l2Size) {
        bCoef = bwCoef;
    }
    if (transA && newM0 % CONST_256 == 0) {
        aCoef *= NUM2;
    }
    if (!transB && newN0 % CONST_256 == 0) {
        bCoef *= NUM2;
    }
    return 1 / (aCoef * static_cast<float>(newN0)) + 1 / (bCoef * static_cast<float>(newM0));
}

void TilingFunc(MmTilingData* tParam, const HardwareInfo& hwInfo)
{
    bool priFlag = !(tParam->mSize < tParam->nSize);
    uint32_t roundBase = pow(2, ceil(log(CeilDiv(priFlag ? tParam->nSize : tParam->mSize, CONST_16)))) * CONST_16;
    uint32_t priAxes = RoundUp(priFlag ? tParam->mSize : tParam->nSize, CONST_16);
    uint32_t subAxes = RoundUp(priFlag ? tParam->nSize : tParam->mSize, roundBase);
    float minCost = __FLT_MAX__;
    uint32_t maxAxes0 = AXES_ALIGN_SIZE / sizeof(int8_t);
    if (!tParam->isInt8) {
        maxAxes0 = AXES_ALIGN_SIZE / sizeof(uint16_t);
    }
    uint32_t maxPriAxes0 = Min(maxAxes0, priAxes);
    uint32_t maxSubAxes0 = Min(maxAxes0, subAxes);
    for (uint32_t priAxes0 = CONST_16; priAxes0 <= maxPriAxes0; priAxes0 *= BASE_BLOCK_STEP) {
        for (uint32_t subAxes0 = CONST_16; subAxes0 <= maxSubAxes0; subAxes0 *= BASE_BLOCK_STEP) {
            if (priAxes0 * subAxes0 * sizeof(float) > hwInfo.l0CSize) {
                continue;
            }
            uint32_t newM0 = priFlag ? priAxes0 : subAxes0;
            uint32_t newN0 = priFlag ? subAxes0 : priAxes0;
            if (newN0 > CONST_256 && tParam->isInt8) {
                continue;
            }
            float cost = GetCost(tParam, hwInfo, newM0, newN0, tParam->transA, tParam->transB);
            if (cost < minCost) {
                minCost = cost;
                tParam->SetBaseOp(hwInfo.numCore, newM0, newN0);
            }
        }
    }
    tParam->Swizzle();
    uint32_t k0Max = static_cast<uint32_t>(static_cast<float>(L1AB_PINGPONG_BUFFER_LEN) /
                                           ((tParam->m0 + tParam->n0) * sizeof(uint16_t)));
    if (tParam->isInt8) {
        tParam->k0 = k0Max < CONST_512 ? RoundDown(k0Max, CONST_32) : RoundDown(k0Max, CONST_512);
    } else {
        tParam->k0 = k0Max < CONST_256 ? RoundDown(k0Max, CONST_16) : RoundDown(k0Max, CONST_256);
    }
    if (tParam->k0 > CONST_512) {
        tParam->k0 = RoundDown(tParam->k0, CONST_512);
    }
    tParam->kLoop = CeilDiv(tParam->kSize, tParam->k0);
}

void GetmmTiling(const HardwareInfo& hwInfo, MmTilingData* tilingData, uint32_t bSize,
                 uint32_t mSize, uint32_t kSize, uint32_t nSize)
{
    tilingData->bSize = bSize;
    tilingData->mSize = mSize;
    tilingData->kSize = kSize;
    tilingData->nSize = nSize;
    TilingFunc(tilingData, hwInfo);
    return;
}


class MlaPreprocessTiling {
public:
    MLATilingData tilingData;
    Mki::Status Init(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo);
    void RmsNormQuantTiling(OpParam::MlaPreprocess &param, const uint32_t &aicNum);
    void RopeConcatTiling(const OpParam::MlaPreprocess &param, const uint32_t &aicNum);
    void SetMMTiling(uint32_t *tilingParam, MmTilingData &mmtiling, const uint32_t &index);
    void SetTiling(uint32_t *tilingParam);
};

void MlaPreprocessTiling::RmsNormQuantTiling(OpParam::MlaPreprocess &param,
                                             const uint32_t &aicNum)
{
    tilingData.rmsNumCore1 = aicNum * NUM2;
    tilingData.rmsNumCol1 = HIDDEN_STRATE;
    tilingData.rmsNumRow1 = param.N;
    tilingData.rmsQuantMin1 = -CONST_128;
    tilingData.rmsNumCore2 = aicNum * NUM2;
    tilingData.rmsNumCol2 = HIDDEN_STRATE_MM;
    tilingData.rmsNumRow2 = param.N;
    tilingData.rmsQuantMin2 = -CONST_128;
}

void MlaPreprocessTiling::RopeConcatTiling(const OpParam::MlaPreprocess &param, const uint32_t &aicNum)
{
    uint32_t ntokens = param.N;
    uint32_t hiddenSizeQ = HEADDIM * param.headNum;
    uint32_t headDim = HEADDIM;
    uint32_t headNumQ = hiddenSizeQ / headDim;
    uint32_t concatSize = CONCAT_SIZE;
    uint32_t maxCore = aicNum * 2;
    uint32_t maxUbSize = UB_SIZE;
    uint32_t allHeadNum = ntokens * headNumQ;

    uint32_t tempCore = (allHeadNum + maxCore - 1) / maxCore;
    uint32_t realCore = (allHeadNum + tempCore - 1) / tempCore; // 实际运算核数
    uint32_t nlCoreRun = (allHeadNum + realCore - 1) / realCore; // 前核运算head数
    uint32_t lCoreRun =  allHeadNum - (realCore - 1) * nlCoreRun; // 尾核运算head数
    
    uint32_t dataTypeSize = 2;

    // 计算一次能搬几行 q 4+2、reverseq 4、neg 4、sin 4+2、cos 4+2  + concat 2
    uint32_t allSize = headDim * (3 * (4 + dataTypeSize) + 2 * 4) + concatSize * dataTypeSize;  // rope内部升精度计算
    uint32_t maxNPerLoopForUb = maxUbSize / allSize;  // ub每次能载入最大行数（包括所有计算数据）
    uint32_t preCoreLoopTime = (nlCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 前核循环次数
    uint32_t preCoreLoopNLast = nlCoreRun - (preCoreLoopTime - 1) * maxNPerLoopForUb;  // 前核最后一批处理数据行数
    uint32_t lastCoreLoopTime = (lCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 尾核循环次数
    uint32_t lastCoreLoopNLast = lCoreRun - (lastCoreLoopTime - 1) * maxNPerLoopForUb;  // 尾核最后一批处理数据行数

    tilingData.hiddenSizeQ = hiddenSizeQ;
    tilingData.headNumQ = headNumQ;
    tilingData.headDim = headDim;
    tilingData.concatSize = concatSize;
    tilingData.rotaryCoeff = NUM2;
    tilingData.ntokens = ntokens;
    tilingData.realCore = realCore;
    tilingData.nlCoreRun = nlCoreRun;
    tilingData.lCoreRun = nlCoreRun;
    tilingData.maxNPerLoopForUb = maxNPerLoopForUb;
    tilingData.preCoreLoopTime = preCoreLoopTime;
    tilingData.preCoreLoopNLast = preCoreLoopNLast;
    tilingData.lastCoreLoopTime = lastCoreLoopTime;
    tilingData.lastCoreLoopNLast = lastCoreLoopNLast;
}

void MlaPreprocessTiling::SetMMTiling(uint32_t *tilingParam, MmTilingData &mmtiling, const uint32_t &index)
{
    tilingParam[index + NUM0] = mmtiling.bSize;
    tilingParam[index + NUM1] = mmtiling.mSize;
    tilingParam[index + NUM2] = mmtiling.kSize;
    tilingParam[index + NUM3] = mmtiling.nSize;
    tilingParam[index + NUM4] = mmtiling.m0;
    tilingParam[index + NUM5] = mmtiling.k0;
    tilingParam[index + NUM6] = mmtiling.n0;
    tilingParam[index + NUM7] = mmtiling.mLoop;
    tilingParam[index + NUM8] = mmtiling.kLoop;
    tilingParam[index + NUM9] = mmtiling.nLoop;
    tilingParam[index + NUM10] = mmtiling.coreLoop;
    tilingParam[index + NUM11] = mmtiling.swizzleCount;
    tilingParam[index + NUM12] = mmtiling.enSplitK;
    tilingParam[index + NUM13] = mmtiling.blockDim;
}

void MlaPreprocessTiling::SetTiling(uint32_t *tilingParam)
{
    tilingParam[NUM0] = tilingData.n;
    tilingParam[NUM43] = tilingData.perTaskNum;
    tilingParam[NUM44] = tilingData.resTaskNum;
    tilingParam[NUM45] = tilingData.numCore;

    tilingParam[NUM46] = tilingData.rmsNumCore1;
    tilingParam[NUM47] = tilingData.rmsNumCol1;

    tilingParam[NUM48] = tilingData.rmsNumCore2;
    tilingParam[NUM49] = tilingData.rmsNumCol2;

    tilingParam[NUM50] = tilingData.hiddenSizeQ;
    tilingParam[NUM51] = tilingData.headNumQ;
    tilingParam[NUM52] = tilingData.headDim;
    tilingParam[NUM53] = tilingData.concatSize;
    tilingParam[NUM54] = tilingData.rotaryCoeff;
    tilingParam[NUM55] = tilingData.ntokens;
    tilingParam[NUM56] = tilingData.realCore;
    tilingParam[NUM57] = tilingData.nlCoreRun;
    tilingParam[NUM58] = tilingData.lCoreRun;
    tilingParam[NUM59] = tilingData.maxNPerLoopForUb;
    tilingParam[NUM60] = tilingData.preCoreLoopTime;
    tilingParam[NUM61] = tilingData.preCoreLoopNLast;
    tilingParam[NUM62] = tilingData.lastCoreLoopTime;
    tilingParam[NUM63] = tilingData.lastCoreLoopNLast;
}

Mki::Status MlaPreprocessTiling::Init(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo)
{
    OpParam::MlaPreprocess param = AnyCast<OpParam::MlaPreprocess>(launchParam.GetParam());
    auto inDtype = launchParam.GetInTensor(0).desc.dtype;
    const uint32_t &aicNum = Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_CUBE);
    tilingData.n = param.N;
    tilingData.numCore = aicNum;
    uint64_t tilingSize = kernelInfo.GetTilingSize();
    uint32_t *tilingParam = reinterpret_cast<uint32_t *>(kernelInfo.GetTilingHostAddr());
    HardwareInfo hwInfo;
    RmsNormQuantTiling(param, aicNum);
    RopeConcatTiling(param, aicNum);
    MmTilingData mmtiling;
    mmtiling.transA = false;
    mmtiling.transB = true;
    mmtiling.isInt8 = 1;
    GetmmTiling(hwInfo, &mmtiling, 1, param.N, HIDDEN_STRATE, HIDDEN_STRATE_MM);
    SetMMTiling(tilingParam, mmtiling, NUM1);
    mmtiling.transA = false;
    mmtiling.transB = true;
    mmtiling.isInt8 = 1;
    GetmmTiling(hwInfo, &mmtiling, 1, param.N, HIDDEN_STRATE_RMS, param.headNum * HIDDEN_STRATE_ROPE);
    SetMMTiling(tilingParam, mmtiling, NUM15);
    mmtiling.transA = false;
    mmtiling.transB = false;
    mmtiling.isInt8 = 0;
    GetmmTiling(hwInfo, &mmtiling, param.headNum, param.N, CONST_128, CONCAT_SIZE);
    SetMMTiling(tilingParam, mmtiling, NUM29);
    SetTiling(tilingParam);
    MKI_LOG_INFO << "tilingSize: " << tilingSize;
    for (int i = 0; i < NUM64; i++) {
        MKI_LOG_INFO <<i<< ": " << tilingParam[i];
    }
    uint64_t workSizeS1 = static_cast<uint64_t>(tilingData.n) *  HIDDEN_STRATE * sizeof(int8_t);
    uint64_t workSizeS2 = static_cast<uint64_t>(tilingData.n) * param.headNum * HIDDEN_STRATE_ROPE * sizeof(uint16_t);
    uint64_t workSizeS3 = static_cast<uint64_t>(tilingData.n) * HIDDEN_STRATE_MM * sizeof(uint16_t);
    uint64_t workSizeS4 = static_cast<uint64_t>(tilingData.n) *
                          std::max(param.headNum * HIDDEN_STRATE_ROPE, HIDDEN_STRATE_MM) *
                          sizeof(uint32_t);
    if (inDtype == TENSOR_DTYPE_BF16) {
        kernelInfo.GetScratchSizes() = {workSizeS1, workSizeS2, workSizeS3, workSizeS4};
    } else {
        kernelInfo.GetScratchSizes() = {workSizeS1, workSizeS2, workSizeS3};
    }
    kernelInfo.SetBlockDim(aicNum);
    return Mki::Status::OkStatus();
}

Mki::Status GetMLAProprecessTiling(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo)
{
    MlaPreprocessTiling mlaTiling;
    MKI_CHECK(mlaTiling.Init(launchParam, kernelInfo).Ok(), "Init MlaPreprocessTiling failed",
              return Mki::Status::FailStatus(Mki::ERROR_ATTR_INVALID_TYPE));
    return Status::OkStatus();
}
} // namespace AtbOps
