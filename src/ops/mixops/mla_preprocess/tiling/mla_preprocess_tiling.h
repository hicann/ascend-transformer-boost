/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_MOE_GMM_TILING_H
#define ASCEND_OPS_MOE_GMM_TILING_H

#include <cstdint>
#include <mki/utils/status/status.h>
#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/platform/platform_info.h>
#include "tiling/tiling_api.h"

using namespace AscendC;
using namespace Mki;
namespace AtbOps {
Mki::Status GetMLAProprecessTiling(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo);
} // namespace AtbOps

struct HardwareInfo {
    uint32_t numCore{0};
    uint32_t l2Size{0};
    uint32_t l1Size{0};
    uint32_t l0aSize{0};
    uint32_t l0bSize{0};
    uint32_t l0CSize{0};
    uint32_t hbmBw{0};
    uint32_t l2Bw{0};
    HardwareInfo()
    {
        auto platform = PlatformInfo::Instance();
        numCore = platform.GetCoreNum(CoreType::CORE_TYPE_CUBE);
        l2Size = platform.GetL2Size();
        l1Size = platform.GetL1Size();
        l0aSize = platform.GetL0ASize();
        l0bSize = platform.GetL0BSize();
        l0CSize = platform.GetL0CSize();
        hbmBw = 1;
        l2Bw = 5; // 5x faster than hbm.
    }
};

struct MmTilingData {
    uint32_t bSize{0};
    uint32_t mSize{0};
    uint32_t kSize{0};
    uint32_t nSize{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    uint32_t mLoop{1};
    uint32_t kLoop{1};
    uint32_t nLoop{1};
    uint32_t coreLoop{1};
    uint32_t swizzleCount{1};
    uint32_t tilingKey{0};
    uint32_t blockDim{1};
    uint32_t swizzleDirect{0};
    uint32_t enSplitK{0};
    uint32_t transA{0};
    uint32_t transB{0};
    uint32_t isInt8{0};

    void SetBaseOp(const uint32_t numCore, const uint32_t newM0, const uint32_t newN0);
    void Swizzle();
};

struct MLATilingData {
    uint32_t numCore;
    // mm
    uint32_t n;
    uint32_t perTaskNum;
    uint32_t resTaskNum;

    uint32_t mm1batchSize;
    uint32_t mm1m;
    uint32_t mm1k;
    uint32_t mm1n;
    uint32_t mm1m0;
    uint32_t mm1k0;
    uint32_t mm1n0;
    uint32_t mm1mLoop;
    uint32_t mm1kLoop;
    uint32_t mm1nLoop;
    uint32_t mm1coreLoop;
    uint32_t mm1swizzleCnt;
    uint32_t mm1enShuffleK;
    uint32_t mm1blockDim;

    uint32_t mm2batchSize;
    uint32_t mm2m;
    uint32_t mm2k;
    uint32_t mm2n;
    uint32_t mm2m0;
    uint32_t mm2k0;
    uint32_t mm2n0;
    uint32_t mm2mLoop;
    uint32_t mm2kLoop;
    uint32_t mm2nLoop;
    uint32_t mm2coreLoop;
    uint32_t mm2swizzleCnt;
    uint32_t mm2enShuffleK;
    uint32_t mm2blockDim;

    uint32_t mm3batchSize;
    uint32_t mm3m;
    uint32_t mm3k;
    uint32_t mm3n;
    uint32_t mm3m0;
    uint32_t mm3k0;
    uint32_t mm3n0;
    uint32_t mm3mLoop;
    uint32_t mm3kLoop;
    uint32_t mm3nLoop;
    uint32_t mm3coreLoop;
    uint32_t mm3swizzleCnt;
    uint32_t mm3enShuffleK;
    uint32_t mm3blockDim;

    uint32_t rmsNumCore1;
    uint32_t rmsNumCol1;
    uint32_t rmsNumRow1;
    uint32_t rmsQuantMin1;

    uint32_t rmsNumCore2;
    uint32_t rmsNumCol2;
    uint32_t rmsNumRow2;
    uint32_t rmsQuantMin2;
    
    uint32_t hiddenSizeQ;
    uint32_t headNumQ;
    uint32_t headDim;
    uint32_t concatSize;
    uint32_t rotaryCoeff;
    uint32_t ntokens;
    uint32_t realCore;
    uint32_t nlCoreRun;
    uint32_t lCoreRun;
    uint32_t maxNPerLoopForUb;
    uint32_t preCoreLoopTime;
    uint32_t preCoreLoopNLast;
    uint32_t lastCoreLoopTime;
    uint32_t lastCoreLoopNLast;
};
#endif // ASCEND_OPS_MOE_GMM_TILING_H