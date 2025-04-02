/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MLA_ATTENTION_H
#define MLA_ATTENTION_H

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

struct MLATilingData {
    uint32_t numCore;
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
    uint32_t mm1swizzlDirect;
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
    uint32_t mm2swizzlDirect;
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
    uint32_t mm3swizzlDirect;
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

#endif