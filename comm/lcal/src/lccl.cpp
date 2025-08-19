/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lccl.h"
#include "lcal_internal.h"

#include <chrono>
#include <mutex>
#include <thread>
#include <acl/acl.h>

#include <mki/utils/log/log.h>
#include <mki/utils/env/env.h>

#include "profiling/report_timing.h"

using namespace std;
using namespace chrono;
using namespace Mki;

namespace Lcal {

uint32_t GetLocalReduceBlockDum(int64_t dataSize)
{
    constexpr int oneDataSize = 190 * 1024;
    constexpr int maxBlockDim = 8;
    int blockDim = dataSize / oneDataSize + 1;
    return blockDim <= maxBlockDim ? blockDim : maxBlockDim;
}

bool GetParallel()
{
    static int parallel = -1;
    if (parallel == -1) {
        static const char *ENV = Mki::GetEnv("LCCL_PARALLEL");
        parallel = (ENV && (string(ENV) == "1" || string(ENV) == "true")) ? 1 : 0;
        MKI_LOG(INFO) << "LCCL_PARALLEL is " << parallel;
    }
    return static_cast<bool>(parallel);
}

uint32_t GetAllReduceDetermBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t quickOneshotRankSize = 2;
    constexpr uint32_t twoBlockNum = 2;
    constexpr uint32_t treeBlockNum = 3;
    constexpr uint32_t rankSize910a3 = 16;
    constexpr uint32_t dbRingBlockNum = 34;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024;
    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) {
        
    }

    uint32_t blockDim = GetLocalReduceBlockDum(dataSize);
    return (rankSize + blockDim - 1) / blockDim;
}




}