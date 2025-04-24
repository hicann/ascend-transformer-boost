/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/core/tiling_buffer_pool/device_tiling_buffer_pool.h"
#include "atb/utils/log.h"

namespace atb {
DeviceTilingBufferPool::DeviceTilingBufferPool(uint64_t blockNum, uint64_t blockSize)
    : TilingBufferPool(blockNum, blockSize)
{
}

DeviceTilingBufferPool::~DeviceTilingBufferPool() {}

uint8_t *DeviceTilingBufferPool::MallocTotalBuffer(uint64_t bufferSize)
{
    void *buffer = nullptr;
    ATB_LOG(INFO) << "aclrtMalloc bufferSize:" << bufferSize;
    int ret = aclrtMalloc(&buffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ATB_LOG_IF(ret != 0, ERROR) << "aclrtMalloc fail, bufferSize:" << bufferSize << ", ret:" << ret;
    return static_cast<uint8_t *>(buffer);
}

void DeviceTilingBufferPool::FreeTotalBuffer(uint8_t *buffer)
{
    if (buffer) {
        aclError aclRet = aclrtFree(buffer);
        if (aclRet != ACL_SUCCESS) {
            ATB_LOG(ERROR) << "Free total buffer failed! ret: " << aclRet;
        }
    }

    buffer = nullptr;
}

bool DeviceTilingBufferPool::IsDeviceBufferPool()
{
    return true;
}
} // namespace atb