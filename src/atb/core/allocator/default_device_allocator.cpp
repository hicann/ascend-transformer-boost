/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/core/allocator/default_device_allocator.h"
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
namespace atb {
const int ALIGN_INT = 32;
DefaultDeviceAllocator::DefaultDeviceAllocator() {}
DefaultDeviceAllocator::~DefaultDeviceAllocator()
{
    // 释放所有管理的device侧地址
    for (auto it = memMap.begin(); it != memMap.end(); ++it) {
        ATB_LOG(INFO) << "DefaultDeviceAllocator::~DefaultDeviceAllocator aclrtFree free device buffer: " << it->first;
        Status st = aclrtFree(it->first);
        if (st != 0) {
            ATB_LOG(ERROR) << "aclrtFree device buffer failed!";
        }
    }
}

void *DefaultDeviceAllocator::Allocate(size_t bufferSize)
{
    if (bufferSize == 0) {
        ATB_LOG(ERROR) << "bufferSize can not be 0, please check the bufferSize";
        return nullptr;
    }
    void *addr = nullptr;
    bufferSize = TensorUtil::AlignInt(bufferSize, ALIGN_INT);
    ATB_LOG(INFO) << "bufferSize should be 32-bit alignment, automate align upwards to " << bufferSize;
    // aclrtMalloc会自动对于bufferSize+32，不论bufferSize是否是32的整数倍
    // aclrtMallocAlign32 只会对申请的bufferSize向上取整，不会再加32字节
    Status st = aclrtMalloc(&addr, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (st != 0) {
        ATB_LOG(ERROR) << "aclrtMalloc device buffer failed!";
        return nullptr;
    }
    currentAllocateSize_ += bufferSize;
    memMap.insert(std::make_pair(addr, bufferSize));
    ATB_LOG(INFO) << "DefaultDeviceAllocator::Allocate device buffer succes, currentAllocateSize_: "
                  << currentAllocateSize_
                  << " deviceBuffer: " << addr;
    return addr;
}

Status DefaultDeviceAllocator::Deallocate(void *addr)
{
    if (addr == nullptr) {
        ATB_LOG(INFO) << "the addr is nullptr, do not need to deallocate";
        return NO_ERROR;
    }
    auto it = memMap.find(addr);
    if (it == memMap.end()) {
        ATB_LOG(ERROR) << "free fail, can not find the address please check the address is made by allocator";
        return ERROR_RT_FAIL;
    }
    // Free
    Status st = aclrtFree(addr);
    if (st != 0) {
        ATB_LOG(ERROR) << "aclrtFree device buffer failed!";
        return ERROR_RT_FAIL;
    }
    currentAllocateSize_ -= it->second;
    memMap.erase(addr);
    ATB_LOG(INFO) << "DefaultDeviceAllocator::Deallocate success, free bufferSize: "<< it->second
                  << ", currentAllocateSize_: " << currentAllocateSize_
                  << " deviceBuffer: " << addr;
    return NO_ERROR;
}
} // namespace atb