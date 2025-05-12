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













#define ATB_CONTEXT_H
#include <acl/acl.h>
#include "atb/types.h"
#include "atb/allocator.h"

//!
//! \file context.h
//!
//! \enum ExecuteType
//!
//! \brief 算子下发类型枚举，通过Context选择加速库算子下发的方式, 支持直接下发和使用分线程两段式下发.
//! \brief 下发接口形式枚举，通过Context选择加速库算子下发接口的形式, 支持单段下发和使用分线程两段式下发.
//!
enum ExecuteType : int {
    EXECUTE_NORMAL = 0,           //!< 直接下发
    EXECUTE_PRELAUNCH,            //!< 用于分线程下发，第一段下发
    EXECUTE_LAUNCH,               //!< 用于分线程下发，第二段下发
    EXECUTE_NORMAL = 0, //!< 直接下发
    EXECUTE_PRELAUNCH,  //!< 用于分线程下发，第一段下发
    EXECUTE_LAUNCH,     //!< 用于分线程下发，第二段下发
};

//!
//! \enum LaunchMode
//!
//! \brief 算子下发模式枚举，通过Context选择算子下发的模式，支持单算子下发与整图下发
//!
enum LaunchMode : int {
    KERNEL_LAUNCH_MODE = 0, //!< 单算子下发模式
    GRAPH_LAUNCH_MODE       //!< 整图下发模式
};

//!
    //!
    //! \return 获取到的ExecuteType类型
    virtual ExecuteType GetExecuteType() = 0;

    //!
    //! \brief 设置算子下发模式
    //!
    //! \param mode 算子下发的模式类型
    //!
    //! \return 状态值，如果设置成功，返回NO_ERROR
    virtual Status SetLaunchMode(LaunchMode mode) = 0;

    //!
    //! \brief 返回当前的算子下发模式
    //!
    //! \return 当前的算子下发模式
    virtual LaunchMode GetLaunchMode() = 0;

    //!
    //! \brief 设置Device侧内存管理类.
    //!
    //! \param allocator 自定义的Device侧内存管理类
    //!
    //! \return 如果设置成功，返回True.
    //!
    virtual Status SetDeviceBufferAllocator(Allocator *allocator) = 0;

    //!
    //! \brief 设置Host侧内存管理类.
    //!
    //! \param allocator 自定义的Host侧内存管理类
    //!
    //! \return 如果设置成功，返回True.
    //!
    virtual Status SetHostBufferAllocator(Allocator *allocator) = 0;
};

//!
src/atb/core/context_base.cpp
CHANGED
Viewed
#include "atb/utils.h"
#include "atb/utils/probe.h"
#include "atb/utils/singleton.h"
#include "atb/core/allocator/default_device_allocator.h"
#include "atb/core/allocator/default_host_allocator.h"

namespace atb {
static constexpr size_t MAX_COPY_EVENT_NUM = 10;

uint8_t *ContextBase::GetHostTilingBuffer()
{
    // 如果走图模式的话直接使用hostAllocator申请内存
    if (mode_ == GRAPH_LAUNCH_MODE) {
        ATB_LOG(INFO) << "At GRAPH_LAUNCH_MODE, contextBase start allocate host tiling buffer using Allocator";
        return reinterpret_cast<uint8_t*>(hostAllocator_->Allocate(TILING_BUFFER_BLOCK_SIZE));
    }
    return hostTilingBufferPool_ ? hostTilingBufferPool_->GetBuffer() : nullptr;
}

uint8_t *ContextBase::GetDeviceTilingBuffer()
{
    // 如果走图模式的话直接使用deviceAllocator申请内存
    if (mode_ == GRAPH_LAUNCH_MODE) {
        ATB_LOG(INFO) << "At GRAPH_LAUNCH_MODE, contextBase start allocate device tiling buffer using Allocator";
        return reinterpret_cast<uint8_t*>(deviceAllocator_->Allocate(TILING_BUFFER_BLOCK_SIZE));
    }
    return deviceTilingBufferPool_ ? deviceTilingBufferPool_->GetBuffer() : nullptr;
}

    return executeType_;
}

Status ContextBase::SetLaunchMode(LaunchMode mode)
{
    if (mode > GRAPH_LAUNCH_MODE || mode < KERNEL_LAUNCH_MODE) {
        ATB_LOG(ERROR) << "LaunchMode set error! mode should in enum range.";
        return ERROR_INVALID_PARAM;
    }
    mode_ = mode;
    return NO_ERROR;
}

LaunchMode ContextBase::GetLaunchMode()
{
    return mode_;
}

Status ContextBase::SetDeviceBufferAllocator(Allocator *allocator)
{
    if (allocator == nullptr) {
        ATB_LOG(ERROR) << "allocator is nullptr";
        return ERROR_INVALID_PARAM;
    }
    // 如果中途切换Allocator的话使用旧的Allocator管理的内存全部释放
    ATB_LOG(WARN) << "Changing to the new Allocator will free all device buffers,"
                  << "which allocated by the old Allocator.";
    deviceAllocator_.reset(allocator);
    return NO_ERROR;
}

void *ContextBase::GetArgsDeviceBuffer(size_t bufferSize)
{
    return deviceAllocator_->Allocate(bufferSize);
}

Status ContextBase::FreeArgsDeviceBuffer(void *addr)
{
    return deviceAllocator_->Deallocate(addr);
}

Status ContextBase::SetHostBufferAllocator(Allocator *allocator)
{
    if (allocator == nullptr) {
        ATB_LOG(ERROR) << "allocator is nullptr";
        return ERROR_INVALID_PARAM;
    }
    // 如果中途切换Allocator的话使用旧的Allocator管理的内存全部释放
    ATB_LOG(WARN) << "Changing to the new Allocator will free all host buffers, which allocated by the old Allocator.";
    hostAllocator_.reset(allocator);
    return NO_ERROR;
}

void *ContextBase::GetArgsHostBuffer(size_t bufferSize)
{
    return hostAllocator_->Allocate(bufferSize);
}

Status ContextBase::FreeArgsHostBuffer(void *addr)
{
    return hostAllocator_->Deallocate(addr);
}
} // namespace atb
src/atb/runner/ops_runner.cpp
CHANGED
Viewed
Status OpsRunner::FillSingleKernelHostTilingBuffer(KernelGraphNode &node, size_t nodeId,
                                                   uint8_t *kernelHostTilingBuffer, size_t tilingSize)
{
    if (node.impl->GetTilingFilledFlag()) {
    if (node.impl->GetTilingFilledFlag() && !needKernelGraphModify_) {
        return NO_ERROR;
    }

src/include/atb/core/context_base.h
CHANGED
Viewed
#include <memory>
#include "atb/context.h"
#include "atb/svector.h"
#include "atb/allocator.h"
#include "atb/core/tiling_buffer_pool/tiling_buffer_pool.h"
#include "atb/core/runner_type.h"
#include "atb/core/runner_pool.h"
    const Tensor &GetOverflowKernelOutTensor();
    Status SetExecuteType(ExecuteType type) override;
    ExecuteType GetExecuteType() override;
    Status SetLaunchMode(LaunchMode mode) override;
    LaunchMode GetLaunchMode() override;
    Status SetDeviceBufferAllocator(Allocator *allocator) override;
    Status SetHostBufferAllocator(Allocator *allocator) override;
    void *GetArgsDeviceBuffer(size_t bufferSize);
    void *GetArgsHostBuffer(size_t bufferSize);
    Status FreeArgsDeviceBuffer(void *addr);
    Status FreeArgsHostBuffer(void *addr);

private:
    Status CreateCopyStreamAndEvents();
    std::vector<RunnerPool> runnerPools_;
    Tensor overflowOutTensor_;
    static thread_local ExecuteType executeType_;
    LaunchMode mode_ = KERNEL_LAUNCH_MODE;
    std::unique_ptr<Allocator> deviceAllocator_;  // 一开始就赋值为defaultDeviceAllocator
    std::unique_ptr<Allocator> hostAllocator_;  // 一开始就赋值为defaultHostAllocator
};
} // namespace atb
#endif